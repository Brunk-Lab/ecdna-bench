#!/usr/bin/env python3
"""
scripts/validate_file_lists.py
==============================
Pre-submission gate for all FIVE BioImage Archive file lists.

WHY THIS EXISTS
---------------
`scripts/generate_file_lists.py` builds and validates four lists. The fifth,
`filelist_predicted_roi.tsv` (2,986 rows), was generated separately and has
never been through that validator -- in particular it has never had its
`source_image` column checked against `filelist_images.tsv`.

That check is the one that matters. From the generator's own docstring:

    A dangling source_image passes validation at submission and then breaks
    the archive's image-to-annotation linking silently, which is the
    expensive kind of failure.

This script re-runs every structural rule the generator applies, over all five
lists, and adds the cross-list checks a per-list validator cannot make. It
reads only; it writes nothing and modifies nothing.

USAGE
-----
    python scripts/validate_file_lists.py \
        --dir release/deposition/file_lists

    # optional: also confirm every listed path exists in the upload inventory
    python scripts/validate_file_lists.py \
        --dir       release/deposition/file_lists \
        --inventory release/manifests/deposition_inventory.csv \
        --predicted-roi-inventory release/manifests/predicted_roi_inventory.csv

EXIT CODES
----------
    0   every check passed -- clear to submit
    1   the study-component list is absent, so nothing could be cross-checked
    2   one or more validation failures (each is named individually)

TESTED
------
Exercised against synthetic five-list fixtures of the real shape (2,986 UIDs,
1,145 benchmark) under Python 3.10: clean input exits 0; a dangling
source_image, a duplicated source_image, a short row count, a blank line, a
mangled header, a missing source_image column, a disallowed path character, an
empty source_image cell and an absent list each exit non-zero and name the
cause. Requires only the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# The five lists, and what each is expected to contain.
#
# `rows` is the locked row count from the verified upload census
# (27,080 files total). A mismatch here means the list was regenerated
# against a different tree than the one on the FTP server.
# --------------------------------------------------------------------------

STUDY_COMPONENT = "filelist_images.tsv"

EXPECTED = {
    "filelist_images.tsv":        {"rows": 5976, "prefixes": ("images/", "splits/")},
    "filelist_gt.tsv":            {"rows": 8958, "prefixes": ("images/",)},
    "filelist_roi.tsv":           {"rows": 1145, "prefixes": ("images/",)},
    "filelist_predictions.tsv":   {"rows": 8015, "prefixes": ("predictions/",)},
    "filelist_predicted_roi.tsv": {"rows": 2986, "prefixes": ("predicted_roi/",)},
}

TOTAL_ROWS = 27080

# EBI: alphanumerics plus these, and the path separator.
# https://www.ebi.ac.uk/bioimage-archive/help-file-list/
ALLOWED_PATH = set("!-_.*'()/ ")


class Report:
    """Collects problems and notes so everything is printed in one place."""

    def __init__(self) -> None:
        self.problems: list[str] = []
        self.notes: list[str] = []

    def fail(self, msg: str) -> None:
        self.problems.append(msg)

    def note(self, msg: str) -> None:
        self.notes.append(msg)


# --------------------------------------------------------------------------
# Structural checks -- identical in intent to generate_file_lists.py::validate
# --------------------------------------------------------------------------

def check_structure(path: Path, name: str, rep: Report) -> list[dict[str, str]]:
    """Structural rules. Returns the parsed rows (possibly empty on failure)."""
    # NB: open(..., newline="") rather than Path.read_text(newline=...), which
    # only exists on Python 3.13+. The cluster env is 3.10.16.
    with open(path, newline="") as f:
        raw = f.read()
    lines = raw.split("\n")

    if not raw.endswith("\n") or lines[-1] != "":
        rep.fail(f"{name}: must end with exactly one newline")
    body = lines[:-1] if lines and lines[-1] == "" else lines

    if not body:
        rep.fail(f"{name}: file is empty")
        return []

    if any(not l.strip() for l in body):
        rep.fail(f"{name}: contains a blank line")

    header = body[0].split("\t")
    if header[0] != "Files":
        rep.fail(f"{name}: first header column is {header[0]!r}, must be 'Files'")
    if name != STUDY_COMPONENT and "source_image" not in header:
        rep.fail(f"{name}: annotation list has no source_image column")

    ragged = [i for i, l in enumerate(body[1:], start=2)
              if len(l.split("\t")) != len(header)]
    if ragged:
        rep.fail(f"{name}: {len(ragged)} ragged row(s), first at line {ragged[0]}")

    with open(path, newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))

    paths = [r.get("Files", "") for r in rows]

    if len(set(paths)) != len(paths):
        dup = [p for p, n in Counter(paths).items() if n > 1]
        rep.fail(f"{name}: {len(dup)} duplicate Files entry/entries, e.g. {dup[0]}")

    for p in paths:
        if "\\" in p or p.startswith(("/", "./", "../")) or p.endswith("/"):
            rep.fail(f"{name}: illegal path form: {p}")
            break
    for p in paths:
        bad = sorted({c for c in p
                      if not (c.isalnum() and c.isascii()) and c not in ALLOWED_PATH})
        if bad:
            rep.fail(f"{name}: disallowed character(s) {bad} in path: {p}")
            break

    # Empty cells in a required column validate structurally but carry no
    # information; the generator's `common()` guards against this upstream and
    # this is the same guard applied to the finished artefact.
    for col in ("Files", "source_image"):
        if col in header:
            blank = sum(1 for r in rows if not (r.get(col) or "").strip())
            if blank:
                rep.fail(f"{name}: {blank} row(s) with an empty {col}")

    exp = EXPECTED.get(name)
    if exp:
        if len(rows) != exp["rows"]:
            rep.fail(f"{name}: {len(rows)} rows, expected {exp['rows']} "
                     f"(the list disagrees with the verified upload census)")
        off = [p for p in paths if not p.startswith(exp["prefixes"])]
        if off:
            rep.fail(f"{name}: {len(off)} path(s) outside {exp['prefixes']}, "
                     f"e.g. {off[0]}")

    return rows


# --------------------------------------------------------------------------
# Cross-list checks -- the ones a per-list validator cannot make
# --------------------------------------------------------------------------

def check_source_images(all_rows: dict[str, list[dict[str, str]]],
                        rep: Report) -> None:
    """Every source_image must name a file present in the study-component list."""
    image_files = {r["Files"] for r in all_rows.get(STUDY_COMPONENT, [])}
    if not image_files:
        rep.fail("cannot check source_image integrity: "
                 f"{STUDY_COMPONENT} produced no rows")
        return

    for name, rows in all_rows.items():
        if name == STUDY_COMPONENT:
            continue
        srcs = {(r.get("source_image") or "").strip() for r in rows}
        srcs.discard("")
        missing = sorted(srcs - image_files)
        if missing:
            rep.fail(f"{name}: {len(missing)} source_image value(s) absent from "
                     f"{STUDY_COMPONENT}, e.g. {missing[0]}")
        else:
            rep.note(f"{name}: {len(srcs)} distinct source_image value(s), "
                     f"all resolve")


def check_no_collisions(all_rows: dict[str, list[dict[str, str]]],
                        rep: Report) -> None:
    """The same file must not be claimed by two sections."""
    seen: dict[str, str] = {}
    for name, rows in all_rows.items():
        for r in rows:
            p = r.get("Files", "")
            if p in seen and seen[p] != name:
                rep.fail(f"{p} appears in both {seen[p]} and {name}")
                return
            seen[p] = name
    rep.note(f"no path claimed by two sections ({len(seen)} distinct paths)")


def check_total(all_rows: dict[str, list[dict[str, str]]], rep: Report) -> None:
    total = sum(len(v) for v in all_rows.values())
    if total != TOTAL_ROWS:
        rep.fail(f"rows across all lists total {total}, expected {TOTAL_ROWS}")
    else:
        rep.note(f"row total {total} matches the verified upload census")


def check_predicted_roi_coverage(all_rows: dict[str, list[dict[str, str]]],
                                 rep: Report) -> None:
    """
    The predicted-ROI list is the only one that must cover the FULL resource.

    Its stated purpose is that predicted and manual masks can be compared on
    the benchmark images where both exist, so:
      * one row per RGB image in the study component, no more, no fewer;
      * every manual-ROI source_image also present here.
    """
    pred = all_rows.get("filelist_predicted_roi.tsv", [])
    if not pred:
        return

    rgb = {r["Files"] for r in all_rows.get(STUDY_COMPONENT, [])
           if r["Files"].startswith("images/rgb/")}
    src = [(r.get("source_image") or "").strip() for r in pred]

    dup = [s for s, n in Counter(src).items() if n > 1]
    if dup:
        rep.fail(f"filelist_predicted_roi.tsv: {len(dup)} source_image value(s) "
                 f"used more than once, e.g. {dup[0]}")

    uncovered = sorted(rgb - set(src))
    if uncovered:
        rep.fail(f"filelist_predicted_roi.tsv: {len(uncovered)} RGB image(s) have "
                 f"no predicted ROI row, e.g. {uncovered[0]}")
    else:
        rep.note("filelist_predicted_roi.tsv: covers every RGB image exactly once")

    manual = {(r.get("source_image") or "").strip()
              for r in all_rows.get("filelist_roi.tsv", [])}
    gap = sorted(manual - set(src))
    if gap:
        rep.fail(f"filelist_predicted_roi.tsv: {len(gap)} benchmark image(s) have a "
                 f"manual ROI but no predicted ROI, e.g. {gap[0]}")
    elif manual:
        rep.note(f"filelist_predicted_roi.tsv: all {len(manual)} benchmark images "
                 f"carry both a manual and a predicted mask")


def check_inventory(all_rows: dict[str, list[dict[str, str]]],
                    inventories: list[Path], rep: Report) -> None:
    """Optional: every listed path must exist in the upload inventory."""
    on_disk: set[str] = set()
    for inv in inventories:
        with open(inv, newline="") as f:
            reader = csv.DictReader(f)
            col = next((c for c in ("upload_relpath", "Files", "relative_path")
                        if c in (reader.fieldnames or [])), None)
            if col is None:
                rep.fail(f"{inv.name}: no upload_relpath/Files/relative_path column "
                         f"(columns: {reader.fieldnames})")
                return
            on_disk |= {(r.get(col) or "").strip() for r in reader}
    on_disk.discard("")

    for name, rows in all_rows.items():
        listed = {r["Files"] for r in rows}
        # splits/ are added by the generator and are not in the local inventory
        missing = sorted(p for p in listed - on_disk
                         if not p.startswith("splits/"))
        if missing:
            rep.fail(f"{name}: {len(missing)} listed path(s) not in the inventory, "
                     f"e.g. {missing[0]}")
        else:
            rep.note(f"{name}: every listed path present in the inventory")


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True,
                    help="Directory holding the five .tsv file lists.")
    ap.add_argument("--inventory", type=Path, action="append", default=[],
                    dest="inventories", metavar="CSV",
                    help="Optional upload inventory CSV. Repeatable.")
    ap.add_argument("--predicted-roi-inventory", type=Path, default=None,
                    help="Convenience alias for a second --inventory.")
    args = ap.parse_args()

    inventories = list(args.inventories)
    if args.predicted_roi_inventory:
        inventories.append(args.predicted_roi_inventory)

    rep = Report()
    all_rows: dict[str, list[dict[str, str]]] = {}

    print("=== FILE LISTS ===")
    for name in EXPECTED:
        path = args.dir / name
        if not path.is_file():
            print(f"  {name:<30} MISSING")
            rep.fail(f"{name}: not found in {args.dir}")
            continue
        md5 = hashlib.md5(path.read_bytes()).hexdigest()
        rows = check_structure(path, name, rep)
        all_rows[name] = rows
        print(f"  {name:<30} {len(rows):>6} rows   md5 {md5}")

    if STUDY_COMPONENT not in all_rows:
        print("\nCannot continue: the study-component list is missing.")
        return 1

    check_source_images(all_rows, rep)
    check_no_collisions(all_rows, rep)
    check_total(all_rows, rep)
    check_predicted_roi_coverage(all_rows, rep)
    if inventories:
        check_inventory(all_rows, inventories, rep)
    else:
        rep.note("inventory cross-check skipped (no --inventory given)")

    print("\n=== CHECKS ===")
    for n in rep.notes:
        print(f"  ok    {n}")

    print()
    if rep.problems:
        print("=== VALIDATION FAILED ===")
        for p in rep.problems:
            print(f"  {p}")
        print(f"\n{len(rep.problems)} problem(s). DO NOT SUBMIT until these are clear.")
        return 2

    print("=== VALIDATION PASSED ===")
    print("  structure, headers, paths, row counts, source_image links,")
    print("  section exclusivity and predicted-ROI coverage all clean")
    print("  CLEAR TO SUBMIT")
    return 0


if __name__ == "__main__":
    sys.exit(main())
