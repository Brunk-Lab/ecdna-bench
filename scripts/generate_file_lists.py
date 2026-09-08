#!/usr/bin/env python3
"""
scripts/generate_file_lists.py
==============================
Build the four BioImage Archive file lists for the ecdna-bench deposition.

A file list is the "table of contents" of a submission section: one row per
file, saying what that file is.  EBI requires:

  * TSV or .xlsx
  * the first column header is the literal word "Files"
  * one file per line, no blank lines
  * forward-slash path separators, relative to the submission root
  * annotation sections additionally carry a "source_image" column naming the
    image each annotation describes

Reference: https://www.ebi.ac.uk/bioimage-archive/help-file-list/

Four sections are produced:

  filelist_images.tsv        Study component. RGB and DAPI images, plus the
                             image index and the split lists as supporting
                             files.
  filelist_gt.tsv            Annotation. Ground-truth ecDNA annotations in
                             three representations.
  filelist_roi.tsv           Annotation. Manual metaphase-spread ROI masks.
  filelist_predictions.tsv   Annotation. Prediction masks from the
                             benchmarked methods.
  filelist_predicted_roi.tsv Annotation. Model-predicted metaphase-spread ROI
                             masks for all 2,986 images. Optional: emitted
                             only when an inventory covering them is passed.

Every list is validated after writing, against the structural rules EBI's
parser enforces and against one rule it does not: that every `source_image`
value resolves to a file that is actually present in the study-component
list. A dangling source_image passes validation at submission and then breaks
the archive's image-to-annotation linking silently, which is the expensive
kind of failure.

Design note on columns.  EBI advises including only attributes that take at
least two distinct values across the files in a given list; anything constant
belongs in the study description entered on the web form, not repeated 2,986
times.  That is why image dimensions (constant at 2448 x 2048 px), organism
and imaging modality do not appear here, and why "Subset" is present in the
image list but absent from the ROI list, where every file is a benchmark
image.

Usage
-----
    python scripts/generate_file_lists.py \
        --inventory  release/manifests/deposition_inventory.csv \
        --counts     release/figures/notebook05/source_figS_gt_count_disclosure_long.csv \
        --split-dir  release/split_files \
        --out-dir    release/deposition/file_lists
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

# --------------------------------------------------------------------------
# Vocabulary
# --------------------------------------------------------------------------

# Directory -> human-readable data type, for the study-component list.
IMAGE_TYPE = {
    "rgb": "RGB FISH image",
    "dapi": "DAPI image",
}

# Directory -> annotation type, for the ground-truth annotation list.
GT_TYPE = {
    "gt_image": "Segmentation mask (binary, 8-bit PNG, values 0 and 255)",
    "gt_coords": "Point coordinates (NumPy array)",
    "gt_npz": "Sparse segmentation mask (compressed NumPy archive)",
}

# Prediction directory -> canonical method name.
#
# The six benchmarked methods use the locked names from the manuscript and
# must not be respelled.  `classical_default` is the classical pipeline
# *before* Bayesian optimisation; it is released for the before/after
# comparison and has no locked name, so the spelling below is provisional.
METHOD = {
    "eccount_peaks": "ecCount (peaks)",
    "eccount_threshold": "ecCount (threshold mask)",
    "label_engine": "Label Engine",
    "mia": "MIA",
    "classical_optimised": "Classic (after opt)",
    "ecseg": "ecSeg",
    # Confirmed by E. Brunk, 2 September 2026. The before-optimisation
    # pipeline is not one of the six benchmarked methods and so has no name in
    # the manuscript's locked list; this spelling is the agreed parallel form.
    "classical_default": "Classic (before opt)",
}

# Supporting (non-image) files and how to describe them.
SUPPORTING = {
    "metadata_internal.csv": "Image index: per-image cell line, condition and modality filenames",
    "train_ids.csv": "Split definition: training set unique identifiers",
    "val_ids.csv": "Split definition: validation set unique identifiers",
    "test_ids.csv": "Split definition: test set unique identifiers",
}

# `p13` is a passage number, not a treatment. Left unmapped it would produce a
# condition value held by a single image. Folded into its parent condition.
CONDITION_OVERRIDE = {
    "sum159pt_dc_ctrl_p13_2_1": "dc_ctrl",
}


def condition_from_uid(uid: str) -> str:
    """
    Recover the experimental condition encoded in a unique identifier.

    A UID is `<cell line>_<condition tokens>_<image index>[_<qualifier>]`.
    The condition is everything between the cell-line prefix and the image
    index.  The index is located as the *last* purely numeric token, which
    correctly leaves earlier numeric tokens in place -- the `0223` in
    `facs_fish_0223_high_her2` is a date, not an index.  A handful of UIDs
    have a malformed final token instead of a clean number (`22tif`, `29f`);
    those are stripped by the fallback.
    """
    if uid in CONDITION_OVERRIDE:
        return CONDITION_OVERRIDE[uid]
    rest = uid.split("_")[1:]
    numeric = [i for i, t in enumerate(rest) if t.isdigit()]
    if numeric:
        rest = rest[: numeric[-1]]
    elif len(rest) > 1 and re.fullmatch(r"\d+\w*", rest[-1]):
        rest = rest[:-1]
    return "_".join(rest)


# --------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------

def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_ids(path: Path) -> set[str]:
    """Read a split file. Handles the CRLF line endings these files carry."""
    out = set()
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            uid = (row.get("unique_id") or "").strip()
            if uid:
                out.add(uid)
    return out


def write_tsv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    """
    Write a file list.

    `csv.writer` with a tab delimiter and QUOTE_MINIMAL would quote any field
    containing a tab; none of ours do, and EBI's parser does not expect
    quoting, so fields are asserted tab-free and written raw.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        f.write("\t".join(columns) + "\n")
        for r in rows:
            values = [str(r.get(c, "")) for c in columns]
            for v in values:
                if "\t" in v or "\n" in v:
                    raise ValueError(f"field contains a tab or newline: {v!r}")
            f.write("\t".join(values) + "\n")



# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

ALLOWED_PATH = set("!-_.*'()/ ")


def validate(out_dir: Path) -> int:
    """
    Check every written list against the EBI structural rules, plus
    source_image integrity. Returns the number of problems found.
    """
    problems: list[str] = []
    image_files: set[str] = set()

    lists = [p.name for p in sorted(out_dir.glob("filelist_*.tsv"))]
    if "filelist_images.tsv" not in lists:
        return 1

    for name in lists:
        path = out_dir / name
        raw = open(path, newline="").read()
        lines = raw.split("\n")

        if not raw.endswith("\n") or lines[-1] != "":
            problems.append(f"{name}: must end with exactly one newline")
        body = [l for l in lines[:-1]]

        if any(not l.strip() for l in body):
            problems.append(f"{name}: contains a blank line")

        header = body[0].split("\t")
        if header[0] != "Files":
            problems.append(f"{name}: first header column is {header[0]!r}, must be 'Files'")
        if name != "filelist_images.tsv" and "source_image" not in header:
            problems.append(f"{name}: annotation list has no source_image column")

        ragged = [i for i, l in enumerate(body[1:], start=2)
                  if len(l.split("\t")) != len(header)]
        if ragged:
            problems.append(f"{name}: {len(ragged)} ragged row(s), first at line {ragged[0]}")

        rows = list(csv.DictReader(open(path, newline=""), delimiter="\t"))
        paths = [r["Files"] for r in rows]
        if len(set(paths)) != len(paths):
            dup = [p for p, n in Counter(paths).items() if n > 1]
            problems.append(f"{name}: {len(dup)} duplicate Files entry/entries, e.g. {dup[0]}")

        for p in paths:
            if "\\" in p or p.startswith(("/", "./", "../")) or p.endswith("/"):
                problems.append(f"{name}: illegal path form: {p}")
                break
            if not all((c.isalnum() and c.isascii()) or c in ALLOWED_PATH for c in p):
                problems.append(f"{name}: disallowed character in path: {p}")
                break

        if name == "filelist_images.tsv":
            image_files = set(paths)

    # source_image integrity, once the image list is known
    for name in [n for n in lists if n != "filelist_images.tsv"]:
        rows = list(csv.DictReader(open(out_dir / name, newline=""), delimiter="\t"))
        srcs = {r["source_image"] for r in rows}
        missing = sorted(srcs - image_files)
        if missing:
            problems.append(f"{name}: {len(missing)} source_image value(s) absent "
                            f"from filelist_images.tsv, e.g. {missing[0]}")

    print()
    if problems:
        print("=== VALIDATION FAILED ===")
        for p in problems:
            print(f"  {p}")
    else:
        print("=== VALIDATION PASSED ===")
        print("  structure, headers, paths and source_image links all clean")
    return len(problems)


# --------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inventory", type=Path, action="append", required=True,
                    metavar="PATH",
                    help="Inventory CSV from verify_deposition_tree.py. "
                         "Repeatable: pass the main inventory and, if the "
                         "predicted ROI masks are being deposited, theirs too.")
    ap.add_argument("--counts", type=Path, required=True,
                    help="Full-resource per-image ecDNA counts (2,986 rows).")
    ap.add_argument("--split-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    inv: list[dict[str, str]] = []
    for path in args.inventory:
        inv.extend(read_csv(path))
    counts_rows = read_csv(args.counts)

    counts = {r["unique_id"]: r["ecDNA_gt"] for r in counts_rows}
    cell_line = {r["unique_id"]: r["cell_line"] for r in counts_rows}

    splits: dict[str, str] = {}
    for name in ("train", "val", "test"):
        for uid in read_ids(args.split_dir / f"{name}_ids.csv"):
            splits[uid] = name

    benchmark = {r["uid"] for r in inv if r["top_dir"] == "roi_mask"}

    # ---- guards ----------------------------------------------------------
    # Silent partial coverage is the failure mode that matters here: a file
    # list that quietly omits an attribute for some rows still validates.
    assert len(counts) == 2986, f"expected 2,986 count rows, got {len(counts)}"
    assert len(splits) == 1145, f"expected 1,145 split assignments, got {len(splits)}"
    assert len(benchmark) == 1145, f"expected 1,145 benchmark UIDs, got {len(benchmark)}"
    assert benchmark == set(splits), "benchmark UIDs and split UIDs disagree"

    def common(uid: str) -> dict[str, str]:
        return {
            "Cell Line": cell_line[uid],
            "Condition": condition_from_uid(uid),
            "Subset": "benchmark" if uid in benchmark else "extension",
            "Split": splits.get(uid, ""),
            "ecDNA Count": counts[uid],
        }

    images: list[dict[str, str]] = []
    gt: list[dict[str, str]] = []
    roi: list[dict[str, str]] = []
    pred: list[dict[str, str]] = []
    pred_roi: list[dict[str, str]] = []
    unrouted: list[str] = []

    for r in inv:
        rel, top, uid, fname = r["upload_relpath"], r["top_dir"], r["uid"], r["filename"]
        src = f"images/rgb/{uid}.tif"

        # Predicted ROI masks sit at the root of their own upload directory,
        # so they are routed on root_label rather than on a subdirectory name.
        if r["root_label"] == "predicted_roi":
            c = common(uid)
            pred_roi.append({"Files": rel, "source_image": src,
                             "Annotation Type": "Region of interest mask "
                                                "(metaphase spread, binary)",
                             "Annotation Method": "Automated prediction",
                             **c})
        elif fname in SUPPORTING:
            images.append({"Files": rel, "Data Type": SUPPORTING[fname]})
        elif top in IMAGE_TYPE:
            images.append({"Files": rel, "Data Type": IMAGE_TYPE[top], **common(uid)})
        elif top in GT_TYPE:
            gt.append({"Files": rel, "source_image": src,
                       "Annotation Type": GT_TYPE[top],
                       "Annotation Method": "Manual", **common(uid)})
        elif top == "roi_mask":
            c = common(uid)
            roi.append({"Files": rel, "source_image": src,
                        "Annotation Type": "Region of interest mask "
                                           "(metaphase spread, binary)",
                        "Annotation Method": "Manual",
                        "Cell Line": c["Cell Line"], "Condition": c["Condition"],
                        "Split": c["Split"]})
        elif top in METHOD:
            c = common(uid)
            pred.append({"Files": rel, "source_image": src,
                         "Annotation Type": "Segmentation mask (binary)",
                         "Annotation Method": "Automated prediction",
                         "Method": METHOD[top],
                         "Cell Line": c["Cell Line"], "Condition": c["Condition"],
                         "Split": c["Split"]})
        else:
            unrouted.append(rel)

    if unrouted:
        print(f"FATAL: {len(unrouted)} file(s) matched no section:", file=sys.stderr)
        for u in unrouted[:20]:
            print(f"  {u}", file=sys.stderr)
        sys.exit(1)

    # The split CSVs live in splits/ on the server but are not in the local
    # inventory, which scanned only the two data roots. Add them explicitly.
    for name in ("train", "val", "test"):
        fname = f"{name}_ids.csv"
        if not any(r["Files"].endswith(fname) for r in images):
            images.append({"Files": f"splits/{fname}",
                           "Data Type": SUPPORTING[fname]})

    assert len(pred_roi) in (0, 2986), \
        f"expected 0 or 2,986 predicted ROI masks, got {len(pred_roi)}"

    images.sort(key=lambda r: r["Files"])
    gt.sort(key=lambda r: r["Files"])
    roi.sort(key=lambda r: r["Files"])
    pred.sort(key=lambda r: r["Files"])
    pred_roi.sort(key=lambda r: r["Files"])

    out = args.out_dir
    write_tsv(out / "filelist_images.tsv",
              ["Files", "Data Type", "Cell Line", "Condition", "Subset",
               "Split", "ecDNA Count"], images)
    write_tsv(out / "filelist_gt.tsv",
              ["Files", "source_image", "Annotation Type", "Annotation Method",
               "Cell Line", "Condition", "Subset", "Split", "ecDNA Count"], gt)
    write_tsv(out / "filelist_roi.tsv",
              ["Files", "source_image", "Annotation Type", "Annotation Method",
               "Cell Line", "Condition", "Split"], roi)
    write_tsv(out / "filelist_predictions.tsv",
              ["Files", "source_image", "Annotation Type", "Annotation Method",
               "Method", "Cell Line", "Condition", "Split"], pred)

    if pred_roi:
        write_tsv(out / "filelist_predicted_roi.tsv",
                  ["Files", "source_image", "Annotation Type",
                   "Annotation Method", "Cell Line", "Condition", "Subset",
                   "Split", "ecDNA Count"], pred_roi)

    print(f"filelist_images.tsv       {len(images):>6} rows")
    print(f"filelist_gt.tsv           {len(gt):>6} rows")
    print(f"filelist_roi.tsv          {len(roi):>6} rows")
    print(f"filelist_predictions.tsv  {len(pred):>6} rows")
    if pred_roi:
        print(f"filelist_predicted_roi.tsv{len(pred_roi):>6} rows")
    print(f"total                     "
          f"{len(images) + len(gt) + len(roi) + len(pred) + len(pred_roi):>6} rows")

    if validate(out):
        sys.exit(1)


if __name__ == "__main__":
    main()