#!/usr/bin/env python3
"""
scripts/verify_deposition_tree.py
=================================
Pre-upload gate for the BioImage Archive deposition.

Runs one pass over the directories that will be uploaded and answers four
questions, all of which must be clean before 54.70 GB moves over FTP:

  1. Does every file's extension agree with its magic bytes?
     (Project bug #8: `River/input/dapi/*.tif` were PNG bytes under a .tif
     name. A repeat of that in the archive tree would ship a mislabelled
     dataset that no downstream reader can open by extension.)

  2. Does every file and directory name obey the EBI character rules?
     Allowed: a-z A-Z 0-9 and  ! - _ . * ' ( )  and space.
     Anything else fails BioStudies validation *after* the upload, not
     before it.  Trailing spaces are trimmed by EBI and so are flagged too.

  3. Does the census reconcile with release/manifests/manifest_v1.0.csv?
     Files on disk but not in the manifest are unhashed; files in the
     manifest but not on disk have been moved or deleted since hashing.

  4. What is the exact inventory?  Written to --out as a CSV, one row per
     file, with the *upload-relative* path that will appear in the file
     lists.  This CSV is the input to the file-list generator.

Nothing is written except the inventory CSV.  The tree is read-only to
this script.

Usage
-----
    python scripts/verify_deposition_tree.py \
        --root images=/proj/.../ecDNA_Data/bioimage_archive \
        --root predictions=/proj/.../ecDNA_Data/benchmark/predictions \
        --manifest /proj/.../ecdna-bench/release/manifests/manifest_v1.0.csv \
        --out /proj/.../ecdna-bench/release/manifests/deposition_inventory.csv

`--root LABEL=PATH` may be repeated.  LABEL becomes the top-level directory
name on the BioStudies FTP server, and therefore the first path component in
every file-list entry.  Choose labels that obey the character rules above.

Manifest reconciliation matches on `<parent directory name>/<filename>`
rather than on the full relative path, because the manifest was generated
against a different root than the upload staging labels.  Every directory
basename in this tree is unique, so that key is unambiguous.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

# --------------------------------------------------------------------------
# EBI file/directory name rules
# https://www.ebi.ac.uk/bioimage-archive/help-file-list/#namerules
# --------------------------------------------------------------------------

ALLOWED_SPECIAL = set("!-_.*'() ")


def name_problems(name: str) -> list[str]:
    """Return a list of reasons `name` violates the EBI rules (empty = fine)."""
    problems: list[str] = []
    bad = sorted({c for c in name if not (c.isalnum() and c.isascii()) and c not in ALLOWED_SPECIAL})
    if bad:
        problems.append("disallowed characters: " + " ".join(repr(c) for c in bad))
    if name != name.rstrip():
        problems.append("trailing whitespace")
    if " " in name:
        problems.append("contains a space (allowed, but avoid)")
    return problems


# --------------------------------------------------------------------------
# Magic-byte detection
# --------------------------------------------------------------------------

MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"II+\x00", "BIGTIFF"),
    (b"MM\x00+", "BIGTIFF"),
    (b"\x93NUMPY", "NPY"),
    (b"PK\x03\x04", "ZIP"),          # .npz is a zip container
    (b"\xff\xd8\xff", "JPEG"),
    (b"%PDF", "PDF"),
]

# Which container each extension is expected to be.  ZIP covers .npz.
EXPECTED = {
    ".png": {"PNG"},
    ".tif": {"TIFF", "BIGTIFF"},
    ".tiff": {"TIFF", "BIGTIFF"},
    ".npy": {"NPY"},
    ".npz": {"ZIP"},
    ".jpg": {"JPEG"},
    ".jpeg": {"JPEG"},
    ".pdf": {"PDF"},
    ".csv": {"TEXT"},
    ".tsv": {"TEXT"},
    ".txt": {"TEXT"},
    ".md": {"TEXT"},
    ".json": {"TEXT"},
    ".yaml": {"TEXT"},
    ".yml": {"TEXT"},
    ".pt": {"ZIP"},                  # torch checkpoints are zip archives
    ".pth": {"ZIP"},
}


def detect_magic(path: Path) -> str:
    """Identify a file by its leading bytes.  Returns TEXT, UNKNOWN or a format."""
    try:
        with open(path, "rb") as f:
            head = f.read(16)
    except OSError as exc:                                   # pragma: no cover
        return f"UNREADABLE({exc.__class__.__name__})"
    if not head:
        return "EMPTY"
    for sig, name in MAGIC:
        if head.startswith(sig):
            return name
    # Heuristic for text: no NUL byte and decodes as UTF-8.
    if b"\x00" not in head:
        try:
            head.decode("utf-8")
            return "TEXT"
        except UnicodeDecodeError:
            pass
    return "UNKNOWN"


# --------------------------------------------------------------------------
# Manifest loading
# --------------------------------------------------------------------------

def load_manifest_keys(manifest: Path) -> set[str]:
    """Return {'<parent dir>/<filename>'} for every row of the manifest."""
    with open(manifest, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"Manifest has no header: {manifest}")
        # The project has shipped both spellings; accept either.
        col = next(
            (c for c in ("relative_path", "relpath", "path") if c in reader.fieldnames),
            None,
        )
        if col is None:
            raise SystemExit(
                f"Manifest {manifest} has none of relative_path / relpath / path. "
                f"Columns present: {reader.fieldnames}"
            )
        keys = set()
        for row in reader:
            raw = (row.get(col) or "").strip()
            if not raw:
                continue
            p = Path(raw.replace("\\", "/"))
            keys.add(f"{p.parent.name}/{p.name}")
    return keys


# --------------------------------------------------------------------------
# Main scan
# --------------------------------------------------------------------------

def scan(roots: list[tuple[str, Path]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, root in roots:
        root = root.resolve()
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        print(f"scanning {label} -> {root}", file=sys.stderr)
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(root).as_posix()
            upload_rel = f"{label}/{rel}"
            ext = p.suffix.lower()
            magic = detect_magic(p)
            expected = EXPECTED.get(ext)
            if expected is None:
                format_ok = ""            # no expectation registered
            else:
                format_ok = "yes" if magic in expected else "NO"
            problems: list[str] = []
            for part in Path(upload_rel).parts:
                for msg in name_problems(part):
                    problems.append(f"{part}: {msg}")
            rows.append(
                {
                    "upload_relpath": upload_rel,
                    "root_label": label,
                    "top_dir": rel.split("/")[0] if "/" in rel else "(root)",
                    "filename": p.name,
                    "uid": p.stem,
                    "ext": ext,
                    "size_bytes": p.stat().st_size,
                    "magic": magic,
                    "format_ok": format_ok,
                    "name_problems": "; ".join(problems),
                    "manifest_key": f"{p.parent.name}/{p.name}",
                    "abs_path": str(p),
                }
            )
    return rows


def report(rows: list[dict[str, object]], manifest: Path | None) -> int:
    failures = 0

    # ---- census -----------------------------------------------------------
    census: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        key = (str(r["root_label"]), str(r["top_dir"]), str(r["ext"]))
        census[key][0] += 1
        census[key][1] += int(r["size_bytes"])  # type: ignore[arg-type]

    print("\n=== CENSUS ===")
    print(f"{'root':<12}{'directory':<26}{'ext':<8}{'files':>8}{'size':>12}")
    total_files = 0
    total_bytes = 0
    for (label, top, ext), (n, nbytes) in sorted(census.items()):
        print(f"{label:<12}{top:<26}{ext:<8}{n:>8}{nbytes / 1e9:>10.2f} GB")
        total_files += n
        total_bytes += nbytes
    print(f"{'TOTAL':<46}{total_files:>8}{total_bytes / 1e9:>10.2f} GB")
    print(f"{'':<46}{'':>8}{total_bytes / 2**30:>10.2f} GiB")

    # ---- format mismatches ------------------------------------------------
    bad_format = [r for r in rows if r["format_ok"] == "NO"]
    unregistered = sorted({str(r["ext"]) for r in rows if r["format_ok"] == ""})
    print("\n=== FORMAT vs EXTENSION ===")
    if bad_format:
        failures += len(bad_format)
        print(f"MISMATCHES: {len(bad_format)}")
        counts = Counter((str(r["ext"]), str(r["magic"])) for r in bad_format)
        for (ext, magic), n in counts.most_common():
            print(f"  {ext} files containing {magic}: {n}")
        for r in bad_format[:10]:
            print(f"    {r['upload_relpath']}  ({r['magic']})")
        if len(bad_format) > 10:
            print(f"    ... and {len(bad_format) - 10} more (see the inventory CSV)")
    else:
        print("clean — every registered extension matches its magic bytes")
    if unregistered:
        print(f"extensions with no registered expectation (not checked): {unregistered}")

    # ---- name rules -------------------------------------------------------
    named_bad = [r for r in rows if r["name_problems"]]
    hard_bad = [r for r in named_bad if "disallowed" in str(r["name_problems"])
                or "trailing" in str(r["name_problems"])]
    print("\n=== EBI NAME RULES ===")
    if hard_bad:
        failures += len(hard_bad)
        print(f"BLOCKING violations: {len(hard_bad)}")
        for r in hard_bad[:10]:
            print(f"  {r['upload_relpath']}  [{r['name_problems']}]")
        if len(hard_bad) > 10:
            print(f"  ... and {len(hard_bad) - 10} more")
    else:
        print("clean — no disallowed characters, no trailing whitespace")
    soft_bad = [r for r in named_bad if r not in hard_bad]
    if soft_bad:
        print(f"advisory (spaces in names, permitted but better avoided): {len(soft_bad)}")

    # ---- manifest reconciliation -----------------------------------------
    print("\n=== MANIFEST RECONCILIATION ===")
    if manifest is None:
        print("skipped (no --manifest given)")
    else:
        man = load_manifest_keys(manifest)
        disk = {str(r["manifest_key"]) for r in rows}
        only_disk = sorted(disk - man)
        only_man = sorted(man - disk)
        print(f"manifest rows (unique keys): {len(man)}")
        print(f"files on disk (unique keys): {len(disk)}")
        print(f"in both:                     {len(disk & man)}")
        print(f"on disk, NOT hashed:         {len(only_disk)}")
        print(f"hashed, NOT on disk:         {len(only_man)}")
        for k in only_disk[:15]:
            print(f"  + {k}")
        if len(only_disk) > 15:
            print(f"  ... and {len(only_disk) - 15} more")
        for k in only_man[:15]:
            print(f"  - {k}")
        if len(only_man) > 15:
            print(f"  ... and {len(only_man) - 15} more")
        if only_man:
            failures += len(only_man)

    # ---- upload-method guidance ------------------------------------------
    largest = max(rows, key=lambda r: int(r["size_bytes"]))  # type: ignore[arg-type]
    print("\n=== UPLOAD METHOD ===")
    gb = total_bytes / 1e9
    print(f"total {gb:.2f} GB; largest single file "
          f"{int(largest['size_bytes']) / 1e6:.1f} MB ({largest['upload_relpath']})")
    if gb < 50:
        print("-> under 50 GB: the browser submission tool would work")
    elif gb < 1000:
        print("-> between 50 GB and 1 TB: use FTP")
    else:
        print("-> above 1 TB: use Aspera")

    print("\n=== VERDICT ===")
    if failures:
        print(f"NOT CLEAR TO UPLOAD — {failures} blocking issue(s) above")
    else:
        print("CLEAR TO UPLOAD")
    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", action="append", required=True, metavar="LABEL=PATH",
                    help="Upload root. Repeatable. LABEL becomes the top-level "
                         "directory on the FTP server.")
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest_v1.0.csv, for presence reconciliation.")
    ap.add_argument("--out", type=Path, required=True,
                    help="Inventory CSV to write.")
    args = ap.parse_args()

    roots: list[tuple[str, Path]] = []
    for spec in args.root:
        if "=" not in spec:
            raise SystemExit(f"--root must be LABEL=PATH, got: {spec}")
        label, _, path = spec.partition("=")
        label = label.strip()
        if name_problems(label):
            raise SystemExit(f"Root label {label!r} breaks the EBI name rules.")
        roots.append((label, Path(path.strip())))

    rows = scan(roots)
    if not rows:
        raise SystemExit("No files found.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["upload_relpath", "root_label", "top_dir", "filename", "uid", "ext",
              "size_bytes", "magic", "format_ok", "name_problems",
              "manifest_key", "abs_path"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\ninventory written: {args.out}  ({len(rows)} rows)", file=sys.stderr)

    failures = report(rows, args.manifest)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
