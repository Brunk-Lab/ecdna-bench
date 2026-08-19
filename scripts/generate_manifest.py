#!/usr/bin/env python3
"""
scripts/generate_manifest.py
============================
Generate a SHA256 manifest for the data release.

WHAT CHANGED (2026-08-18)
    The previous version globbed only '*.png'. That omitted rgb/ (2,986 TIFF,
    ~48 GB), gt_coords/ (.npy), gt_npz/ (.npz) and the metadata CSVs -- i.e.
    the great majority of the deposition by size. It also had no scope
    control, so it descended into drug_treatment/ (a separate project) and
    would have picked up editor and backup files.

    This version hashes EVERY file under the in-scope directories, excludes
    by explicit glob, and derives file_type from the path relative to the
    data root rather than by substring-matching the absolute path.

Usage
-----
    # list what would be hashed, without hashing (fast -- check scope first)
    python scripts/generate_manifest.py --data-root /path/to/ecDNA_Data --list

    # write the manifest
    python scripts/generate_manifest.py \
        --data-root /path/to/ecDNA_Data \
        --out release/manifests/manifest_v1.0.csv

Columns: relative_path, sha256, size_bytes, file_type

NOTE
    Run this AFTER `standardise_formats.py --apply --delete-originals`, so the
    manifest describes the final state rather than a mixture of .tif and .png.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

# Top-level directories that form the release. drug_treatment/ is a separate
# project and is deliberately absent.
DEFAULT_INCLUDE = ["bioimage_archive", "benchmark"]

# Never hashed: backups, editor droppings, interrupted conversions.
DEFAULT_EXCLUDE = [
    "*.bak", "*.bak_pre_png", "*.tmp", "*.png.tmp", "*~",
    ".DS_Store", "Thumbs.db", "*.swp", "__pycache__/*", "*.pyc",
]


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_type(rel: Path) -> str:
    """Classify from the path relative to the data root, not the absolute path."""
    parts = rel.parts
    if len(parts) < 2:
        return "metadata" if rel.suffix.lower() == ".csv" else "other"

    top, sub = parts[0], parts[1]

    if top == "bioimage_archive":
        return {
            "rgb":       "rgb",
            "dapi":      "dapi",
            "gt_image":  "gt_mask",
            "roi_mask":  "roi_mask",
            "gt_coords": "gt_coords",
            "gt_npz":    "gt_npz",
        }.get(sub, "metadata" if rel.suffix.lower() == ".csv" else "other")

    if top == "benchmark":
        if sub == "predictions":
            # benchmark/predictions/<model>/<file>
            return f"prediction_{parts[2]}" if len(parts) > 2 else "prediction"
        if sub == "splits":
            return "split"
        return "metadata" if rel.suffix.lower() == ".csv" else "other"

    return "other"


def collect(data_root: Path, include: list[str], exclude: list[str]) -> list[Path]:
    files: list[Path] = []
    for name in include:
        d = data_root / name
        if not d.is_dir():
            print(f"WARNING: include directory not found: {d}", file=sys.stderr)
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            rel = p.relative_to(data_root)
            if any(p.match(pat) or rel.match(pat) for pat in exclude):
                continue
            files.append(p)
    return sorted(files)


def summarise(files: list[Path], data_root: Path) -> None:
    by_type: Counter = Counter()
    size_by_type: Counter = Counter()
    for p in files:
        t = file_type(p.relative_to(data_root))
        by_type[t] += 1
        size_by_type[t] += p.stat().st_size
    total = sum(size_by_type.values())
    print(f"\n{'file_type':<28}{'files':>8}{'size':>12}", file=sys.stderr)
    print("-" * 48, file=sys.stderr)
    for t in sorted(by_type):
        print(f"{t:<28}{by_type[t]:>8}{size_by_type[t]/1e9:>10.2f} GB",
              file=sys.stderr)
    print("-" * 48, file=sys.stderr)
    print(f"{'TOTAL':<28}{len(files):>8}{total/1e9:>10.2f} GB\n", file=sys.stderr)


def generate_manifest(data_root: Path, out_path: Path, include: list[str],
                      exclude: list[str], list_only: bool) -> None:
    data_root = data_root.resolve()
    files = collect(data_root, include, exclude)
    print(f"Found {len(files)} files under {data_root}", file=sys.stderr)
    summarise(files, data_root)

    if list_only:
        print("--list given; nothing hashed, nothing written.", file=sys.stderr)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, p in enumerate(files, 1):
        rel = p.relative_to(data_root)
        rows.append({
            "relative_path": str(rel),
            "sha256": sha256_file(p),
            "size_bytes": p.stat().st_size,
            "file_type": file_type(rel),
        })
        if i % 250 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ...", file=sys.stderr, flush=True)

    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["relative_path", "sha256",
                                          "size_bytes", "file_type"])
        w.writeheader()
        w.writerows(rows)

    print(f"Manifest written to {out_path} ({len(rows)} files)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate SHA256 manifest for data release.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--out", type=Path,
                   default=Path("release/manifests/manifest_v1.0.csv"))
    p.add_argument("--include", nargs="+", default=DEFAULT_INCLUDE,
                   help=f"top-level dirs to hash (default: {DEFAULT_INCLUDE})")
    p.add_argument("--exclude", nargs="+", default=DEFAULT_EXCLUDE,
                   help="glob patterns to skip")
    p.add_argument("--list", action="store_true", dest="list_only",
                   help="report scope and sizes without hashing")
    a = p.parse_args()
    generate_manifest(a.data_root, a.out, a.include, a.exclude, a.list_only)


if __name__ == "__main__":
    main()