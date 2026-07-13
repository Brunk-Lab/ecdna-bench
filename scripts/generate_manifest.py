#!/usr/bin/env python3
"""
scripts/generate_manifest.py
==============================
Generate a SHA256 manifest for the data release.

Usage
-----
    python scripts/generate_manifest.py \
        --data-root /path/to/ecdna_data \
        --out release/manifests/manifest_v1.0.csv

The manifest CSV has columns: relative_path, sha256, size_bytes, file_type.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def file_type(path: Path) -> str:
    stem = path.stem.lower()
    parent = str(path.parent).lower()
    if "gt" in parent or "gt_mask" in stem:
        return "gt_mask"
    if "roi" in parent or "roi" in stem:
        return "roi_mask"
    if "dapi" in parent or "dapi" in stem:
        return "dapi"
    if "rgb" in parent:
        return "rgb"
    if "mia" in parent or "mia" in stem:
        return "mia_mask"
    return "other"


def generate_manifest(data_root: Path, out_path: Path) -> None:
    data_root = data_root.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    png_files = sorted(data_root.rglob("*.png"))
    print(f"Found {len(png_files)} PNG files under {data_root}", file=sys.stderr)

    rows = []
    for i, p in enumerate(png_files):
        rel = p.relative_to(data_root)
        digest = sha256_file(p)
        rows.append({
            "relative_path": str(rel),
            "sha256": digest,
            "size_bytes": p.stat().st_size,
            "file_type": file_type(p),
        })
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(png_files)} ...", file=sys.stderr)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["relative_path", "sha256", "size_bytes", "file_type"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest written to {out_path} ({len(rows)} files)", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate SHA256 manifest for data release.")
    p.add_argument("--data-root", type=Path, required=True, help="Root data directory.")
    p.add_argument("--out", type=Path, default=Path("release/manifests/manifest_v1.0.csv"))
    args = p.parse_args()
    generate_manifest(args.data_root, args.out)


if __name__ == "__main__":
    main()
