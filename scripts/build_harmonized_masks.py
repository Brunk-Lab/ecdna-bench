#!/usr/bin/env python
"""Rebuild release/harmonized_masks from the BioImage Archive predictions.

The figure notebooks 03 and 05 read the four baseline models from
release/harmonized_masks, which is not in git. This script recreates it from
$ECDNA_DATA_ROOT/predictions using the benchmark's harmonization rules:

  classical     <- predictions/classical_optimised   foreground = gray > 0
  ecseg         <- predictions/ecseg                 foreground = gray > 0
  mia           <- predictions/mia (TIFF)            foreground = gray > 0
  label_engine  <- predictions/label_engine (RGB)    gray > 0.5, then drop
                                                     components < 3 px (8-conn)

Output: {out}/{model}/{uid}.png with values 0/255. Checked on Longleaf against
the masks behind the frozen results: same foreground for all 4 x 1,145 images.

Usage:
  python scripts/build_harmonized_masks.py [--data-root DIR] [--out DIR] [--force]
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

MODELS = {  # output folder: (archive folder, threshold, min_area)
    "classical": ("classical_optimised", 0.0, 0),
    "ecseg": ("ecseg", 0.0, 0),
    "label_engine": ("label_engine", 0.5, 3),
    "mia": ("mia", 0.0, 0),
}
EXTS = (".png", ".tif", ".tiff")
EXPECTED = 1145


def binarize(path, threshold):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise OSError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    return img.astype(np.float32) > threshold


def min_area_filter(b, n):
    if n <= 0:
        return b
    k, lab, stats, _ = cv2.connectedComponentsWithStats(b.astype(np.uint8), connectivity=8)
    keep = np.zeros(k, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= n
    return keep[lab]


def work(job):
    src, dst, threshold, min_area = job
    try:
        b = min_area_filter(binarize(src, threshold), min_area)
        if not cv2.imwrite(str(dst), b.astype(np.uint8) * 255):
            return f"write failed: {dst}"
        return None
    except Exception as e:
        return f"{src.name}: {e}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.environ.get("ECDNA_DATA_ROOT"),
                    help="archive root containing predictions/ (default: $ECDNA_DATA_ROOT)")
    ap.add_argument("--out", default="release/harmonized_masks")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true", help="overwrite existing masks")
    a = ap.parse_args(argv)
    if not a.data_root:
        sys.exit("set --data-root or ECDNA_DATA_ROOT")
    pred_root, out = Path(a.data_root).expanduser() / "predictions", Path(a.out).expanduser()

    failed = False
    for model, (folder, thr, min_area) in MODELS.items():
        src_dir, dst_dir = pred_root / folder, out / model
        if not src_dir.is_dir():
            print(f"[SKIP] {model}: {src_dir} not found")
            failed = True
            continue
        if dst_dir.is_dir() and any(dst_dir.iterdir()) and not a.force:
            print(f"[KEEP] {model}: {dst_dir} already has files (use --force to rebuild)")
            continue
        dst_dir.mkdir(parents=True, exist_ok=True)
        srcs = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in EXTS)
        jobs = [(p, dst_dir / f"{p.stem}.png", thr, min_area) for p in srcs]
        with ProcessPoolExecutor(a.workers) as ex:
            errors = [e for e in ex.map(work, jobs, chunksize=8) if e]
        note = "" if len(srcs) == EXPECTED else f"  (expected {EXPECTED})"
        print(f"[{'OK' if not errors else 'FAIL'}] {model}: {len(srcs) - len(errors)} masks "
              f"from {folder}{note}")
        for e in errors[:5]:
            print(f"    {e}")
        failed |= bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
