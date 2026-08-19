#!/usr/bin/env python3
"""
update_metadata_after_standardisation.py
========================================
Bring the two metadata files back in step with the data after
`standardise_formats.py --apply` has converted dapi/ and gt_image/ to PNG.

TWO INDEPENDENT PROBLEMS ARE FIXED HERE
    1. Extensions.  dapi/ and gt_image/ are now .png, but every filename
       column still says .tif.
    2. Stale roots.  The *_fullpath columns still point at
       /work/users/b/e/behnamie/... , which has not been valid since the
       149 GB migration to /proj.

FILES TOUCHED
    <data_root>/bioimage_archive/metadata_internal.csv   (2,986 rows)
        dapi_filename, gt_image_filename           .tif -> .png

    release/manifests/metadata.csv                       (1,145 rows)
        dapi_relpath,  gt_mask_relpath             .tif -> .png
        dapi_fullpath, gt_fullpath                 .tif -> .png  AND  /work -> /proj
        rgb_fullpath,  roi_fullpath                              /work -> /proj

DELIBERATELY NOT TOUCHED
    original_dapi_name / original_gt_name / original_rgb_name / original_roi_name
        These record the filenames as the files ARRIVED. They are provenance,
        not pointers, and should keep saying .tif. Override with
        --rewrite-original-names if you decide otherwise.

SAFETY
    - Dry run by default. Nothing is written without --apply.
    - Every rewritten path is checked to exist on disk BEFORE anything is
      written. If a single one is missing the script refuses and writes
      nothing -- that is the check that turns a text substitution into a
      verified one.
    - Both CSVs are backed up to <name>.bak_pre_png before being replaced.
    - Every column is read and written as text, so booleans, integers and
      empty cells come back out byte-identical to how they went in.

USAGE
    python scripts/update_metadata_after_standardisation.py            # dry run
    python scripts/update_metadata_after_standardisation.py --apply

AFTER THIS SCRIPT
    Reclaim the disk, then rebuild the manifest against the final state:
      python scripts/standardise_formats.py --apply --delete-originals
      python scripts/generate_manifest.py \
          --data-root /proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data \
          --out release/manifests/manifest_v1.0.csv
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

DATA_ROOT = Path("/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data")
REPO_ROOT = Path("/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench")

OLD_PREFIX = "/work/users/b/e/behnamie/ecDNA_Data"
NEW_PREFIX = str(DATA_ROOT)

# internal CSV: column -> subdirectory it lives in (for existence checking)
INTERNAL_DIRS = {
    "rgb_filename":       "rgb",
    "dapi_filename":      "dapi",
    "gt_image_filename":  "gt_image",
    "gt_npz_filename":    "gt_npz",
    "gt_coords_filename": "gt_coords",
    "roi_mask_filename":  "roi_mask",
}
INTERNAL_RETARGET = ["dapi_filename", "gt_image_filename"]

RELEASE_RETARGET = ["dapi_relpath", "gt_mask_relpath",
                    "dapi_fullpath", "gt_fullpath"]
RELEASE_FULLPATHS = ["rgb_fullpath", "dapi_fullpath",
                     "roi_fullpath", "gt_fullpath"]
RELEASE_RELPATHS = ["rgb_relpath", "dapi_relpath",
                    "roi_mask_relpath", "gt_mask_relpath"]
ORIGINAL_NAMES = ["original_dapi_name", "original_gt_name"]


def read_csv_as_text(path: Path) -> pd.DataFrame:
    """Read every cell as a string so nothing is silently reformatted."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, na_filter=False)


def retarget_ext(s: pd.Series) -> tuple[pd.Series, int]:
    mask = s.str.endswith(".tif")
    return s.mask(mask, s.str.slice(0, -4) + ".png"), int(mask.sum())


def retarget_root(s: pd.Series) -> tuple[pd.Series, int]:
    mask = s.str.startswith(OLD_PREFIX)
    return s.mask(mask, s.str.replace(OLD_PREFIX, NEW_PREFIX, regex=False)), int(mask.sum())


def check_exists(paths: pd.Series, label: str, missing: list) -> int:
    """Confirm every non-empty path resolves. Returns count checked."""
    checked = 0
    for v in paths:
        if not v:
            continue
        checked += 1
        if not Path(v).exists():
            missing.append(f"{label}: {v}")
    return checked


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DATA_ROOT)
    ap.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    ap.add_argument("--apply", action="store_true",
                    help="write the files (default is a dry run)")
    ap.add_argument("--rewrite-original-names", action="store_true",
                    help="also change original_dapi_name / original_gt_name "
                         "to .png (default: keep as .tif, they are provenance)")
    args = ap.parse_args()

    internal_csv = args.data_root / "bioimage_archive" / "metadata_internal.csv"
    release_csv = args.repo_root / "release" / "manifests" / "metadata.csv"

    for p in (internal_csv, release_csv):
        if not p.is_file():
            sys.exit(f"not found: {p}")

    if not args.apply:
        print("DRY RUN — nothing will be written. Add --apply to commit.\n")

    missing: list[str] = []
    changes: list[str] = []

    # ---------------------------------------------------------------- internal
    print("=" * 70)
    print(f"[internal] {internal_csv}")
    di = read_csv_as_text(internal_csv)
    print(f"  {len(di)} rows, {len(di.columns)} columns")

    for col in INTERNAL_RETARGET:
        if col not in di.columns:
            sys.exit(f"  column missing: {col}")
        di[col], n = retarget_ext(di[col])
        changes.append(f"internal.{col}: {n} .tif -> .png")
        print(f"  {col:20s} {n:5d} renamed")

    checked = 0
    for col, sub in INTERNAL_DIRS.items():
        if col not in di.columns:
            continue
        base = args.data_root / "bioimage_archive" / sub
        checked += check_exists(
            di[col].map(lambda v: str(base / v) if v else ""),
            f"internal.{col}", missing)
    print(f"  verified {checked} file paths on disk")

    # ----------------------------------------------------------------- release
    print("=" * 70)
    print(f"[release]  {release_csv}")
    dr = read_csv_as_text(release_csv)
    print(f"  {len(dr)} rows, {len(dr.columns)} columns")

    for col in RELEASE_RETARGET:
        if col not in dr.columns:
            sys.exit(f"  column missing: {col}")
        dr[col], n = retarget_ext(dr[col])
        changes.append(f"release.{col}: {n} .tif -> .png")
        print(f"  {col:20s} {n:5d} renamed")

    for col in RELEASE_FULLPATHS:
        dr[col], n = retarget_root(dr[col])
        changes.append(f"release.{col}: {n} /work -> /proj")
        print(f"  {col:20s} {n:5d} re-rooted")

    if args.rewrite_original_names:
        for col in ORIGINAL_NAMES:
            if col in dr.columns:
                dr[col], n = retarget_ext(dr[col])
                changes.append(f"release.{col}: {n} .tif -> .png (opt-in)")
                print(f"  {col:20s} {n:5d} renamed (--rewrite-original-names)")
    else:
        print("  original_*_name columns left as .tif (provenance)")

    checked = 0
    for col in RELEASE_FULLPATHS:
        checked += check_exists(dr[col], f"release.{col}", missing)
    for col in RELEASE_RELPATHS:
        if col in dr.columns:
            base = args.data_root / "bioimage_archive"
            checked += check_exists(
                dr[col].map(lambda v: str(base / v) if v else ""),
                f"release.{col}", missing)
    print(f"  verified {checked} file paths on disk")

    # ------------------------------------------------------------------ verdict
    print("=" * 70)
    if missing:
        print(f"REFUSING TO WRITE — {len(missing)} path(s) do not resolve.")
        for m in missing[:15]:
            print(f"  {m}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")
        print("\nMost likely cause: standardise_formats.py --apply has not "
              "finished, or finished with failures. Check its log first.")
        sys.exit(1)

    print("All rewritten paths resolve on disk.")
    for c in changes:
        print(f"  {c}")

    if not args.apply:
        print("\nDry run — no files written. Re-run with --apply.")
        return

    for path, frame in ((internal_csv, di), (release_csv, dr)):
        backup = path.with_suffix(path.suffix + ".bak_pre_png")
        shutil.copy2(path, backup)
        frame.to_csv(path, index=False)
        print(f"  wrote {path}  (backup: {backup.name})")

    print("\nDone. Next: --delete-originals, then regenerate the manifest.")


if __name__ == "__main__":
    main()