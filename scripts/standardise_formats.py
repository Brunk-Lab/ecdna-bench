#!/usr/bin/env python3
"""
standardise_formats.py
======================
One-pass format standardisation of the ecDNA image resource ahead of the
BioImage Archive deposition.

WHAT IT DOES
    dapi/     *.tif  ->  *.png   (2,984 files are already PNG bytes with a
                                  .tif extension; 2 are genuine TIFF and are
                                  re-encoded. All become real .png.)
    gt_image/ *.tif  ->  *.png   (binary masks, currently uncompressed TIFF at
                                  ~5.8 MB each; PNG is lossless and ~50 KB)
    rgb/      unchanged (genuine 3-channel TIFF)
    roi_mask/ unchanged (already PNG)

WHY
    Every file's extension should match its content before deposition, and
    17 GB of uncompressed binary masks is not a reasonable thing to ask people
    to download when the same data is ~1 GB losslessly.

SAFETY
    - Dry run by default. Nothing is written without --apply.
    - Every converted file is read back and compared pixel-for-pixel with the
      source before the source is removed.
    - Sources are only deleted with --delete-originals, which is a separate
      opt-in from --apply. Run once without it, check the result, then reclaim.
    - Resumable: a file whose .png already exists and verifies is skipped.

USAGE
    # 1. See what would happen
    python standardise_formats.py

    # 2. Convert (originals kept, disk usage temporarily higher)
    python standardise_formats.py --apply

    # 3. Verify independently, then reclaim space
    python standardise_formats.py --verify-only
    python standardise_formats.py --apply --delete-originals

AFTER THIS SCRIPT
    The filename columns in metadata still say .tif. Update them next:
      - <data_root>/bioimage_archive/metadata_internal.csv
            columns: dapi_filename, gt_image_filename
      - release/manifests/metadata.csv
            columns: dapi_relpath, gt_mask_relpath (and the *_fullpath columns)
    Then regenerate the SHA256 manifest:
      python scripts/generate_manifest.py --data-root <data_root> \
          --out release/manifests/manifest_v1.0.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import imageio.v3 as iio
except ImportError:
    sys.exit("imageio is required:  pip install imageio")

DEFAULT_ROOT = Path(
    "/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/bioimage_archive"
)

# Directories to convert, and the expected shape rank of their contents.
TARGETS = {
    "dapi":     {"expect_ndim": 2, "expect_binary": False},
    "gt_image": {"expect_ndim": 2, "expect_binary": True},
}


def magic(path: Path) -> str:
    head = path.open("rb").read(4)
    if head[:2] in (b"II", b"MM"):
        return "TIFF"
    if head[:4] == b"\x89PNG":
        return "PNG"
    return repr(head)


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def convert_one(src: Path, dst: Path, spec: dict, apply: bool) -> tuple[bool, str]:
    """Convert src -> dst. Returns (ok, message)."""
    try:
        arr = iio.imread(src)
    except Exception as exc:  # noqa: BLE001
        return False, f"unreadable source: {exc}"

    if arr.ndim != spec["expect_ndim"]:
        return False, f"unexpected ndim {arr.ndim} (expected {spec['expect_ndim']})"
    if arr.dtype != np.uint8:
        return False, f"unexpected dtype {arr.dtype} (expected uint8)"
    if spec["expect_binary"]:
        vals = np.unique(arr)
        if not set(vals.tolist()) <= {0, 255}:
            return False, f"expected binary {{0,255}}, found {vals[:6].tolist()}"

    if not apply:
        return True, "would convert"

    # Skip if an already-verified output exists (resumability).
    if dst.exists():
        try:
            if magic(dst) == "PNG" and np.array_equal(iio.imread(dst), arr):
                return True, "already converted"
        except Exception:  # noqa: BLE001
            pass  # fall through and rewrite

    tmp = dst.with_suffix(".png.tmp")
    iio.imwrite(tmp, arr, extension=".png")

    # Read back and compare before committing the name.
    check = iio.imread(tmp)
    if not np.array_equal(check, arr):
        tmp.unlink(missing_ok=True)
        return False, "VERIFY FAILED: round-trip mismatch"
    if magic(tmp) != "PNG":
        fmt = magic(tmp)
        tmp.unlink(missing_ok=True)
        return False, f"VERIFY FAILED: wrote {fmt}, not PNG"

    tmp.replace(dst)
    return True, "converted"


def process(root: Path, apply: bool, delete_originals: bool,
            verify_only: bool, sample: int = 0) -> int:
    failures = 0

    for name, spec in TARGETS.items():
        d = root / name
        if not d.is_dir():
            print(f"[{name}] SKIP — directory not found: {d}")
            continue

        sources = sorted(d.glob("*.tif"))
        total = len(sources)
        print(f"\n{'='*70}\n[{name}] {total} .tif files in {d}", flush=True)
        if sample and sample < total:
            step = max(1, total // sample)
            sources = sources[::step][:sample]
            print(f"  sampling {len(sources)} of {total} for validation "
                  f"(use --sample 0 to check every file)", flush=True)

        if verify_only:
            ok = bad = 0
            for src in sources:
                dst = src.with_suffix(".png")
                if not dst.exists():
                    print(f"  MISSING  {dst.name}")
                    bad += 1
                    continue
                try:
                    if np.array_equal(iio.imread(src), iio.imread(dst)):
                        ok += 1
                    else:
                        print(f"  MISMATCH {dst.name}")
                        bad += 1
                except Exception as exc:  # noqa: BLE001
                    print(f"  ERROR    {dst.name}: {exc}")
                    bad += 1
            print(f"  verified: {ok} ok, {bad} bad")
            failures += bad
            continue

        magics = {}
        bytes_before = bytes_after = 0
        converted = skipped = 0

        for i, src in enumerate(sources, 1):
            magics[magic(src)] = magics.get(magic(src), 0) + 1
            dst = src.with_suffix(".png")

            ok, msg = convert_one(src, dst, spec, apply)
            if not ok:
                print(f"  FAIL {src.name}: {msg}")
                failures += 1
                continue

            if msg == "already converted":
                skipped += 1
            elif msg == "converted":
                converted += 1

            bytes_before += src.stat().st_size
            if apply and dst.exists():
                bytes_after += dst.stat().st_size

            if i % 200 == 0:
                print(f"  ... {i}/{len(sources)}", flush=True)

        print(f"  source magic bytes: {magics}")
        if apply:
            print(f"  converted={converted} already-done={skipped} failed={failures}")
            print(f"  size: {human(bytes_before)} -> {human(bytes_after)}"
                  f"  (saved {human(bytes_before - bytes_after)})")
        else:
            print(f"  would convert {len(sources)} files "
                  f"({human(bytes_before)} of .tif sources)")

        # Deleting originals is a separate opt-in, and only after every
        # counterpart exists and matches.
        if apply and delete_originals and failures == 0:
            print(f"  re-verifying all {len(sources)} pairs before deletion...")
            all_ok = True
            for src in sources:
                dst = src.with_suffix(".png")
                if not dst.exists() or not np.array_equal(
                    iio.imread(src), iio.imread(dst)
                ):
                    print(f"  REFUSING TO DELETE — mismatch at {src.name}")
                    all_ok = False
                    break
            if all_ok:
                for src in sources:
                    src.unlink()
                print(f"  deleted {len(sources)} original .tif files")

    return failures


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help="bioimage_archive directory")
    ap.add_argument("--apply", action="store_true",
                    help="actually write files (default is a dry run)")
    ap.add_argument("--delete-originals", action="store_true",
                    help="remove source .tif files after full verification")
    ap.add_argument("--verify-only", action="store_true",
                    help="compare existing .png against .tif and report")
    ap.add_argument("--sample", type=int, default=None,
                    help="validate only N files spread across the directory. "
                         "Default: 60 for a dry run, 0 (all) otherwise.")
    args = ap.parse_args()

    if not args.root.is_dir():
        sys.exit(f"not a directory: {args.root}")

    if not args.apply and not args.verify_only:
        print("DRY RUN — nothing will be written. Add --apply to convert.\n")

    sample = args.sample
    if sample is None:
        sample = 0 if (args.apply or args.verify_only) else 60
    failures = process(args.root, args.apply, args.delete_originals,
                       args.verify_only, sample)

    print(f"\n{'='*70}")
    if failures:
        print(f"FINISHED WITH {failures} FAILURE(S) — do not proceed to the "
              f"metadata update until these are resolved.")
        sys.exit(1)
    print("No failures.")
    if args.apply and not args.delete_originals:
        print("Originals kept. Run --verify-only, then re-run with "
              "--apply --delete-originals to reclaim space.")


if __name__ == "__main__":
    main()