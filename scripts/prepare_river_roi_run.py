#!/usr/bin/env python3
"""
prepare_river_roi_run.py
========================

Prepare everything needed to run ecCount inside River's *predicted* ROI masks,
without changing a single line of ecdna-bench code.

Why no code change is needed
----------------------------
``cli/run_eccount.py`` reads ``roi_fullpath`` from the consistency CSV and
multiplies the RGB by that mask before the forward pass.
``benchmark/run.py`` reads ``gt_fullpath``.  So redirecting those two columns
redirects the entire run.  This script writes the redirected CSVs and the
ROI-masked GT masks they point at.

The evaluation rule
-------------------
The predicted ROI is applied to BOTH sides — the RGB ecCount sees and the GT
it is scored against.  Anything outside the predicted ROI is outside the
measurement for both.  Scoring ecCount against an unmasked GT would penalise
it for ecDNA it was never shown, which measures River's model, not ecCount.

What it writes (and nothing else)
---------------------------------
    release/river_roi_run/
        gt_mask_river/                 GT ∧ River ROI, one PNG per image
        consistency_benchmark.csv      the 1,145, roi+gt redirected
        consistency_extension.csv      the ~1,841 that have no manual ROI
        config_river_eccount.yaml      output dirs for inference
        config_river_benchmark.yaml    output dirs for scoring
        preparation_report.csv         per-image record of what happened

Nothing under ``release/frozen_results/`` or ``ecDNA_Data/`` is touched.

Usage
-----
    conda activate /proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench
    export PYTHONNOUSERSITE=1

    python scripts/prepare_river_roi_run.py --dry-run    # inspect, writes nothing
    python scripts/prepare_river_roi_run.py --apply

Stop-on-bad-assumption
----------------------
Every path is verified before use.  Formats are checked by magic bytes, not by
extension.  Shape mismatches, transposed masks, empty intersections and missing
files are counted and reported; if any category exceeds its tolerance the
script stops instead of producing a directory of quietly wrong masks.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Defaults — override on the command line if any of these are wrong.
# --------------------------------------------------------------------------
PROJ = Path("/proj/brunk_ecdna_cv_project/Poorya")
REPO = PROJ / "ecdna-bench"
BIA = PROJ / "ecDNA_Data/bioimage_archive"

D_CONSISTENCY = REPO / "release/manifests/dl_master_metadata_stage1_step3_consistency.csv"
D_DISCLOSURE = (REPO / "release/figures/notebook05/"
                       "source_figS_gt_count_disclosure_long.csv")
D_RIVER = PROJ / "River/output/all"
D_OUT = REPO / "release/river_roi_run"

IMG_EXT = {".png", ".tif", ".tiff"}

# Tolerances. Exceeding any of these is a stop, not a warning.
MAX_MISSING_FRAC = 0.05      # images with no River mask
MAX_SHAPE_FAIL = 0           # shape mismatches: zero tolerance
MAX_EMPTY_FRAC = 0.02        # GT ∧ ROI empty where GT was not


# --------------------------------------------------------------------------
def read_image(path: Path) -> np.ndarray:
    import cv2
    arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise IOError(f"cannot decode {path}")
    return arr


def magic_format(path: Path) -> str:
    """Real format from the leading bytes. Extensions have lied here before."""
    with open(path, "rb") as fh:
        head = fh.read(8)
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG"
    if head[:2] in (b"II", b"MM"):
        return "TIFF"
    if head.startswith(b"\xff\xd8\xff"):
        return "JPEG"
    return f"UNKNOWN({head[:4]!r})"


def to_binary(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        arr = arr.max(axis=2)
    return arr > 0


def write_png(mask_bool: np.ndarray, out_path: Path) -> None:
    """Atomic PNG write with an explicit extension, then verify the bytes.

    The temp name carries .png so the encoder cannot infer a different format
    from the suffix — the exact failure that put TIFF bytes under .png names
    earlier in this project.
    """
    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.png")
    if not cv2.imwrite(str(tmp), (mask_bool.astype(np.uint8) * 255)):
        raise IOError(f"imwrite failed for {tmp}")
    tmp.replace(out_path)
    fmt = magic_format(out_path)
    if fmt != "PNG":
        raise IOError(f"wrote {out_path} but bytes say {fmt}")


def index_masks(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        sys.exit(f"ERROR: not a directory: {directory}")
    return {p.stem: p for p in directory.iterdir()
            if p.is_file() and p.suffix.lower() in IMG_EXT}


def guess_path(root: Path, uid: str, subdir: str) -> Path | None:
    """Resolve <root>/<subdir>/<uid>.<ext> without assuming the extension."""
    d = root / subdir
    if not d.is_dir():
        return None
    for ext in (".png", ".tif", ".tiff"):
        p = d / f"{uid}{ext}"
        if p.is_file():
            return p
    return None


# --------------------------------------------------------------------------
def build_rows(args) -> tuple[list[dict], list[dict]]:
    """Return (benchmark_rows, extension_rows) with every path verified."""
    import pandas as pd

    cons = pd.read_csv(args.consistency)
    need = {"unique_id", "rgb_fullpath", "gt_fullpath", "roi_fullpath"}
    missing = need - set(cons.columns)
    if missing:
        sys.exit(f"ERROR: consistency CSV lacks columns {sorted(missing)}\n"
                 f"  {args.consistency}")
    bench_uids = set(cons["unique_id"].astype(str))
    print(f"benchmark images from consistency CSV : {len(bench_uids):,}")

    disc = pd.read_csv(args.disclosure)
    all_uids = list(dict.fromkeys(disc["unique_id"].astype(str)))
    print(f"full resource from disclosure CSV     : {len(all_uids):,}")
    ext_uids = [u for u in all_uids if u not in bench_uids]
    print(f"extension images (no manual ROI)      : {len(ext_uids):,}")

    cell_of = dict(zip(disc["unique_id"].astype(str), disc["cell_line"]))
    gtcount_of = dict(zip(disc["unique_id"].astype(str), disc["ecDNA_gt"]))

    bench = cons[["unique_id", "cell_line", "split",
                  "rgb_fullpath", "gt_fullpath", "roi_fullpath"]].copy()
    bench["unique_id"] = bench["unique_id"].astype(str)
    bench_rows = bench.to_dict("records")

    # Extension rows: paths derived from the archive layout, then verified.
    ext_rows, unresolved = [], []
    for uid in ext_uids:
        rgb = guess_path(args.archive, uid, "rgb")
        gt = guess_path(args.archive, uid, "gt_image") or \
             guess_path(args.archive, uid, "gt_mask")
        if rgb is None or gt is None:
            unresolved.append((uid, "rgb" if rgb is None else "gt"))
            continue
        ext_rows.append({
            "unique_id": uid,
            "cell_line": cell_of.get(uid, "unknown"),
            "split": "extension",
            "rgb_fullpath": str(rgb),
            "gt_fullpath": str(gt),
            "roi_fullpath": "",
            "ecDNA_gt": gtcount_of.get(uid, ""),
        })

    if unresolved:
        print(f"\n  {len(unresolved):,} extension images could not be resolved "
              f"in {args.archive}")
        for uid, which in unresolved[:8]:
            print(f"    {uid}: no {which}")
        if len(unresolved) > MAX_MISSING_FRAC * max(len(ext_uids), 1):
            sys.exit("\nERROR: too many unresolved extension images. The archive "
                     "layout is not what this script assumed — pass --archive or "
                     "tell me the correct subdirectory names. Not guessing.")
    return bench_rows, ext_rows


def process(rows: list[dict], river: dict[str, Path], out_gt_dir: Path,
            label: str, apply: bool) -> tuple[list[dict], dict]:
    """Intersect GT with the River mask for each row. Returns (records, tally)."""
    print(f"\n{'=' * 74}\n{label}  (n = {len(rows):,})\n{'=' * 74}")
    recs = []
    tally = dict(ok=0, no_river=0, shape_fail=0, read_fail=0,
                 empty_after=0, transposed=0)

    for i, row in enumerate(rows, 1):
        uid = row["unique_id"]
        rec = {"unique_id": uid, "cell_line": row.get("cell_line", ""),
               "split": row.get("split", ""), "status": "", "gt_px": 0,
               "roi_px": 0, "kept_px": 0, "kept_frac": np.nan}

        rp = river.get(uid)
        if rp is None:
            rec["status"] = "no_river_mask"
            tally["no_river"] += 1
            recs.append(rec)
            continue
        try:
            gt = to_binary(read_image(Path(row["gt_fullpath"])))
            roi = to_binary(read_image(rp))
        except Exception as exc:
            rec["status"] = f"read_fail: {exc}"
            tally["read_fail"] += 1
            recs.append(rec)
            continue

        if gt.shape != roi.shape:
            if gt.shape == roi.T.shape:
                rec["status"] = f"TRANSPOSED {gt.shape} vs {roi.shape}"
                tally["transposed"] += 1
            else:
                rec["status"] = f"shape {gt.shape} vs {roi.shape}"
            tally["shape_fail"] += 1
            recs.append(rec)
            continue

        keep = gt & roi
        rec.update(gt_px=int(gt.sum()), roi_px=int(roi.sum()),
                   kept_px=int(keep.sum()),
                   kept_frac=float(keep.sum() / gt.sum()) if gt.sum() else np.nan)
        if gt.sum() > 0 and keep.sum() == 0:
            rec["status"] = "empty_after_mask"
            tally["empty_after"] += 1
        else:
            rec["status"] = "ok"
            tally["ok"] += 1

        if apply:
            write_png(keep, out_gt_dir / f"{uid}.png")
        recs.append(rec)

        if i % 250 == 0:
            print(f"    ... {i:,} / {len(rows):,}   ok={tally['ok']:,}")

    print(f"  ok               {tally['ok']:,}")
    print(f"  no River mask    {tally['no_river']:,}")
    print(f"  read failures    {tally['read_fail']:,}")
    print(f"  shape mismatch   {tally['shape_fail']:,}"
          f"   (of which transposed: {tally['transposed']:,})")
    print(f"  empty after mask {tally['empty_after']:,}")

    kf = np.array([r["kept_frac"] for r in recs if np.isfinite(r["kept_frac"])])
    if kf.size:
        print(f"  GT pixels retained inside the predicted ROI: "
              f"median {np.median(kf):.4f}   p10 {np.percentile(kf, 10):.4f}   "
              f"min {kf.min():.4f}")

    n = max(len(rows), 1)
    if tally["shape_fail"] > MAX_SHAPE_FAIL:
        sys.exit("\nERROR: shape mismatches present. Transposed or resized masks "
                 "would silently corrupt every downstream number. Fix first.")
    if tally["no_river"] > MAX_MISSING_FRAC * n:
        sys.exit(f"\nERROR: {tally['no_river']:,} images have no River mask "
                 f"(> {MAX_MISSING_FRAC:.0%}). Ask River before continuing.")
    if tally["empty_after"] > MAX_EMPTY_FRAC * n:
        sys.exit(f"\nERROR: {tally['empty_after']:,} GT masks are wiped out "
                 f"entirely by the predicted ROI (> {MAX_EMPTY_FRAC:.0%}). "
                 f"That is a mask-correspondence problem, not a model result.")
    return recs, tally


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {len(rows):,} rows -> {path}")


CONFIG_ECCOUNT = """\
# Written by prepare_river_roi_run.py — ecCount inference inside River's ROI.
# Merged over configs/default.yaml; every model parameter is unchanged.
paths:
  consistency_csv: {consistency}
  eccount_checkpoint: {ckpt}
  eccount_threshold_masks: {out}/masks/eccount_threshold
  eccount_peaks_masks: {out}/masks/eccount_peaks
"""

CONFIG_BENCH = """\
# Written by prepare_river_roi_run.py — scoring inside River's ROI.
# GT is the ROI-masked GT; predictions are the ROI-restricted ecCount output.
paths:
  consistency_csv: {consistency}
  frozen_results_dir: {out}/frozen_results
  harmonized_masks_dir: {out}/harmonized_masks
  eccount_threshold_masks: {out}/masks/eccount_threshold
  eccount_peaks_masks: {out}/masks/eccount_peaks
benchmark:
  n_workers: 8
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--consistency", type=Path, default=D_CONSISTENCY)
    ap.add_argument("--disclosure", type=Path, default=D_DISCLOSURE)
    ap.add_argument("--river", type=Path, default=D_RIVER)
    ap.add_argument("--archive", type=Path, default=BIA)
    ap.add_argument("--out", type=Path, default=D_OUT)
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO / "release/model_checkpoints/eccount_best.pt")
    ap.add_argument("--benchmark-only", action="store_true",
                    help="Skip the extension set (§3A only).")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true", help="Verify only; write nothing.")
    g.add_argument("--apply", action="store_true", help="Write the masks and CSVs.")
    args = ap.parse_args()

    apply = bool(args.apply)
    print(f"mode: {'APPLY' if apply else 'DRY RUN (nothing will be written)'}\n")

    for p in (args.consistency, args.disclosure):
        if not p.is_file():
            sys.exit(f"ERROR: missing input: {p}")

    river = index_masks(args.river)
    print(f"River predicted masks indexed: {len(river):,}")
    sample = list(river.values())[:20]
    fmts = {magic_format(p) for p in sample}
    print(f"  real format (n=20 sample): {fmts}")
    if len(fmts) > 1:
        print("  NOTE: mixed formats in the predicted-mask directory.")

    bench_rows, ext_rows = build_rows(args)

    gt_dir = args.out / "gt_mask_river"
    if apply:
        gt_dir.mkdir(parents=True, exist_ok=True)

    recs_b, _ = process(bench_rows, river, gt_dir, "BENCHMARK SET (§3A)", apply)
    recs_x = []
    if not args.benchmark_only and ext_rows:
        recs_x, _ = process(ext_rows, river, gt_dir, "EXTENSION SET (§3B)", apply)

    if not apply:
        print("\nDRY RUN complete. Nothing was written. Re-run with --apply.")
        return

    # Redirected consistency CSVs -----------------------------------------
    ok_b = {r["unique_id"] for r in recs_b if r["status"] == "ok"}
    ok_x = {r["unique_id"] for r in recs_x if r["status"] == "ok"}

    fields = ["unique_id", "cell_line", "split", "rgb_fullpath",
              "gt_fullpath", "roi_fullpath", "count_mask_consistent"]

    out_b = []
    for row in bench_rows:
        uid = row["unique_id"]
        if uid not in ok_b:
            continue
        out_b.append({**row,
                      "gt_fullpath": str(gt_dir / f"{uid}.png"),
                      "roi_fullpath": str(river[uid]),
                      "count_mask_consistent": True})
    write_csv(out_b, args.out / "consistency_benchmark.csv", fields)

    if recs_x:
        out_x = []
        for row in ext_rows:
            uid = row["unique_id"]
            if uid not in ok_x:
                continue
            out_x.append({**row,
                          "gt_fullpath": str(gt_dir / f"{uid}.png"),
                          "roi_fullpath": str(river[uid]),
                          "count_mask_consistent": True})
        write_csv(out_x, args.out / "consistency_extension.csv", fields)

    # Report ---------------------------------------------------------------
    all_recs = recs_b + recs_x
    write_csv(all_recs, args.out / "preparation_report.csv",
              ["unique_id", "cell_line", "split", "status",
               "gt_px", "roi_px", "kept_px", "kept_frac"])

    # Configs --------------------------------------------------------------
    for name, tpl, cons in [
        ("config_river_eccount.yaml", CONFIG_ECCOUNT,
         args.out / "consistency_benchmark.csv"),
        ("config_river_benchmark.yaml", CONFIG_BENCH,
         args.out / "consistency_benchmark.csv"),
    ]:
        (args.out / name).write_text(
            tpl.format(consistency=cons, out=args.out, ckpt=args.checkpoint))
        print(f"  wrote {args.out / name}")

    print(f"\nDone. Outputs under {args.out}")
    print("Nothing under release/frozen_results/ or ecDNA_Data/ was modified.")
    print("\nNext: sbatch scripts/slurm/submit_river_roi.sh")


if __name__ == "__main__":
    main()
