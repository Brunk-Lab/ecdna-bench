#!/usr/bin/env python
"""
roi_accuracy_metrics.py — score the predicted ROI against the manual ROI.

Computes, for every benchmark image that has a manual ROI, how well the ROI
model's predicted mask agrees with it, and how much of the annotated ecDNA
survives inside the prediction. Writes one row per image, then per-cell-line
summaries for the held-out test split and for all benchmark images.

    python scripts/roi_accuracy_metrics.py                  # all 1,145 images
    python scripts/roi_accuracy_metrics.py --split test     # the 175 test rows
    python scripts/roi_accuracy_metrics.py --resume         # continue a part run

Runs serially with checkpointing every 50 images, so an interrupted run is
resumed rather than restarted. Expect roughly five minutes for 1,145 images.

METRICS, AND WHY EACH IS DEFINED THIS WAY
-----------------------------------------
iou
    |manual AND predicted| / |manual OR predicted|. Symmetric. Penalises both
    over- and under-coverage.

dice
    2|manual AND predicted| / (|manual| + |predicted|). Also symmetric, but
    weights agreement more heavily than IoU at the same overlap. Both are
    reported because reviewers ask for whichever one is absent.

frac_manual_covered
    |manual AND predicted| / |manual|. Recall of the manual ROI. This is the
    number that matters for "did the prediction miss part of the spread".

area_ratio
    |predicted| / |manual|. Above 1 means the prediction is larger than the
    manual annotation. This is the quantity behind the positive count bias in
    the ROI run: a larger ROI lets ecCount find real ecDNA that the manual
    ground truth never covered, and those score as false positives.

object_retention
    Annotated ecDNA objects whose centroid lies inside the predicted ROI,
    divided by all annotated objects in the image. An OBJECT fraction.

pixel_retention
    Annotated ecDNA foreground pixels inside the predicted ROI, divided by all
    annotated foreground pixels. A PIXEL fraction.

    These last two are not interchangeable and the project has been bitten by
    treating them as one number. preparation_report.csv `kept_frac` is the
    pixel quantity; roi_retention_summary.csv is the object quantity. Both are
    written here, in separate columns, with the units in the column name.

Object counting uses the canonical operating point — 8-connectivity, minimum
area 3 px — applied with cv2.connectedComponentsWithStats so that the counting
rule is visible in this file rather than inherited.

LOW-BURDEN CELL LINES
---------------------
COLO320DM has a mean of 46 annotated objects per image and SUM159PT has 5.
A retention *fraction* on an image with four objects moves in steps of 0.25.
The summary therefore reports the object numerator and denominator alongside
every fraction, and the per-cell-line table carries a pooled retention
(total kept / total annotated) as well as the mean of per-image fractions.
Quote the pooled figure for those lines.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pandas as pd

logger = logging.getLogger("roi_acc")

MIN_AREA = 3
CONNECTIVITY = 8
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF")
CHECKPOINT_EVERY = 50

EXPECTED_N_BENCHMARK = 1145
EXPECTED_GT_TOTAL = 228_039


def index_masks(folder: Path) -> Dict[str, Path]:
    """Map filename stem -> path for every mask in *folder*."""
    out: Dict[str, Path] = {}
    if not folder.exists():
        logger.error("Mask directory does not exist: %s", folder)
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_SUFFIXES:
            out.setdefault(p.stem, p)
    return out


def read_binary(path: Path) -> Optional[np.ndarray]:
    """Read a mask as a boolean array.

    Reads unchanged rather than as grayscale, because some masks in this
    project are three-channel PNGs whose content lives in one channel only.
    Taking the channel maximum is correct for both cases; cv2's grayscale
    conversion would weight the channels and can zero a single-channel-encoded
    mask. File extensions in this project are also not reliable indicators of
    format, so nothing here branches on the suffix.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img.max(axis=2)
    return img > 0


def count_objects(mask: np.ndarray) -> np.ndarray:
    """Centroids of connected components at the canonical operating point.

    Returns an (N, 2) array of (x, y) centroids, components smaller than
    MIN_AREA removed. Background is label 0 and is always dropped.
    """
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=CONNECTIVITY
    )
    if n <= 1:
        return np.empty((0, 2), dtype=float)
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = areas >= MIN_AREA
    return centroids[1:][keep]


def score_one(
    uid: str,
    manual_path: Path,
    pred_path: Path,
    gt_path: Path,
) -> Optional[dict]:
    """Score one image. Returns None when the masks cannot be compared."""
    manual = read_binary(manual_path)
    pred = read_binary(pred_path)
    if manual is None or pred is None:
        logger.warning("%s: unreadable mask, skipped", uid)
        return None
    if manual.shape != pred.shape:
        logger.warning(
            "%s: shape mismatch manual %s vs predicted %s, skipped",
            uid, manual.shape, pred.shape,
        )
        return None

    inter = int(np.count_nonzero(manual & pred))
    man_px = int(np.count_nonzero(manual))
    pred_px = int(np.count_nonzero(pred))
    union = man_px + pred_px - inter

    row = {
        "uid": uid,
        "manual_px": man_px,
        "pred_px": pred_px,
        "intersection_px": inter,
        "iou": inter / union if union else np.nan,
        "dice": 2 * inter / (man_px + pred_px) if (man_px + pred_px) else np.nan,
        "frac_manual_covered": inter / man_px if man_px else np.nan,
        "area_ratio": pred_px / man_px if man_px else np.nan,
        "pred_outside_manual_px": pred_px - inter,
        "pred_roi_empty": pred_px == 0,
    }

    row["manual_only_px"] = man_px - inter

    gt = read_binary(gt_path)
    if gt is None or gt.shape != pred.shape:
        for k in (
            "gt_objects", "gt_objects_in_roi", "object_retention",
            "gt_fg_px", "gt_fg_px_in_roi", "pixel_retention",
            "gt_objects_in_manual", "gt_objects_in_both",
            "gt_objects_manual_only", "gt_objects_pred_only",
            "retention_vs_manual",
        ):
            row[k] = np.nan
        return row

    centroids = count_objects(gt)
    n_obj = len(centroids)
    if n_obj:
        # A centroid is inside a mask when that mask is True at the rounded
        # centroid pixel. Clipped so a centroid on the border does not index
        # out of range.
        xs = np.clip(np.round(centroids[:, 0]).astype(int), 0, pred.shape[1] - 1)
        ys = np.clip(np.round(centroids[:, 1]).astype(int), 0, pred.shape[0] - 1)
        in_pred = pred[ys, xs]
        in_man = manual[ys, xs]
        n_in = int(np.count_nonzero(in_pred))
        n_in_man = int(np.count_nonzero(in_man))
        n_both = int(np.count_nonzero(in_pred & in_man))
        n_man_only = int(np.count_nonzero(in_man & ~in_pred))
        n_pred_only = int(np.count_nonzero(in_pred & ~in_man))
    else:
        n_in = n_in_man = n_both = n_man_only = n_pred_only = 0

    gt_px = int(np.count_nonzero(gt))
    gt_px_in = int(np.count_nonzero(gt & pred))

    row.update(
        gt_objects=n_obj,
        gt_objects_in_roi=n_in,
        object_retention=n_in / n_obj if n_obj else np.nan,
        gt_fg_px=gt_px,
        gt_fg_px_in_roi=gt_px_in,
        pixel_retention=gt_px_in / gt_px if gt_px else np.nan,
        # Manual-relative accounting. gt_objects_manual_only is the quantity
        # that answers "what is lost by using the predicted ROI instead of the
        # human's"; gt_objects_pred_only is what the larger predicted ROI picks
        # up outside the manual annotation, which is where the positive count
        # bias in the ROI benchmark run comes from.
        gt_objects_in_manual=n_in_man,
        gt_objects_in_both=n_both,
        gt_objects_manual_only=n_man_only,
        gt_objects_pred_only=n_pred_only,
        retention_vs_manual=n_both / n_in_man if n_in_man else np.nan,
    )
    return row


def summarise(df: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Per-group summary. Pooled retention alongside per-image means."""
    out = []
    for keys, g in df.groupby(group_cols):
        keys = keys if isinstance(keys, tuple) else (keys,)
        rec = dict(zip(group_cols, keys))
        rec.update(
            n_images=len(g),
            iou_median=g["iou"].median(),
            iou_mean=g["iou"].mean(),
            iou_q25=g["iou"].quantile(0.25),
            iou_q75=g["iou"].quantile(0.75),
            dice_median=g["dice"].median(),
            dice_mean=g["dice"].mean(),
            dice_q25=g["dice"].quantile(0.25),
            dice_q75=g["dice"].quantile(0.75),
            frac_manual_covered_median=g["frac_manual_covered"].median(),
            area_ratio_median=g["area_ratio"].median(),
            # Pooled = total kept / total annotated. Use this for low-burden
            # lines, where a mean of per-image fractions is dominated by
            # images holding a handful of objects.
            object_retention_pooled=(
                g["gt_objects_in_roi"].sum() / g["gt_objects"].sum()
                if g["gt_objects"].sum() else np.nan
            ),
            object_retention_mean_per_image=g["object_retention"].mean(),
            pixel_retention_pooled=(
                g["gt_fg_px_in_roi"].sum() / g["gt_fg_px"].sum()
                if g["gt_fg_px"].sum() else np.nan
            ),
            gt_objects_total=int(g["gt_objects"].sum()),
            gt_objects_in_roi_total=int(g["gt_objects_in_roi"].sum()),
            n_pred_roi_empty=int(g["pred_roi_empty"].sum()),
            frac_pred_over_20pct_larger=float((g["area_ratio"] > 1.2).mean()),
        )
        # Manual-relative accounting, and the density argument that explains
        # why a Dice of 0.90 still retains 98 % of the objects: the part of the
        # manual ROI the prediction misses is ecDNA-poor compared with the part
        # it covers. Densities are objects per megapixel.
        if "gt_objects_in_manual" in g.columns:
            in_man = g["gt_objects_in_manual"].sum()
            both = g["gt_objects_in_both"].sum()
            man_only = g["gt_objects_manual_only"].sum()
            covered_px = g["intersection_px"].sum()
            missed_px = g["manual_only_px"].sum()
            rec.update(
                gt_objects_in_manual_total=int(in_man),
                gt_objects_manual_only_total=int(man_only),
                gt_objects_pred_only_total=int(g["gt_objects_pred_only"].sum()),
                retention_vs_manual_pooled=(both / in_man if in_man else np.nan),
                density_per_Mpx_covered=(
                    1e6 * both / covered_px if covered_px else np.nan),
                density_per_Mpx_missed=(
                    1e6 * man_only / missed_px if missed_px else np.nan),
            )
        out.append(rec)
    return pd.DataFrame(out).round(4)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Score predicted ROI against manual ROI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--repo-root",
        default="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench",
    )
    ap.add_argument(
        "--roi-pred-dir",
        default="/proj/brunk_ecdna_cv_project/Poorya/River/output/all",
        help="Predicted ROI masks. NOT River/checkpoints — that is the "
             "superseded May model.",
    )
    ap.add_argument(
        "--consistency-csv", default=None,
        help="Default: <repo>/release/manifests/"
             "dl_master_metadata_stage1_step3_consistency.csv",
    )
    ap.add_argument("--out-dir", default=None,
                    help="Default: <repo>/outputs/roi_accuracy")
    ap.add_argument("--split", choices=["train", "val", "test", "all"],
                    default="all")
    ap.add_argument("--resume", action="store_true",
                    help="Skip images already present in the partial CSV.")
    ap.add_argument("--limit", type=int, default=0,
                    help="Score only the first N images. For a smoke test.")
    ap.add_argument("--log-level", default="INFO",
                    choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    repo = Path(args.repo_root)
    out_dir = Path(args.out_dir) if args.out_dir else repo / "outputs" / "roi_accuracy"
    out_dir.mkdir(parents=True, exist_ok=True)

    cons = (
        Path(args.consistency_csv) if args.consistency_csv
        else repo / "release" / "manifests"
             / "dl_master_metadata_stage1_step3_consistency.csv"
    )
    if not cons.is_file():
        logger.error("Consistency CSV not found: %s", cons)
        logger.error("Pass --consistency-csv with the correct path.")
        return 1

    df = pd.read_csv(cons)
    df = df[df["count_mask_consistent"].fillna(False)].reset_index(drop=True)

    # Guards. A silent mismatch here would invalidate every number below.
    if len(df) != EXPECTED_N_BENCHMARK:
        logger.error("Expected %d benchmark images, found %d. Stopping.",
                     EXPECTED_N_BENCHMARK, len(df))
        return 2
    gt_total = int(df["ecDNA_gt"].sum())
    if gt_total != EXPECTED_GT_TOTAL:
        logger.error("Expected %d annotated objects, found %d. Stopping.",
                     EXPECTED_GT_TOTAL, gt_total)
        return 2
    logger.info("Metadata guards passed: %d images, %d objects.",
                len(df), gt_total)

    pred_idx = index_masks(Path(args.roi_pred_dir))
    if not pred_idx:
        logger.error(
            "No predicted ROI masks found in %s. An empty listing is a "
            "failure, not an empty result.", args.roi_pred_dir,
        )
        return 2
    logger.info("Indexed %d predicted ROI masks.", len(pred_idx))

    work = df if args.split == "all" else df[df["split"] == args.split]
    work = work.reset_index(drop=True)
    if args.limit:
        work = work.head(args.limit)

    partial_csv = out_dir / f"roi_accuracy_per_image.{args.split}.partial.csv"
    done: Dict[str, dict] = {}
    if args.resume and partial_csv.exists():
        prev = pd.read_csv(partial_csv)
        done = {str(r["uid"]): dict(r) for _, r in prev.iterrows()}
        logger.info("Resuming: %d images already scored.", len(done))

    rows = list(done.values())
    missing_pred = []
    n_total = len(work)

    for i, row in enumerate(work.itertuples(index=False), start=1):
        uid = str(row.unique_id)
        if uid in done:
            continue
        pred_path = pred_idx.get(uid)
        if pred_path is None:
            missing_pred.append(uid)
            continue
        manual_path = Path(str(row.roi_fullpath))
        gt_path = Path(str(row.gt_fullpath))
        if not manual_path.is_file():
            logger.warning("%s: manual ROI missing at %s", uid, manual_path)
            continue

        rec = score_one(uid, manual_path, pred_path, gt_path)
        if rec is None:
            continue
        rec["cell_line"] = str(row.cell_line)
        rec["split"] = str(row.split)
        rec["ecDNA_gt_metadata"] = int(row.ecDNA_gt)
        rows.append(rec)

        if len(rows) % CHECKPOINT_EVERY == 0:
            pd.DataFrame(rows).to_csv(partial_csv, index=False)
            logger.info("Progress: %d/%d scored", i, n_total)

    if not rows:
        logger.error("No images were scored. Refusing to write empty output.")
        return 2

    per_image = pd.DataFrame(rows)

    # The object count recomputed here should agree with the metadata column.
    # A systematic disagreement means the counting rule has drifted.
    cmp = per_image.dropna(subset=["gt_objects"])
    if len(cmp):
        exact = float((cmp["gt_objects"] == cmp["ecDNA_gt_metadata"]).mean())
        logger.info(
            "Recomputed object count matches metadata on %.1f %% of images "
            "(median difference %+.1f).",
            100 * exact,
            float((cmp["gt_objects"] - cmp["ecDNA_gt_metadata"]).median()),
        )
        if exact < 0.95:
            logger.warning(
                "Object counting disagrees with the metadata on more than "
                "5 %% of images. Check MIN_AREA and CONNECTIVITY before "
                "quoting any retention figure."
            )

    per_image = per_image.sort_values("uid").reset_index(drop=True)
    per_image_csv = out_dir / f"roi_accuracy_per_image.{args.split}.csv"
    per_image.to_csv(per_image_csv, index=False)
    if partial_csv.exists():
        partial_csv.unlink()

    by_line = summarise(per_image, ["cell_line"])
    by_line_split = summarise(per_image, ["cell_line", "split"])
    overall = summarise(per_image.assign(_all="all benchmark"), ["_all"])

    by_line.to_csv(out_dir / f"source_roi_accuracy_by_cell_line.{args.split}.csv",
                   index=False)
    by_line_split.to_csv(
        out_dir / f"source_roi_accuracy_by_cell_line_split.{args.split}.csv",
        index=False)
    overall.to_csv(out_dir / f"source_roi_accuracy_overall.{args.split}.csv",
                   index=False)

    print()
    print("=" * 78)
    print(f"  ROI accuracy — {len(per_image)} images, split={args.split}")
    print("=" * 78)
    cols = ["cell_line", "n_images", "iou_median", "dice_median",
            "frac_manual_covered_median", "area_ratio_median",
            "object_retention_pooled", "pixel_retention_pooled"]
    print(by_line[cols].to_string(index=False))
    print()
    print("Overall:")
    print(overall[["n_images", "iou_median", "dice_median",
                   "object_retention_pooled",
                   "pixel_retention_pooled"]].to_string(index=False))

    dens_cols = [c for c in ("cell_line", "gt_objects_in_manual_total",
                             "gt_objects_manual_only_total",
                             "retention_vs_manual_pooled",
                             "density_per_Mpx_covered",
                             "density_per_Mpx_missed")
                 if c in by_line.columns]
    if len(dens_cols) > 1:
        print()
        print("Why the geometry score and the retention diverge — objects per")
        print("megapixel inside the region the prediction covers, versus inside")
        print("the region of the manual ROI it misses:")
        print(by_line[dens_cols].to_string(index=False))
    print()
    if missing_pred:
        print(f"{len(missing_pred)} images had no predicted ROI mask. "
              f"Listed in missing_predicted_roi.txt")
        (out_dir / "missing_predicted_roi.txt").write_text(
            "\n".join(missing_pred) + "\n")
    n_empty = int(per_image["pred_roi_empty"].sum())
    if n_empty:
        print(f"{n_empty} images have an EMPTY predicted ROI. These must be "
              f"stated as an exclusion wherever retention is reported.")
    print()
    print(f"Per-image rows : {per_image_csv}")
    print(f"Summaries      : {out_dir}")
    print()
    print("Now draw the figures:")
    print(f"  python scripts/roi_accuracy_plots.py --in-dir {out_dir} "
          f"--split {args.split}")
    print(f"  python scripts/roi_example_figures.py --in-dir {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())