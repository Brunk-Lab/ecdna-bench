#!/usr/bin/env python
"""
roi_example_figures.py — best and worst predicted ROI masks, one per figure.

Picks the highest- and lowest-scoring images from the per-image CSV written by
roi_accuracy_metrics.py and draws each as its own figure: the RGB image with the
manual ROI and the predicted ROI outlined, and every annotated ecDNA object that
falls outside the predicted ROI marked individually.

    python scripts/roi_example_figures.py --in-dir outputs/roi_accuracy
    python scripts/roi_example_figures.py --metric dice --n 3

Each example is written separately as
``figR7_best_1_<uid>.png`` / ``figR8_worst_1_<uid>.png`` plus SVG, with a
one-row source CSV beside it, so any single panel can go into a figure or a
slide without carrying the others.

WHY THE LOST OBJECTS ARE MARKED
-------------------------------
The geometry scores and the retention figure disagree — IoU sits near 0.82
while retention sits near 0.98. Marking the objects that actually fall outside
the predicted ROI shows why: on a good example there are none to mark, and on a
bad one the disagreement is usually a region of the spread that holds few
objects. A reader who sees only the two outlines cannot tell the difference
between a boundary that moved through empty space and one that cut through a
cluster.

SELECTION
---------
Ranked by ``--metric``, default IoU. Images with an empty predicted ROI are
excluded from the "worst" ranking by default and reported separately, because
an empty prediction is a different failure from a poorly shaped one and would
otherwise occupy every worst slot. Pass ``--include-empty`` to rank them in.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("roi_examples")

MIN_AREA = 3
CONNECTIVITY = 8

COLOUR_MANUAL = "#FFC845"   # amber
COLOUR_PRED = "#3FA9F5"     # blue
COLOUR_LOST = "#FF3B30"     # red

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "savefig.bbox": "tight",
}


def read_binary(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 3:
        img = img.max(axis=2)
    return img > 0


def object_centroids(mask) -> np.ndarray:
    n, _lab, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=CONNECTIVITY)
    if n <= 1:
        return np.empty((0, 2), float)
    keep = stats[1:, cv2.CC_STAT_AREA] >= MIN_AREA
    return cents[1:][keep]


def draw_example(
    uid: str,
    rgb_path: Path,
    manual_path: Path,
    pred_path: Path,
    gt_path: Path,
    row: pd.Series,
    title_prefix: str,
) -> plt.Figure | None:
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        logger.warning("%s: cannot read RGB at %s", uid, rgb_path)
        return None
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

    manual = read_binary(manual_path)
    pred = read_binary(pred_path)
    gt = read_binary(gt_path)
    if manual is None or pred is None:
        logger.warning("%s: cannot read a mask", uid)
        return None
    if manual.shape != rgb.shape[:2] or pred.shape != rgb.shape[:2]:
        logger.warning("%s: mask and image shapes differ, skipped", uid)
        return None

    h, w = rgb.shape[:2]
    fig_w = 7.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * h / w))
    ax.imshow(rgb)

    # Outlines rather than filled overlays, so the underlying signal stays
    # visible. A filled ROI would hide exactly the pixels the reader wants.
    ax.contour(manual.astype(float), levels=[0.5], colors=[COLOUR_MANUAL],
               linewidths=1.6)
    ax.contour(pred.astype(float), levels=[0.5], colors=[COLOUR_PRED],
               linewidths=1.6, linestyles="--")

    n_lost = 0
    if gt is not None and gt.shape == pred.shape:
        cents = object_centroids(gt)
        if len(cents):
            xs = np.clip(np.round(cents[:, 0]).astype(int), 0, w - 1)
            ys = np.clip(np.round(cents[:, 1]).astype(int), 0, h - 1)
            lost = manual[ys, xs] & ~pred[ys, xs]
            n_lost = int(np.count_nonzero(lost))
            if n_lost:
                ax.scatter(xs[lost], ys[lost], s=26, marker="x",
                           c=COLOUR_LOST, linewidths=1.1, zorder=5)

    handles = [
        mpatches.Patch(facecolor="none", edgecolor=COLOUR_MANUAL,
                       linewidth=1.6, label="manual ROI"),
        mpatches.Patch(facecolor="none", edgecolor=COLOUR_PRED,
                       linewidth=1.6, linestyle="--", label="predicted ROI"),
    ]
    if n_lost:
        handles.append(plt.Line2D([], [], color=COLOUR_LOST, marker="x",
                                  linestyle="none", markersize=6,
                                  label=f"ecDNA outside predicted ROI "
                                        f"(n = {n_lost})"))
    else:
        handles.append(plt.Line2D([], [], color="none", marker="",
                                  linestyle="none",
                                  label="no ecDNA lost"))
    ax.legend(handles=handles, loc="upper right", frameon=True,
              framealpha=0.85, fontsize=7.5)

    stat_bits = [
        f"IoU {row['iou']:.3f}",
        f"Dice {row['dice']:.3f}",
        f"manual ROI covered {row['frac_manual_covered']:.3f}",
        f"area ratio {row['area_ratio']:.3f}",
    ]
    if not pd.isna(row.get("object_retention", np.nan)):
        stat_bits.append(f"ecDNA retained {row['object_retention']:.3f}")
    ax.text(
        0.012, 0.012, "   ·   ".join(stat_bits), transform=ax.transAxes,
        fontsize=7.5, va="bottom", ha="left", color="#111111",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.82,
                  edgecolor="#999999", linewidth=0.5),
    )

    ax.set_title(
        f"{title_prefix} — {row['cell_line']} · {row['split']} split\n{uid}",
        fontsize=9,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", default="outputs/roi_accuracy")
    ap.add_argument("--out-dir", default=None, help="Default: <in-dir>/figures")
    ap.add_argument("--split", default="all",
                    help="Which per-image CSV to read.")
    ap.add_argument(
        "--repo-root",
        default="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench")
    ap.add_argument(
        "--roi-pred-dir",
        default="/proj/brunk_ecdna_cv_project/Poorya/River/output/all")
    ap.add_argument("--consistency-csv", default=None)
    ap.add_argument("--metric", default="iou",
                    choices=["iou", "dice", "object_retention",
                             "frac_manual_covered"])
    ap.add_argument("--n", type=int, default=2,
                    help="How many best and how many worst.")
    ap.add_argument("--restrict-split", default=None,
                    choices=["train", "val", "test"],
                    help="Draw examples only from this split.")
    ap.add_argument("--include-empty", action="store_true",
                    help="Allow empty predicted ROIs into the worst ranking.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    repo = Path(args.repo_root)

    per_image_csv = in_dir / f"roi_accuracy_per_image.{args.split}.csv"
    if not per_image_csv.is_file():
        logger.error("Not found: %s — run roi_accuracy_metrics.py first.",
                     per_image_csv)
        return 1
    df = pd.read_csv(per_image_csv)
    if df.empty:
        logger.error("%s is empty.", per_image_csv)
        return 2

    cons = (Path(args.consistency_csv) if args.consistency_csv
            else repo / "release" / "manifests"
                 / "dl_master_metadata_stage1_step3_consistency.csv")
    if not cons.is_file():
        logger.error("Consistency CSV not found: %s", cons)
        return 1
    meta = pd.read_csv(cons).set_index("unique_id")

    if args.restrict_split:
        df = df[df["split"] == args.restrict_split]
        if df.empty:
            logger.error("No images in split %s.", args.restrict_split)
            return 2

    ranked = df.dropna(subset=[args.metric])
    n_empty = int(ranked.get("pred_roi_empty", pd.Series(dtype=bool)).sum())
    if not args.include_empty and "pred_roi_empty" in ranked.columns:
        ranked = ranked[~ranked["pred_roi_empty"]]
    ranked = ranked.sort_values(args.metric)

    if len(ranked) < 2 * args.n:
        logger.warning("Only %d images available; reducing n.", len(ranked))
        args.n = max(1, len(ranked) // 2)

    worst = ranked.head(args.n)
    best = ranked.tail(args.n).iloc[::-1]

    plt.rcParams.update(RC)
    written = []

    for kind, frame, prefix, fignum in (
        ("best", best, "Best predicted ROI", "figR7"),
        ("worst", worst, "Worst predicted ROI", "figR8"),
    ):
        for rank, (_, row) in enumerate(frame.iterrows(), start=1):
            uid = str(row["uid"])
            if uid not in meta.index:
                logger.warning("%s: not in the consistency CSV, skipped", uid)
                continue
            m = meta.loc[uid]
            pred_path = Path(args.roi_pred_dir) / f"{uid}.png"
            if not pred_path.is_file():
                cands = list(Path(args.roi_pred_dir).glob(f"{uid}.*"))
                if not cands:
                    logger.warning("%s: no predicted ROI mask found", uid)
                    continue
                pred_path = cands[0]

            fig = draw_example(
                uid,
                Path(str(m["rgb_fullpath"])),
                Path(str(m["roi_fullpath"])),
                pred_path,
                Path(str(m["gt_fullpath"])),
                row,
                f"{prefix} ({args.metric} rank {rank})",
            )
            if fig is None:
                continue
            name = f"{fignum}_{kind}_{rank}_{uid}"
            for ext in ("png", "svg"):
                fig.savefig(out_dir / f"{name}.{ext}", format=ext)
            plt.close(fig)
            pd.DataFrame([row]).to_csv(out_dir / f"source_{name}.csv",
                                       index=False)
            written.append((kind, rank, uid, float(row[args.metric])))
            logger.info("wrote %s", out_dir / f"{name}.png")

    print()
    print(f"Ranked by {args.metric}"
          + (f", split={args.restrict_split}" if args.restrict_split else "")
          + f", {len(ranked)} images considered.")
    for kind, rank, uid, val in written:
        print(f"  {kind:5s} #{rank}  {args.metric} = {val:.4f}   {uid}")
    if n_empty and not args.include_empty:
        print(f"\n{n_empty} images with an empty predicted ROI were excluded "
              f"from the ranking. Pass --include-empty to rank them in.")
    print(f"\nFigures: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
