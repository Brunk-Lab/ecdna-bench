#!/usr/bin/env python
"""
roi_accuracy_plots.py — figures for predicted-versus-manual ROI agreement.

Reads the per-image CSV written by roi_accuracy_metrics.py and draws four
panels, per cell line, for the held-out test split and for all benchmark
images side by side.

    python scripts/roi_accuracy_plots.py --in-dir outputs/roi_accuracy

Each figure is written as PNG and SVG, with the source data beside it as
source_<name>.csv, so every plotted value is traceable to a file.

The panels are box plots with the individual images overlaid as points.
COLO320DM contributes 64 images and NCI-H716 only 70, so a box alone would
imply more data than exists; the points show the reader what the box is made
of. Boxes are drawn without outlier fliers because every point is already
shown.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("roi_plots")

# Cell-line order is fixed so every figure in the paper stacks the same way.
CELL_LINE_ORDER = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]

# Colourblind-safe, and distinct in greyscale print.
PALETTE = {
    "COLO320DM": "#4C72B0",
    "NCI-H2170": "#DD8452",
    "NCI-H716": "#55A868",
    "SNU16": "#C44E52",
}

RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.bbox": "tight",
}

PANELS = [
    ("iou", "IoU, predicted vs manual ROI", (0, 1), None),
    ("dice", "Dice, predicted vs manual ROI", (0, 1), None),
    ("object_retention", "Annotated ecDNA retained in predicted ROI", (0, 1), None),
    ("area_ratio", "Predicted ROI area / manual ROI area", None, 1.0),
]


def save_fig(fig: plt.Figure, out_dir: Path, name: str) -> None:
    for ext in ("png", "svg"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, format=ext)
        logger.info("wrote %s", path)
    plt.close(fig)


def save_source(df: pd.DataFrame, out_dir: Path, name: str) -> None:
    path = out_dir / f"source_{name}.csv"
    df.to_csv(path, index=False)
    logger.info("wrote %s", path)


def box_strip(ax, df: pd.DataFrame, metric: str, lines: list, rng, ref) -> None:
    """Box plot per cell line with every image overlaid as a point."""
    data, positions, labels, colours = [], [], [], []
    for i, line in enumerate(lines):
        vals = df.loc[df["cell_line"] == line, metric].dropna().values
        if not len(vals):
            continue
        data.append(vals)
        positions.append(i)
        labels.append(f"{line}\nn = {len(vals)}")
        colours.append(PALETTE.get(line, "#888888"))

    if not data:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes)
        return

    bp = ax.boxplot(
        data, positions=positions, widths=0.55, patch_artist=True,
        showfliers=False, medianprops=dict(color="#111111", linewidth=1.4),
        whiskerprops=dict(color="#555555", linewidth=0.9),
        capprops=dict(color="#555555", linewidth=0.9),
        boxprops=dict(linewidth=0.9, edgecolor="#555555"),
    )
    for patch, colour in zip(bp["boxes"], colours):
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)

    rng_gen = np.random.default_rng(0)  # jitter is deterministic
    for vals, pos, colour in zip(data, positions, colours):
        jitter = rng_gen.uniform(-0.16, 0.16, size=len(vals))
        ax.scatter(np.full(len(vals), pos) + jitter, vals, s=6, alpha=0.45,
                   color=colour, linewidths=0, zorder=3)

    if ref is not None:
        ax.axhline(ref, color="#111111", linestyle="--", linewidth=0.8,
                   zorder=1)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    if rng is not None:
        ax.set_ylim(*rng)


def figure_for_subset(df: pd.DataFrame, title: str) -> plt.Figure:
    lines = [c for c in CELL_LINE_ORDER if c in set(df["cell_line"])]
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6))
    for ax, (metric, label, rng, ref) in zip(axes.ravel(), PANELS):
        box_strip(ax, df, metric, lines, rng, ref)
        ax.set_ylabel(label)
    fig.suptitle(title, y=0.995, fontsize=11)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bar figures
# ---------------------------------------------------------------------------

SPLIT_ORDER = ["train", "val", "test"]
SPLIT_COLOURS = {
    "train": "#8C8C8C",
    "val": "#7B9FD4",
    "test": "#2F5C9E",
    "all": "#1A1A1A",
}

# The three headline quantities. `pooled` marks a ratio of totals, which has no
# per-image distribution and therefore gets no error bar — see note below.
BAR_METRICS = [
    ("iou", "IoU", False),
    ("dice", "Dice", False),
    ("object_retention", "ecDNA retained", True),
]


def _stat(g: pd.DataFrame, metric: str, pooled: bool):
    """Return (value, lower_error, upper_error) for one group.

    A pooled retention is total objects retained ÷ total objects annotated.
    It is not the average of per-image fractions, and it has no interquartile
    range, because it is one ratio rather than a distribution. Reporting an
    error bar on it would invent a spread that does not exist. Images holding
    four objects would otherwise dominate a per-image mean, which is why the
    pooled form is the one plotted for low-burden lines.
    """
    if g.empty:
        return np.nan, 0.0, 0.0
    if pooled:
        denom = g["gt_objects"].sum()
        if not denom:
            return np.nan, 0.0, 0.0
        return g["gt_objects_in_roi"].sum() / denom, 0.0, 0.0
    v = g[metric].dropna()
    if v.empty:
        return np.nan, 0.0, 0.0
    med = v.median()
    return med, max(med - v.quantile(0.25), 0), max(v.quantile(0.75) - med, 0)


def _label_bars(ax, bars, values, fmt="{:.3f}") -> None:
    for b, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.012,
                fmt.format(v), ha="center", va="bottom", fontsize=7)


def figure_bars_by_split(df: pd.DataFrame) -> plt.Figure:
    """Three metrics, grouped by split, with a whole-dataset group."""
    groups = [(s, df[df["split"] == s]) for s in SPLIT_ORDER]
    groups.append(("all", df))
    labels = [f"{s if s != 'all' else 'all benchmark'}\nn = {len(g)}"
              for s, g in groups]

    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    width = 0.26
    x = np.arange(len(groups))
    for k, (metric, label, pooled) in enumerate(BAR_METRICS):
        vals, los, his = zip(*[_stat(g, metric, pooled) for _, g in groups])
        offset = (k - 1) * width
        bars = ax.bar(x + offset, vals, width,
                      yerr=[los, his] if any(los + his) else None,
                      capsize=2.5, label=label, linewidth=0.6,
                      edgecolor="#333333",
                      color=["#B8CBE8", "#7FB07F", "#E8B87F"][k])
        _label_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Median (IoU, Dice) or pooled fraction (retention)")
    # The legend sits above the axes: every bar here lands between 0.75 and
    # 1.0, so any in-axes corner would be covered.
    ax.set_title("Predicted vs manual ROI, by split", pad=26)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout()
    return fig


def figure_bars_by_cell_line_and_split(df: pd.DataFrame,
                                       zoom: bool = False) -> plt.Figure:
    """One panel per metric; cell lines on x, splits as grouped bars.

    Drawn twice. The zero-baseline version is the honest one and carries the
    real message — the scores barely move across splits, so the bars are level.
    The zoomed version exists because at a 0-1 scale a difference of 0.02 is
    invisible, and it is labelled as truncated on every panel, because a bar
    chart that does not start at zero misrepresents ratios unless it says so.
    """
    lines = [c for c in CELL_LINE_ORDER if c in set(df["cell_line"])]
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 7.4), sharex=True)
    handles = labels = None

    for ax, (metric, label, pooled) in zip(axes, BAR_METRICS):
        x = np.arange(len(lines))
        width = 0.20
        lo_seen, hi_seen = [], []
        for k, split in enumerate(SPLIT_ORDER + ["all"]):
            vals, los, his = [], [], []
            for line in lines:
                sub = df[df["cell_line"] == line]
                if split != "all":
                    sub = sub[sub["split"] == split]
                v, lo, hi = _stat(sub, metric, pooled)
                vals.append(v)
                los.append(lo)
                his.append(hi)
            offset = (k - 1.5) * width
            ax.bar(x + offset, vals, width,
                   yerr=[los, his] if any(los + his) else None,
                   capsize=2, label=split if split != "all" else "all benchmark",
                   color=SPLIT_COLOURS[split], linewidth=0.5,
                   edgecolor="#333333")
            lo_seen += [v - lo for v in vals if not np.isnan(v)]
            hi_seen += [v + hi for v in vals if not np.isnan(v)]

        ax.set_ylabel(label)
        if zoom and lo_seen:
            bottom = max(0.0, min(lo_seen) - 0.04)
            ax.set_ylim(bottom, min(1.02, max(hi_seen) + 0.03))
            ax.text(0.004, 0.96, "y-axis truncated", transform=ax.transAxes,
                    fontsize=6.5, style="italic", color="#666666",
                    va="top", ha="left")
        else:
            ax.set_ylim(0, 1.05)
        if handles is None:
            handles, labels = ax.get_legend_handles_labels()

    axes[-1].set_xticks(np.arange(len(lines)))
    axes[-1].set_xticklabels(lines)
    fig.suptitle(
        "Predicted vs manual ROI, by cell line and split"
        + (" — truncated axes" if zoom else ""),
        y=0.985, fontsize=10.5)
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    return fig


def figure_geometry_vs_retention(df: pd.DataFrame) -> plt.Figure:
    """The point: the geometry scores sit near 0.8-0.9, retention near 0.98.

    Boundary disagreement between the predicted and manual ROI happens where
    there is little ecDNA, so a Dice well below 1 still keeps almost every
    annotated object.
    """
    lines = [c for c in CELL_LINE_ORDER if c in set(df["cell_line"])]
    groups = [(l, df[df["cell_line"] == l]) for l in lines]
    groups.append(("all benchmark", df))

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    x = np.arange(len(groups))
    width = 0.26
    series = [
        ("iou", "IoU (geometry)", "#B8CBE8", False),
        ("dice", "Dice (geometry)", "#8FA9CC", False),
        ("object_retention", "ecDNA retained", "#2F7A4F", True),
    ]
    for k, (metric, label, colour, pooled) in enumerate(series):
        vals = [_stat(g, metric, pooled)[0] for _, g in groups]
        bars = ax.bar(x + (k - 1) * width, vals, width, label=label,
                      color=colour, linewidth=0.6, edgecolor="#333333")
        _label_bars(ax, bars, vals)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{n}\nn = {len(g)}" for n, g in groups])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Median (geometry) or pooled fraction (retention)")
    ax.set_title("ROI boundary agreement vs annotated ecDNA retained", pad=26)
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, 1.11))
    fig.tight_layout()
    return fig


def figure_test_vs_all(df: pd.DataFrame, metric: str, label: str,
                       rng, ref) -> plt.Figure:
    """One metric, test split beside all benchmark images."""
    lines = [c for c in CELL_LINE_ORDER if c in set(df["cell_line"])]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    subsets = [
        (df[df["split"] == "test"], "Held-out test split"),
        (df, "All benchmark images"),
    ]
    for ax, (sub, name) in zip(axes, subsets):
        box_strip(ax, sub, metric, lines, rng, ref)
        ax.set_title(f"{name}  (n = {len(sub)})")
    axes[0].set_ylabel(label)
    fig.tight_layout()
    return fig


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--in-dir", default="outputs/roi_accuracy")
    ap.add_argument("--out-dir", default=None,
                    help="Default: <in-dir>/figures")
    ap.add_argument("--split", default="all",
                    help="Which per-image CSV to read.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)-8s %(message)s", force=True)

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_csv = in_dir / f"roi_accuracy_per_image.{args.split}.csv"
    if not per_image_csv.is_file():
        logger.error("Not found: %s", per_image_csv)
        logger.error("Run roi_accuracy_metrics.py first.")
        return 1

    df = pd.read_csv(per_image_csv)
    if df.empty:
        logger.error("%s is empty. Refusing to draw figures from no data.",
                     per_image_csv)
        return 2
    logger.info("Loaded %d scored images.", len(df))

    plt.rcParams.update(RC)

    # Images with an empty predicted ROI have an undefined retention and would
    # drag every distribution toward zero for a reason that is a separate
    # finding. They are excluded from the figures and reported as a count.
    n_empty = int(df.get("pred_roi_empty", pd.Series(dtype=bool)).sum())
    drawn = df[~df.get("pred_roi_empty", False)].copy()
    if n_empty:
        logger.warning(
            "%d images have an empty predicted ROI and are excluded from the "
            "figures. State this exclusion in the caption.", n_empty)

    fig = figure_for_subset(
        drawn[drawn["split"] == "test"],
        "Predicted vs manual ROI — held-out test split")
    save_fig(fig, out_dir, "figR1_roi_accuracy_test")
    save_source(drawn[drawn["split"] == "test"], out_dir,
                "figR1_roi_accuracy_test")

    fig = figure_for_subset(
        drawn, "Predicted vs manual ROI — all benchmark images")
    save_fig(fig, out_dir, "figR2_roi_accuracy_all_benchmark")
    save_source(drawn, out_dir, "figR2_roi_accuracy_all_benchmark")

    for metric, label, rng, ref in PANELS:
        fig = figure_test_vs_all(drawn, metric, label, rng, ref)
        save_fig(fig, out_dir, f"figR3_{metric}_test_vs_all")

    # Bar figures.
    fig = figure_bars_by_split(drawn)
    save_fig(fig, out_dir, "figR4_bars_by_split")

    fig = figure_bars_by_cell_line_and_split(drawn, zoom=False)
    save_fig(fig, out_dir, "figR5_bars_by_cell_line_and_split")

    fig = figure_bars_by_cell_line_and_split(drawn, zoom=True)
    save_fig(fig, out_dir, "figR5z_bars_by_cell_line_and_split_zoomed")

    fig = figure_geometry_vs_retention(drawn)
    save_fig(fig, out_dir, "figR6_geometry_vs_retention")

    # Source data for the bar figures: every plotted value, in one file.
    bar_rows = []
    for scope_name, sub in (
        [(f"split: {s}", drawn[drawn["split"] == s]) for s in SPLIT_ORDER]
        + [("all benchmark", drawn)]
        + [(f"{l} / {s}", drawn[(drawn["cell_line"] == l) & (drawn["split"] == s)])
           for l in CELL_LINE_ORDER for s in SPLIT_ORDER]
        + [(f"{l} / all", drawn[drawn["cell_line"] == l])
           for l in CELL_LINE_ORDER]
    ):
        if sub.empty:
            continue
        rec = {"group": scope_name, "n_images": len(sub)}
        for metric, label, pooled in BAR_METRICS:
            v, lo, hi = _stat(sub, metric, pooled)
            rec[f"{metric}_value"] = round(v, 4) if not np.isnan(v) else np.nan
            if not pooled:
                rec[f"{metric}_q25"] = round(v - lo, 4)
                rec[f"{metric}_q75"] = round(v + hi, 4)
        rec["gt_objects_total"] = int(sub["gt_objects"].sum())
        rec["gt_objects_in_roi_total"] = int(sub["gt_objects_in_roi"].sum())
        bar_rows.append(rec)
    save_source(pd.DataFrame(bar_rows), out_dir, "roi_accuracy_bar_values")

    # One tidy summary table for the caption and the Results paragraph.
    rows = []
    for scope, sub in (("test", drawn[drawn["split"] == "test"]),
                       ("all benchmark", drawn)):
        for line in CELL_LINE_ORDER:
            g = sub[sub["cell_line"] == line]
            if g.empty:
                continue
            rows.append({
                "scope": scope,
                "cell_line": line,
                "n_images": len(g),
                "iou_median": round(g["iou"].median(), 3),
                "dice_median": round(g["dice"].median(), 3),
                "frac_manual_covered_median": round(
                    g["frac_manual_covered"].median(), 3),
                "area_ratio_median": round(g["area_ratio"].median(), 3),
                "object_retention_pooled": round(
                    g["gt_objects_in_roi"].sum() / g["gt_objects"].sum(), 3)
                if g["gt_objects"].sum() else np.nan,
                "gt_objects_total": int(g["gt_objects"].sum()),
            })
    summary = pd.DataFrame(rows)
    save_source(summary, out_dir, "roi_accuracy_summary_table")

    print()
    print(summary.to_string(index=False))
    print()
    if n_empty:
        print(f"Excluded from figures: {n_empty} images with an empty "
              f"predicted ROI.")
    print(f"Figures and source data: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())