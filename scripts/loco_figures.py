#!/usr/bin/env python
"""
loco_figures.py — candidate figures for the leave-one-cell-line-out result.

Run after loco_paper_table.py.

    python scripts/loco_figures.py --out-root outputs/eccount_loco

Produces six alternative figures, each as PNG + SVG with its own source CSV, so
the panels can be compared side by side and one or two chosen. Nothing here is
computed: every value is read from
``source_loco_paper_table_<policy>.csv`` or from the locked per-cell-line CSV.

THE FOUR CLAIMS, AND WHICH FIGURE CARRIES EACH
----------------------------------------------
figL1  dumbbell            ecCount transfers to an unseen cell line at a
                           modest, quantified cost. The main claim.
figL2  H2170 decomposition the NCI-H2170 drop is cell-line novelty, not the
                           smaller training set. Answers the one objection a
                           reviewer will raise, using the size-matched control.
figL3  against baselines   ecCount that has never seen a cell line scores above
                           baselines that trained on it. Potentially the
                           strongest panel — but it depends on per-cell-line
                           baseline values, so the script checks they are
                           present rather than assuming.
figL4  cost bars           the same content as figL1 reduced to one number per
                           cell line. The simplest version.
figL5  signed count bias   the mechanism: the F1 drop is driven by systematic
                           under-counting, not uniform degradation.
figL6  composite           figL1 + figL2 + figL3 as one three-panel display
                           item, which is what a Resource paper has room for.

SCOPE, AND WHY THE SCRIPT CHECKS IT
-----------------------------------
The leave-one-out runs score every row of the held-out line; the size-matched
control scores that line's test rows only. The locked per-cell-line CSV may be
either scope. Comparing an all-rows number against a test-rows number puts a
difference in the paper that is partly a change of image set, so the script
reads ``n_images`` from the locked CSV, states which scope it found, and pairs
each figure with the matching leave-one-out column.

It also pools the locked per-cell-line TP/FP/FN back to a single value and
reports it against the locked headline figures (0.942 peaks, 0.917 threshold
over all 1,145 images; 0.939 / 0.916 on the 175-image test split). If neither
matches, the CSV is not what the figures assume and the script says so.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

logger = logging.getLogger("loco_figs")

# --- palette (validated: ordinal blue ramp, diverging blue/red) -------------
BLUE_L = "#86b6ef"   # blue 250 — in-distribution / context
BLUE_M = "#3987e5"   # blue 400 — intermediate
BLUE_D = "#184f95"   # blue 600 — leave-one-out / the subject
RED = "#e34948"      # diverging warm pole (over-counting)
GREY = "#898781"     # muted ink, de-emphasised marks
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

CELL_LINES = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]
PEAKS = "ecCount (peaks)"
THRESH = "ecCount (threshold mask)"

# Locked anchors used only to identify the scope of the per-cell-line CSV.
LOCKED = {
    ("all", PEAKS): 0.942, ("all", THRESH): 0.917,
    ("test", PEAKS): 0.939, ("test", THRESH): 0.916,
}
LOCKED_BIAS_PEAKS = 0.4   # paper model, pooled over all 1,145 images


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "axes.titleweight": "semibold",
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "svg.fonttype": "none",
    })


def sgn(v: float, dp: int = 1) -> str:
    """Signed number with a typographic minus, matching the axis ticks."""
    return f"{v:+.{dp}f}".replace("-", "\u2212")


def neg(v: float, dp: int = 3) -> str:
    return f"\u2212{abs(v):.{dp}f}"


def legend_below(ax, handles, ncol=2, dy=-0.30) -> None:
    """Legends go under the plot. Inside the axes they collide with the data."""
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, dy), ncol=ncol, frameon=False,
              handletextpad=0.4, columnspacing=1.4, borderpad=0)


def tidy(ax, xgrid=True) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    if xgrid:
        ax.xaxis.grid(True)
        ax.yaxis.grid(False)


def save(fig, outdir: Path, name: str, src: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", **kw)
    src.to_csv(outdir / f"source_{name}.csv", index=False)
    plt.close(fig)
    logger.info("wrote %s.{png,svg} + source_%s.csv", name, name)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_indist(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"model", "cell_line", "object_f1"}
    if not need.issubset(df.columns):
        logger.error("%s lacks %s (has %s)", path, need - set(df.columns),
                     list(df.columns))
        sys.exit(2)
    return df


def load_by_scope(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """Read in_distribution_by_scope.csv (rescore_in_distribution_by_scope.py).

    Supplies the in-distribution F1 restricted to the held-out test images —
    the only scope on which an in-distribution score can be compared with a
    leave-one-out score without the in-distribution side including images its
    own model trained on.
    """
    if path is None:
        return None
    if not path.is_file():
        logger.warning("Not found: %s — falling back to the all-rows "
                       "per-cell-line CSV.", path)
        return None
    df = pd.read_csv(path)
    need = {"model", "cell_line", "scope", "object_f1"}
    if not need.issubset(df.columns):
        logger.warning("%s lacks %s; ignoring it.", path, need - set(df.columns))
        return None
    logger.info("Scope-clean in-distribution values loaded from %s.", path)
    return df


def indist_test_rows(by_scope: Optional[pd.DataFrame], model: str,
                     line: str) -> Optional[float]:
    if by_scope is None:
        return None
    r = by_scope[(by_scope.model.astype(str).str.strip() == model)
                 & (by_scope.cell_line.astype(str).str.strip() == line)
                 & (by_scope.scope.astype(str).str.strip() == "test rows")]
    return float(r["object_f1"].iloc[0]) if len(r) else None


def check_scope(ind: pd.DataFrame) -> str:
    """Identify whether the locked per-cell-line CSV is all-rows or test-only."""
    scope = "unknown"
    if "n_images" in ind.columns:
        h = ind[ind["cell_line"] == "NCI-H2170"]["n_images"].dropna().unique()
        if len(h) == 1:
            n = int(h[0])
            scope = {888: "all", 134: "test"}.get(n, "unknown")
            logger.info("Per-cell-line CSV covers %d NCI-H2170 images → "
                        "scope '%s'.", n, scope)

    if {"tp", "fp", "fn"}.issubset(ind.columns):
        for model in (PEAKS, THRESH):
            sub = ind[ind["model"].astype(str).str.lower().map(
                lambda s: ("peak" in s) if model == PEAKS
                else ("peak" not in s and ("mask" in s or "thresh" in s)))]
            sub = sub[sub["cell_line"].isin(CELL_LINES)]
            if sub.empty:
                continue
            tp, fp, fn = sub.tp.sum(), sub.fp.sum(), sub.fn.sum()
            p = tp / (tp + fp) if tp + fp else 0
            r = tp / (tp + fn) if tp + fn else 0
            pooled = 2 * p * r / (p + r) if p + r else 0
            best = min(("all", "test"),
                       key=lambda s: abs(pooled - LOCKED[(s, model)]))
            logger.info(
                "%-26s pools to %.4f across the four lines "
                "(locked all-1,145 %.3f, test %.3f → matches '%s', "
                "off by %.4f)",
                model, pooled, LOCKED[("all", model)], LOCKED[("test", model)],
                best, abs(pooled - LOCKED[(best, model)]),
            )
            if abs(pooled - LOCKED[(best, model)]) > 0.002:
                logger.warning(
                    "  %s does not reconcile with either locked value. The "
                    "per-cell-line CSV may not be what these figures assume — "
                    "check it before using the figures.", model)
            if scope == "unknown":
                scope = best
    if scope == "unknown":
        logger.warning("Could not establish the scope of the per-cell-line "
                       "CSV; assuming all-rows.")
        scope = "all"
    return scope


def build_frame(tab: pd.DataFrame, ind: pd.DataFrame, model: str,
                scope: str) -> pd.DataFrame:
    """One row per cell line: in-distribution, leave-one-out, cost, bias."""
    kind = ("leave-one-out" if scope == "all"
            else "leave-one-out (test-restricted)")
    rows = []
    for line in CELL_LINES:
        loco = tab[(tab.held_out_cell_line == line) & (tab.model == model)
                   & (tab.run_kind == kind)]
        if loco.empty:   # test-restricted exists for NCI-H2170 only
            loco = tab[(tab.held_out_cell_line == line) & (tab.model == model)
                       & (tab.run_kind == "leave-one-out")]
        if loco.empty:
            continue
        loco = loco.iloc[0]
        i = ind[(ind.cell_line == line)
                & (ind.model.astype(str).str.strip() == model)]
        in_f1 = float(i["object_f1"].iloc[0]) if len(i) else np.nan
        rows.append({
            "cell_line": line,
            "model": model,
            "scope": loco.get("scope", ""),
            "n_images": int(loco.n_images),
            "in_distribution_obj_f1": in_f1,
            "leave_one_out_obj_f1": float(loco.obj_f1),
            "cost_of_novelty": in_f1 - float(loco.obj_f1),
            "count_mae": float(loco.count_mae),
            "rel_mae": float(loco.rel_mae),
            "mean_signed_bias": float(loco.mean_signed_bias),
            "reference_obj_f1": float(loco.get("reference_obj_f1", np.nan)),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_dumbbell(d: pd.DataFrame, outdir: Path, model: str, scope: str,
                 name="figL1_generalisation_dumbbell",
                 diff_annotation: str = "none") -> None:
    """Endpoints, and optionally the gap between them.

    ``diff_annotation`` exists because a panel that prints two F1 values to
    three decimals AND their difference can contradict itself: 0.9435 and
    0.8483 print as 0.944 and 0.848, whose difference is 0.096, while the true
    difference is 0.0953 and prints as 0.095. Both numbers are correct; they
    cannot both appear on one panel at three decimals.

        none       endpoints only. Nothing can disagree. Default.
        full       the true difference. May not equal what a reader subtracts.
        displayed  the difference of the printed endpoints. Always reconciles
                   on the page, but may differ from the source CSV.
    """
    d = d.sort_values("cost_of_novelty")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(5.6, 2.9))

    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.leave_one_out_obj_f1, r.in_distribution_obj_f1], [yi, yi],
                color=AXIS, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter(r.in_distribution_obj_f1, yi, s=52, color=BLUE_L,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.scatter(r.leave_one_out_obj_f1, yi, s=52, color=BLUE_D,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.annotate(f"{r.leave_one_out_obj_f1:.3f}",
                    (r.leave_one_out_obj_f1, yi), xytext=(-7, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7.5, color=INK)
        ax.annotate(f"{r.in_distribution_obj_f1:.3f}",
                    (r.in_distribution_obj_f1, yi), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=INK2)
        if diff_annotation != "none":
            gap = (r.cost_of_novelty if diff_annotation == "full"
                   else round(r.in_distribution_obj_f1, 3)
                        - round(r.leave_one_out_obj_f1, 3))
            mid = (r.leave_one_out_obj_f1 + r.in_distribution_obj_f1) / 2
            ax.annotate(neg(gap), (mid, yi), xytext=(0, 7),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=7, color=INK2)

    ax.set_yticks(y, d.cell_line)
    ax.set_xlabel("Object-level F1")
    span = d.in_distribution_obj_f1.max() - d.leave_one_out_obj_f1.min()
    ax.set_xlim(d.leave_one_out_obj_f1.min() - 0.42 * span,
                d.in_distribution_obj_f1.max() + 0.30 * span)
    ax.set_ylim(-0.55, len(d) - 0.35)
    ax.set_title(f"Cost of an unseen cell line — {model}")
    legend_below(ax, [
        Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE_L,
               label="trained on this line (in-distribution)"),
        Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE_D,
               label="never trained on this line"),
    ], ncol=2, dy=-0.24)
    tidy(ax)
    save(fig, outdir, name, d)


def fig_decomposition(tab: pd.DataFrame, ind: pd.DataFrame, outdir: Path,
                      model: str, truncate: bool = False,
                      name="figL2_h2170_decomposition",
                      by_scope: Optional[pd.DataFrame] = None) -> None:
    """Paper model -> size-matched control -> leave-one-out, on 134 test images.

    Two versions are written. The zero-baseline one is honest about magnitude
    but renders a 0.015 step as a sliver; the truncated one makes both steps
    readable and says so on the panel, the same convention as figR5/figR5z.
    """
    def get(kind):
        s_ = tab[(tab.held_out_cell_line == "NCI-H2170") & (tab.model == model)
                 & (tab.run_kind == kind)]
        return None if s_.empty else s_.iloc[0]

    loco = get("leave-one-out (test-restricted)")
    if loco is None:
        loco = get("leave-one-out")
    ctrl = get("size-matched control")
    i = ind[(ind.cell_line == "NCI-H2170")
            & (ind.model.astype(str).str.strip() == model)]
    if loco is None or ctrl is None or i.empty:
        logger.warning("figL2 needs both NCI-H2170 runs and the locked value; "
                       "skipping.")
        return
    # The control and the leave-one-out run are both scored on the 134
    # NCI-H2170 test images. Taking the top bar from the all-rows CSV would put
    # a train-inclusive score (the released model saw 621 of those 888 images)
    # on the same axis as two held-out ones, and the "training-set size" arrow
    # would then measure a change of image set as well as of training volume.
    paper_all = float(i["object_f1"].iloc[0])
    paper_test = indist_test_rows(by_scope, model, "NCI-H2170")
    if paper_test is None:
        paper = paper_all
        indist_scope = "all rows (SCOPE-MIXED)"
        logger.warning(
            "%s: the top bar uses the all-rows in-distribution value (%.4f), "
            "which includes images the released model trained on, while the "
            "other two bars are the 134 test images. Pass "
            "--in-distribution-by-scope to make this panel like-for-like.",
            name, paper_all)
    else:
        paper = paper_test
        indist_scope = "test rows"
        logger.info("%s: top bar %.4f (test rows) rather than %.4f (all rows) "
                    "— all three conditions now cover the same 134 images.",
                    name, paper_test, paper_all)

    vals = [paper, float(ctrl.obj_f1), float(loco.obj_f1)]
    labs = ["Full training set\n(800 images, sees NCI-H2170)",
            "Size-matched control\n(179 images, sees NCI-H2170)",
            "Leave-one-out\n(179 images, never sees NCI-H2170)"]
    src = pd.DataFrame({
        "condition": ["full training set", "size-matched control",
                      "leave-one-out"],
        "n_train_images": [800, 179, 179],
        "sees_nci_h2170": [True, True, False],
        "n_eval_images": [int(loco.n_images) if paper_test is not None
                          else (int(i["n_images"].iloc[0])
                                if "n_images" in i else np.nan),
                          int(ctrl.n_images), int(loco.n_images)],
        "obj_f1": vals, "model": model,
    })
    src["step"] = [np.nan, vals[1] - vals[0], vals[2] - vals[1]]
    src["attributable_to"] = ["", "training-set size", "cell-line novelty"]
    src["in_distribution_scope"] = [indist_scope, "", ""]
    src["in_distribution_all_rows"] = [paper_all, np.nan, np.nan]

    lo = min(vals) - 0.055 if truncate else 0.0
    hi = max(vals) + (0.030 if truncate else 0.11 * max(vals))

    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    y = np.arange(3)[::-1]
    for yi, v, c in zip(y, vals, (BLUE_L, BLUE_M, BLUE_D)):
        ax.barh(yi, v - lo, left=lo, height=0.52, color=c, zorder=2)
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK, fontweight="semibold")

    for yi, a, b, txt, col, bold in (
        (y[0] - 0.5, vals[1], vals[0],
         f"training-set size  {neg(vals[0] - vals[1])}", GREY, False),
        (y[1] - 0.5, vals[2], vals[1],
         f"cell-line novelty  {neg(vals[1] - vals[2])}", BLUE_D, True),
    ):
        ax.annotate("", xy=(a, yi), xytext=(b, yi),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        short = abs(b - a) < 0.28 * (hi - lo)
        ax.annotate(
            txt, (a, yi) if short else ((a + b) / 2, yi),
            xytext=(-6, 0) if short else (0, 5),
            textcoords="offset points",
            ha="right" if short else "center",
            va="center" if short else "bottom",
            fontsize=7.5, color=col,
            fontweight="semibold" if bold else "normal")

    ax.set_yticks(y, labs)
    ax.set_xlabel("Object-level F1 on the 134 NCI-H2170 test images"
                  + (f"  (axis truncated at {lo:.2f})" if truncate else ""))
    ax.set_xlim(lo, hi)
    ax.set_title("Why NCI-H2170 drops: cell-line novelty, not data volume")
    tidy(ax)
    save(fig, outdir, name, src)


def fig_vs_baselines(d: pd.DataFrame, ind: pd.DataFrame, outdir: Path,
                     model: str, name="figL3_vs_baselines") -> None:
    others = sorted(
        set(ind["model"].astype(str).str.strip())
        - {PEAKS, THRESH}
    )
    if not others:
        logger.warning("No baseline models in the per-cell-line CSV; "
                       "skipping figL3.")
        return

    rows, fig_h = [], 0.42 * len(CELL_LINES) + 1.5
    fig, ax = plt.subplots(figsize=(5.8, fig_h))
    y = np.arange(len(CELL_LINES))[::-1]

    for yi, line in zip(y, CELL_LINES):
        sub = ind[ind.cell_line == line]
        for m in others:
            v = sub[sub.model.astype(str).str.strip() == m]["object_f1"]
            if len(v):
                ax.scatter(float(v.iloc[0]), yi, s=34, color=GREY,
                           edgecolor=SURFACE, linewidth=1.0, zorder=2)
                rows.append({"cell_line": line, "model": m,
                             "obj_f1": float(v.iloc[0]),
                             "condition": "in-distribution"})
        r = d[d.cell_line == line]
        if len(r):
            r = r.iloc[0]
            ax.scatter(r.in_distribution_obj_f1, yi, s=52, color=BLUE_L,
                       edgecolor=SURFACE, linewidth=1.1, zorder=3)
            ax.scatter(r.leave_one_out_obj_f1, yi, s=64, color=BLUE_D,
                       marker="D", edgecolor=SURFACE, linewidth=1.1, zorder=4)
            ax.annotate(f"{r.leave_one_out_obj_f1:.3f}",
                        (r.leave_one_out_obj_f1, yi), xytext=(0, -12),
                        textcoords="offset points", ha="center",
                        fontsize=7, color=BLUE_D, fontweight="semibold")
            rows += [
                {"cell_line": line, "model": model,
                 "obj_f1": r.in_distribution_obj_f1,
                 "condition": "in-distribution"},
                {"cell_line": line, "model": model,
                 "obj_f1": r.leave_one_out_obj_f1,
                 "condition": "leave-one-out"},
            ]

    best_other = max(
        (float(ind[(ind.cell_line == l)
                   & (ind.model.astype(str).str.strip() == m)]
               ["object_f1"].iloc[0])
         for l in CELL_LINES for m in others
         if len(ind[(ind.cell_line == l)
                    & (ind.model.astype(str).str.strip() == m)])),
        default=np.nan)
    if np.isfinite(best_other):
        ax.axvline(best_other, color=GREY, lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.annotate("best baseline,\nany cell line", (best_other, y[0] + 0.52),
                    xytext=(-5, 0), textcoords="offset points", ha="right",
                    va="top", fontsize=7, color=GREY)

    ax.set_yticks(y, CELL_LINES)
    ax.set_xlabel("Object-level F1")
    ax.set_ylim(y[-1] - 0.55, y[0] + 0.75)
    ax.set_title("ecCount without the cell line, against models trained on it")
    legend_below(ax, [
        Line2D([], [], marker="D", ls="", ms=6.5, color=BLUE_D,
               label=f"{model}, line unseen"),
        Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE_L,
               label=f"{model}, line seen"),
        Line2D([], [], marker="o", ls="", ms=5.5, color=GREY,
               label="four baselines, line seen"),
    ], ncol=3, dy=-0.26)
    tidy(ax)
    save(fig, outdir, name, pd.DataFrame(rows))


def fig_cost_bars(d: pd.DataFrame, outdir: Path, model: str,
                  name="figL4_cost_bars") -> None:
    d = d.sort_values("cost_of_novelty", ascending=True)
    fig, ax = plt.subplots(figsize=(4.6, 2.2))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.cost_of_novelty, height=0.55, color=BLUE_D, zorder=2)
    for yi, v in zip(y, d.cost_of_novelty):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK)
    ax.set_yticks(y, d.cell_line)
    ax.set_xlabel("Drop in object-level F1 when the cell line is unseen")
    ax.set_xlim(0, d.cost_of_novelty.max() * 1.28)
    ax.set_title(f"Cost of an unseen cell line — {model}")
    tidy(ax)
    save(fig, outdir, name, d)


def fig_bias(d: pd.DataFrame, outdir: Path, model: str,
             name="figL5_signed_bias") -> None:
    d = d.sort_values("mean_signed_bias")
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    y = np.arange(len(d))[::-1]
    cols = [RED if v > 0 else BLUE_D for v in d.mean_signed_bias]
    ax.barh(y, d.mean_signed_bias, height=0.55, color=cols, zorder=2)
    for yi, v in zip(y, d.mean_signed_bias):
        ax.annotate(sgn(v), (v, yi),
                    xytext=(6 if v > 0 else -6, 0),
                    textcoords="offset points", va="center",
                    ha="left" if v > 0 else "right", fontsize=8, color=INK)
    ax.axvline(0, color=AXIS, lw=1.0, zorder=3)
    ax.axvline(LOCKED_BIAS_PEAKS, color=GREY, lw=0.9, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"paper model {sgn(LOCKED_BIAS_PEAKS)}",
                (LOCKED_BIAS_PEAKS, y[0] + 0.52), xytext=(5, 0),
                textcoords="offset points", ha="left", va="top",
                fontsize=7, color=GREY)
    lo, hi = d.mean_signed_bias.min(), d.mean_signed_bias.max()
    pad = 0.22 * (hi - lo)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(y[-1] - 0.5, y[0] + 0.8)
    ax.set_yticks(y, d.cell_line)
    ax.set_xlabel("Mean signed count bias "
                  "(predicted \u2212 annotated, ecDNA per image)")
    ax.set_title(f"How the error is made — {model}, cell line unseen")
    legend_below(ax, [
        Line2D([], [], marker="s", ls="", ms=7, color=BLUE_D,
               label="under-counts"),
        Line2D([], [], marker="s", ls="", ms=7, color=RED,
               label="over-counts"),
    ], ncol=2, dy=-0.26)
    tidy(ax)
    save(fig, outdir, name, d)


def fig_composite(d, tab, ind, outdir: Path, model: str, scope: str,
                  name="figL6_composite",
                  by_scope: Optional[pd.DataFrame] = None) -> None:
    """figL1 + figL2 + figL3 as one display item."""
    fig = plt.figure(figsize=(7.0, 7.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.15], hspace=0.88)

    # (a) dumbbell
    ax = fig.add_subplot(gs[0])
    dd = d.sort_values("cost_of_novelty")
    y = np.arange(len(dd))
    for yi, r in zip(y, dd.itertuples()):
        ax.plot([r.leave_one_out_obj_f1, r.in_distribution_obj_f1], [yi, yi],
                color=AXIS, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter(r.in_distribution_obj_f1, yi, s=46, color=BLUE_L,
                   edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ax.scatter(r.leave_one_out_obj_f1, yi, s=46, color=BLUE_D,
                   edgecolor=SURFACE, linewidth=1.1, zorder=3)
        ax.annotate(f"{r.leave_one_out_obj_f1:.3f}",
                    (r.leave_one_out_obj_f1, yi), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color=INK)
        ax.annotate(f"{r.in_distribution_obj_f1:.3f}",
                    (r.in_distribution_obj_f1, yi), xytext=(6, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7, color=INK2)
    ax.set_yticks(y, dd.cell_line)
    sp = dd.in_distribution_obj_f1.max() - dd.leave_one_out_obj_f1.min()
    ax.set_xlim(dd.leave_one_out_obj_f1.min() - 0.32 * sp,
                dd.in_distribution_obj_f1.max() + 0.22 * sp)
    ax.set_ylim(-0.6, len(dd) - 0.4)
    ax.set_xlabel("Object-level F1")
    ax.set_title("a   Cost of an unseen cell line", loc="left")
    legend_below(ax, [
        Line2D([], [], marker="o", ls="", ms=6, color=BLUE_L,
               label="trained on this line"),
        Line2D([], [], marker="o", ls="", ms=6, color=BLUE_D,
               label="never trained on this line"),
    ], ncol=2, dy=-0.30)
    tidy(ax)

    # (b) decomposition
    ax = fig.add_subplot(gs[1])
    trunc_note = ""

    def get(kind):
        s = tab[(tab.held_out_cell_line == "NCI-H2170") & (tab.model == model)
                & (tab.run_kind == kind)]
        return None if s.empty else s.iloc[0]

    loco = get("leave-one-out (test-restricted)")
    if loco is None:
        loco = get("leave-one-out")
    ctrl = get("size-matched control")
    i = ind[(ind.cell_line == "NCI-H2170")
            & (ind.model.astype(str).str.strip() == model)]
    if loco is not None and ctrl is not None and len(i):
        top = indist_test_rows(by_scope, model, "NCI-H2170")
        if top is None:
            top = float(i["object_f1"].iloc[0])
        vals = [top, float(ctrl.obj_f1), float(loco.obj_f1)]
        base = min(vals) - 0.055
        yy = np.arange(3)[::-1]
        for yi, v, c in zip(yy, vals, (BLUE_L, BLUE_M, BLUE_D)):
            ax.barh(yi, v - base, left=base, height=0.5, color=c, zorder=2)
            ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=INK, fontweight="semibold")
        for yi, a, b, txt, col in (
            (yy[0] - 0.48, vals[1], vals[0],
             f"training-set size  {neg(vals[0] - vals[1])}", GREY),
            (yy[1] - 0.48, vals[2], vals[1],
             f"cell-line novelty  {neg(vals[1] - vals[2])}", BLUE_D),
        ):
            ax.annotate("", xy=(a, yi), xytext=(b, yi),
                        arrowprops=dict(arrowstyle="<->", color=col, lw=1.0,
                                        shrinkA=0, shrinkB=0))
            short = abs(b - a) < 0.28 * (max(vals) + 0.03 - base)
            ax.annotate(
                txt, (a, yi) if short else ((a + b) / 2, yi),
                xytext=(-6, 0) if short else (0, 3),
                textcoords="offset points",
                ha="right" if short else "center",
                va="center" if short else "bottom",
                fontsize=7, color=col)
        ax.set_yticks(yy, ["800 images, sees the line",
                           "179 images, sees the line",
                           "179 images, never sees it"])
        ax.set_xlim(base, max(vals) + 0.030)
        trunc_note = f"  (axis truncated at {base:.2f})"
    ax.set_xlabel("Object-level F1 on the 134 NCI-H2170 test images"
                  + trunc_note)
    ax.set_title("b   NCI-H2170: novelty, not data volume", loc="left")
    tidy(ax)

    # (c) against baselines
    ax = fig.add_subplot(gs[2])
    others = sorted(set(ind["model"].astype(str).str.strip()) - {PEAKS, THRESH})
    yy = np.arange(len(CELL_LINES))[::-1]
    for yi, line in zip(yy, CELL_LINES):
        sub = ind[ind.cell_line == line]
        for m in others:
            v = sub[sub.model.astype(str).str.strip() == m]["object_f1"]
            if len(v):
                ax.scatter(float(v.iloc[0]), yi, s=28, color=GREY,
                           edgecolor=SURFACE, linewidth=0.9, zorder=2)
        r = d[d.cell_line == line]
        if len(r):
            r = r.iloc[0]
            ax.scatter(r.in_distribution_obj_f1, yi, s=46, color=BLUE_L,
                       edgecolor=SURFACE, linewidth=1.0, zorder=3)
            ax.scatter(r.leave_one_out_obj_f1, yi, s=56, color=BLUE_D,
                       marker="D", edgecolor=SURFACE, linewidth=1.0, zorder=4)
    ax.set_yticks(yy, CELL_LINES)
    ax.set_xlabel("Object-level F1")
    ax.set_ylim(yy[-1] - 0.5, yy[0] + 0.55)
    ax.set_title("c   Against models that did train on the line", loc="left")
    legend_below(ax, [
        Line2D([], [], marker="D", ls="", ms=6, color=BLUE_D,
               label="ecCount, line unseen"),
        Line2D([], [], marker="o", ls="", ms=6, color=BLUE_L,
               label="ecCount, line seen"),
        Line2D([], [], marker="o", ls="", ms=5, color=GREY,
               label="baselines, line seen"),
    ], ncol=3, dy=-0.26)
    tidy(ax)

    save(fig, outdir, name, d)


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Candidate figures for the LOCO result.")
    ap.add_argument("--out-root", default="outputs/eccount_loco")
    ap.add_argument("--policy", default="or_matching",
                    choices=["or_matching", "and_matching"])
    ap.add_argument("--model", default=PEAKS, choices=[PEAKS, THRESH])
    ap.add_argument("--in-distribution-csv",
                    default="release/figures/notebook05/"
                            "source_fig6_f1_by_cell_line_heatmap_long.csv")
    ap.add_argument("--in-distribution-by-scope", default=None,
                    help="in_distribution_by_scope.csv from "
                         "rescore_in_distribution_by_scope.py. Supplies the "
                         "in-distribution F1 on held-out test images, which "
                         "makes the NCI-H2170 decomposition like-for-like.")
    ap.add_argument("--diff-annotation", default="none",
                    choices=["none", "full", "displayed"],
                    help="Whether the dumbbell prints the gap between its two "
                         "endpoints, and which version. See fig_dumbbell.")
    ap.add_argument("--figdir", default=None,
                    help="Default: <out-root>/figures")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)
    style()

    root = Path(args.out_root)
    tab_path = root / f"source_loco_paper_table_{args.policy}.csv"
    if not tab_path.is_file():
        logger.error("Not found: %s — run loco_paper_table.py first.", tab_path)
        return 1
    tab = pd.read_csv(tab_path)
    ind = load_indist(Path(args.in_distribution_csv))
    scope = check_scope(ind)

    d = build_frame(tab, ind, args.model, scope)
    if d.empty or d["in_distribution_obj_f1"].isna().all():
        logger.error("No usable rows for %s. Check the model name in both "
                     "CSVs.", args.model)
        return 3
    logger.info("Figures use the '%s' leave-one-out scope to match the "
                "per-cell-line CSV.", scope)

    by_scope = load_by_scope(Path(args.in_distribution_by_scope)
                             if args.in_distribution_by_scope else None)

    outdir = Path(args.figdir) if args.figdir else root / "figures"
    fig_dumbbell(d, outdir, args.model, scope,
                 diff_annotation=args.diff_annotation)
    fig_decomposition(tab, ind, outdir, args.model, truncate=False,
                      by_scope=by_scope)
    fig_decomposition(tab, ind, outdir, args.model, truncate=True,
                      name="figL2z_h2170_decomposition_zoomed",
                      by_scope=by_scope)
    fig_vs_baselines(d, ind, outdir, args.model)
    fig_cost_bars(d, outdir, args.model)
    fig_bias(d, outdir, args.model)
    fig_composite(d, tab, ind, outdir, args.model, scope, by_scope=by_scope)

    print()
    print("=" * 74)
    print(f"  Figures written to {outdir}")
    print("=" * 74)
    print(d.to_string(index=False))
    print()
    print("Each figure has a matching source_<name>.csv. The NCI-H2170 bar in")
    print("figL2 uses the test-restricted leave-one-out row, so all three")
    print("conditions there cover the same 134 images.")

    print()
    print("=" * 74)
    print("  DISPLAY RECONCILIATION — three decimals, per the house style")
    print("=" * 74)
    print(f"  {'cell line':<12}{'in-dist':>9}{'unseen':>9}{'true gap':>11}"
          f"{'endpoints imply':>17}")
    print("  " + "-" * 70)
    clash = []
    for r in d.itertuples():
        a, b = r.in_distribution_obj_f1, r.leave_one_out_obj_f1
        shown = round(a, 3) - round(b, 3)
        bad = round(r.cost_of_novelty, 3) != round(shown, 3)
        if bad:
            clash.append(r.cell_line)
        print(f"  {r.cell_line:<12}{a:>9.3f}{b:>9.3f}"
              f"{r.cost_of_novelty:>11.3f}{shown:>17.3f}"
              + ("   <-- differ" if bad else ""))
    print()
    if clash:
        print(f"  For {', '.join(clash)} the printed endpoints do not "
              f"reproduce the gap.")
        print("  Both numbers are right; three decimals cannot show both. The")
        print("  dumbbell therefore prints endpoints only unless you pass")
        print("  --diff-annotation. Quote the true gap (the cost_of_novelty")
        print("  column) in the text, and let figL4 carry it in a panel.")
    else:
        print("  Every panel reconciles with its own printed endpoints.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
