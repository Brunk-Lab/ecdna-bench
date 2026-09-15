#!/usr/bin/env python
"""
loco_figures_partB.py — Fig. 6e, 6f, 6g and ED Fig. 8e, all on one scope.

    python scripts/loco_figures_partB.py \
        --by-scope outputs/eccount_loco/f1_by_scope.csv \
        --outdir   outputs/eccount_loco/figures_partB

WHAT THIS REPLACES, AND WHY IT IS A NEW SCRIPT
----------------------------------------------
`loco_figures.py` pairs an in-distribution value read from the all-rows
per-cell-line heatmap CSV with a leave-one-out value read from the LOCO paper
table. Those two files are scored on different image sets, so every difference
it computes is part scope change. That is the whole of the column A / B / C
confusion, and it lives in one function (`build_frame`), which never consults
`--in-distribution-by-scope` even when that file is supplied.

This script takes a single input: `f1_by_scope.csv`, written by
`rescore_in_distribution_by_scope.py --loco-run-root`. That file already pools
BOTH sides at BOTH scopes from per-image TP/FP/FN, so:

    cost of novelty = in-distribution(test rows) - leave-one-out(test rows)

is a lookup, not a computation across files. There is no second file to
disagree with, which is the point.

`loco_figures.py` and `loco_story_figures.py` are left alone. They still make
the exploratory figL*/figD* panels. Nothing here overwrites a locked artefact:
everything is written under --outdir, which defaults to a new directory.

THE FOUR PANELS
---------------
    fig6f_cost_of_novelty        B2. Bars, one per cell line.
    fig6g_decomposition          B1. Three conditions on the 134 NCI-H2170
                                 test images, with both step labels.
    edfig8e_cost_of_novelty      B3. The same four values as fig6f, as paired
                                 endpoints. Asserted equal to fig6f.
    fig6e_precision_recall_f1    B4. Precision, recall and F1 on one scope for
                                 all four cell lines, from pooled TP/FP/FN.

DISPLAY RECONCILIATION
----------------------
Three decimals cannot always show two values and their difference: 0.9435 and
0.9290 print as 0.944 and 0.929, whose printed difference is 0.015, while the
true difference is 0.0145 and prints as 0.014. That is exactly how Fig. 6g came
to carry a bar of 0.944 next to a step of -0.014.

`--label-from displayed` (the default) computes every printed step from the
values as printed, so a reader who subtracts what is on the page always gets
what is on the page. `--label-from full` prints the true difference instead and
warns wherever the two disagree. The source CSV always carries both.

EXIT CODES
----------
    0  every panel written and every reconciliation check passed
    1  an input was missing or unusable
    2  panels were written but a reconciliation check FAILED - do not use the
       artwork until the cause is understood
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger("partB")

# --- palette, identical to loco_figures.py so the panels match -------------
BLUE_L = "#86b6ef"
BLUE_M = "#3987e5"
BLUE_D = "#184f95"
BLUE_700 = "#0d366b"
RED = "#e34948"
GREY = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
INK = "#0b0b0b"
INK2 = "#52514e"
SURFACE = "#fcfcfb"

CELL_LINES = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]
PEAKS = "ecCount (peaks)"
THRESH = "ecCount (threshold mask)"

# vocabulary written by rescore_in_distribution_by_scope.py
ALL, TEST = "all rows", "test rows"
IN_DIST, LOCO, CONTROL = ("in-distribution", "leave-one-out",
                          "size-matched control")

# Locked expectations. Only used to tell the operator whether what came out
# matches what the manuscript says; never substituted for a computed value.
EXPECTED_COST = {"COLO320DM": 0.026, "NCI-H2170": 0.108,
                 "NCI-H716": 0.004, "SNU16": 0.024}
EXPECTED_DECOMP = {"full": 0.941, "control": 0.929, "loco": 0.833}
EXPECTED_N = {"COLO320DM": 11, "NCI-H2170": 134, "NCI-H716": 11, "SNU16": 19}


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


def neg(v: float, dp: int = 3) -> str:
    """Typographic minus, so labels match the axis ticks."""
    return f"−{abs(v):.{dp}f}"


def tidy(ax, xgrid=True, ygrid=False) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(xgrid)
    ax.yaxis.grid(ygrid)


def legend_below(ax, handles, ncol=2, dy=-0.26) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, dy),
              ncol=ncol, frameon=False, handletextpad=0.4,
              columnspacing=1.4, borderpad=0)


def save(fig, outdir: Path, name: str, src: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 400}), ("svg", {})):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", **kw)
    src.to_csv(outdir / f"source_{name}.csv", index=False)
    plt.close(fig)
    logger.info("wrote %s.{png,svg} + source_%s.csv", name, name)


# ---------------------------------------------------------------------------
# input
# ---------------------------------------------------------------------------

REQUIRED = {"model", "cell_line", "condition", "scope", "n_images",
            "object_f1"}


def load_by_scope(path: Path) -> pd.DataFrame:
    if not path.is_file():
        logger.error("Not found: %s", path)
        logger.error("Produce it first with:")
        logger.error("  python scripts/rescore_in_distribution_by_scope.py \\")
        logger.error("      --per-image     release/frozen_results/or_matching/"
                     "per_image_metrics.csv \\")
        logger.error("      --metadata      release/manifests/"
                     "dl_master_metadata_stage1_step3_consistency.csv \\")
        logger.error("      --loco-run-root outputs/eccount_loco \\")
        logger.error("      --out           outputs/eccount_loco/"
                     "f1_by_scope.csv")
        logger.error("  The --loco-run-root argument is REQUIRED here: without "
                     "it the file")
        logger.error("  carries only the in-distribution side and no cost can "
                     "be computed.")
        sys.exit(1)

    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        logger.error("%s lacks %s (has %s)", path, sorted(missing),
                     list(df.columns))
        sys.exit(1)

    for c in ("model", "cell_line", "condition", "scope"):
        df[c] = df[c].astype(str).str.strip()

    conds = set(df.condition.unique())
    if LOCO not in conds:
        logger.error("%s has no '%s' rows, so it was written WITHOUT "
                     "--loco-run-root.", path, LOCO)
        logger.error("Re-run rescore_in_distribution_by_scope.py with "
                     "--loco-run-root outputs/eccount_loco.")
        sys.exit(1)
    if TEST not in set(df.scope.unique()):
        logger.error("%s has no '%s' scope rows. Its per-image inputs had no "
                     "split column.", path, TEST)
        sys.exit(1)
    return df


def has_pooled_counts(df: pd.DataFrame) -> bool:
    return {"tp", "fp", "fn"}.issubset(df.columns)


def lookup(df: pd.DataFrame, model: str, line: str, condition: str,
           scope: str) -> Optional[pd.Series]:
    r = df[(df.model == model) & (df.cell_line == line)
           & (df.condition == condition) & (df.scope == scope)]
    if len(r) > 1:
        logger.warning("%s / %s / %s / %s matched %d rows; using the first.",
                       model, line, condition, scope, len(r))
    return r.iloc[0] if len(r) else None


def build(df: pd.DataFrame, model: str, scope: str) -> pd.DataFrame:
    """One row per cell line: both scores, the cost, and pooled counts."""
    rows = []
    for line in CELL_LINES:
        a = lookup(df, model, line, IN_DIST, scope)
        b = lookup(df, model, line, LOCO, scope)
        if a is None or b is None:
            logger.warning("%s: missing %s at scope '%s'; skipped.", line,
                           "in-distribution" if a is None else "leave-one-out",
                           scope)
            continue
        rec = {
            "cell_line": line,
            "model": model,
            "scope": scope,
            "n_images_in_distribution": int(a.n_images),
            "n_images_unseen": int(b.n_images),
            "in_distribution_obj_f1": float(a.object_f1),
            "unseen_obj_f1": float(b.object_f1),
            "cost_of_novelty": float(a.object_f1) - float(b.object_f1),
        }
        rec["cost_of_novelty_displayed"] = (round(rec["in_distribution_obj_f1"], 3)
                                            - round(rec["unseen_obj_f1"], 3))
        if has_pooled_counts(df):
            for side, r in (("in_distribution", a), ("unseen", b)):
                tp, fp, fn = float(r.tp), float(r.fp), float(r.fn)
                rec[f"{side}_tp"] = int(tp)
                rec[f"{side}_fp"] = int(fp)
                rec[f"{side}_fn"] = int(fn)
                rec[f"{side}_precision"] = tp / (tp + fp) if tp + fp else np.nan
                rec[f"{side}_recall"] = tp / (tp + fn) if tp + fn else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# B2 — Fig. 6f
# ---------------------------------------------------------------------------

def fig_6f(d: pd.DataFrame, outdir: Path, model: str, label_from: str,
           name="fig6f_cost_of_novelty") -> pd.DataFrame:
    d = d.sort_values("cost_of_novelty").reset_index(drop=True)
    col = ("cost_of_novelty_displayed" if label_from == "displayed"
           else "cost_of_novelty")

    fig, ax = plt.subplots(figsize=(4.8, 2.35))
    y = np.arange(len(d))[::-1]
    ax.barh(y, d[col], height=0.55, color=BLUE_D, zorder=2)
    for yi, v, n in zip(y, d[col], d.n_images_unseen):
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK, fontweight="semibold")
    ax.set_yticks(y, [f"{l}\n(n = {n})" for l, n
                      in zip(d.cell_line, d.n_images_unseen)])
    ax.set_xlabel("Drop in object-level F1 when the cell line is unseen")
    ax.set_xlim(0, max(float(d[col].max()) * 1.30, 0.02))
    ax.set_title(f"Cost of cellular-context novelty — {model}")
    tidy(ax)
    save(fig, outdir, name, d)
    return d


# ---------------------------------------------------------------------------
# B3 — ED Fig. 8e
# ---------------------------------------------------------------------------

def fig_ed8e(d: pd.DataFrame, outdir: Path, model: str, label_from: str,
             name="edfig8e_cost_of_novelty") -> pd.DataFrame:
    """Paired endpoints. Plots the same four costs as fig6f, by construction:
    both read one dataframe, and main() asserts the printed values match."""
    d = d.sort_values("cost_of_novelty").reset_index(drop=True)
    col = ("cost_of_novelty_displayed" if label_from == "displayed"
           else "cost_of_novelty")

    fig, ax = plt.subplots(figsize=(5.4, 2.6))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        ax.plot([r.unseen_obj_f1, r.in_distribution_obj_f1], [yi, yi],
                color=AXIS, lw=1.6, solid_capstyle="round", zorder=1)
        ax.scatter(r.in_distribution_obj_f1, yi, s=50, color=BLUE_L,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.scatter(r.unseen_obj_f1, yi, s=50, color=BLUE_D,
                   edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.annotate(f"{r.unseen_obj_f1:.3f}", (r.unseen_obj_f1, yi),
                    xytext=(-7, 0), textcoords="offset points", ha="right",
                    va="center", fontsize=7.5, color=INK)
        ax.annotate(f"{r.in_distribution_obj_f1:.3f}",
                    (r.in_distribution_obj_f1, yi), xytext=(7, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=7.5, color=INK2)
        gap = getattr(r, col)
        ax.annotate(neg(gap), ((r.unseen_obj_f1 + r.in_distribution_obj_f1) / 2,
                               yi),
                    xytext=(0, 7), textcoords="offset points", ha="center",
                    va="bottom", fontsize=7, color=INK2)

    ax.set_yticks(y, [f"{l}\n(n = {n})" for l, n
                      in zip(d.cell_line, d.n_images_unseen)])
    ax.set_xlabel("Object-level F1")
    span = float(d.in_distribution_obj_f1.max() - d.unseen_obj_f1.min()) or 0.05
    ax.set_xlim(float(d.unseen_obj_f1.min()) - 0.40 * span,
                float(d.in_distribution_obj_f1.max()) + 0.28 * span)
    ax.set_ylim(-0.6, len(d) - 0.3)
    ax.set_title(f"Cost of cellular-context novelty — {model}")
    legend_below(ax, [
        Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE_L,
               label="released model (trained on this line)"),
        Line2D([], [], marker="o", ls="", ms=6.5, color=BLUE_D,
               label="model that never saw this line"),
    ], ncol=2, dy=-0.30)
    tidy(ax)
    save(fig, outdir, name, d)
    return d


# ---------------------------------------------------------------------------
# B1 — Fig. 6g
# ---------------------------------------------------------------------------

def fig_6g(df: pd.DataFrame, outdir: Path, model: str, label_from: str,
           truncate: bool = True,
           name="fig6g_decomposition") -> Optional[pd.DataFrame]:
    """Three conditions, one image set: the 134 NCI-H2170 held-out test images.

    Both step labels are derived from the same three values that are printed on
    the bars, so the panel cannot contradict itself. That is the defect the
    published version carries: a bar of 0.944 beside a step of -0.014.
    """
    line = "NCI-H2170"
    full = lookup(df, model, line, IN_DIST, TEST)
    ctrl = lookup(df, model, line, CONTROL, TEST)
    loco = lookup(df, model, line, LOCO, TEST)

    absent = [n for n, v in (("in-distribution", full),
                             ("size-matched control", ctrl),
                             ("leave-one-out", loco)) if v is None]
    if absent:
        logger.warning("fig6g needs all three NCI-H2170 conditions at scope "
                       "'%s'; missing %s. Skipped.", TEST, ", ".join(absent))
        return None

    vals = [float(full.object_f1), float(ctrl.object_f1), float(loco.object_f1)]
    ns = [int(full.n_images), int(ctrl.n_images), int(loco.n_images)]
    if len(set(ns)) != 1:
        logger.warning("fig6g: the three conditions cover %s images, not one "
                       "common set. The panel would compare across image sets.",
                       ns)

    if label_from == "displayed":
        steps = [round(vals[0], 3) - round(vals[1], 3),
                 round(vals[1], 3) - round(vals[2], 3)]
    else:
        steps = [vals[0] - vals[1], vals[1] - vals[2]]

    src = pd.DataFrame({
        "condition": ["full training set", "size-matched control",
                      "leave-one-out"],
        "n_train_images": [800, 179, 179],
        "sees_nci_h2170": [True, True, False],
        "n_eval_images": ns,
        "scope": [TEST] * 3,
        "obj_f1": vals,
        "model": model,
        "step_full_precision": [np.nan, vals[0] - vals[1], vals[1] - vals[2]],
        "step_as_printed": [np.nan, steps[0], steps[1]],
        "attributable_to": ["", "training-set size", "cellular novelty"],
    })

    lo = min(vals) - 0.055 if truncate else 0.0
    hi = max(vals) + (0.030 if truncate else 0.11 * max(vals))

    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    y = np.arange(3)[::-1]
    labs = [f"Full training set\n(800 images, sees {line})",
            f"Size-matched control\n(179 images, sees {line})",
            f"Leave-one-out\n(179 images, never sees {line})"]
    for yi, v, c in zip(y, vals, (BLUE_L, BLUE_M, BLUE_D)):
        ax.barh(yi, v - lo, left=lo, height=0.52, color=c, zorder=2)
        ax.annotate(f"{v:.3f}", (v, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8,
                    color=INK, fontweight="semibold")

    for yi, a, b, step, txt, col, bold in (
        (y[0] - 0.5, vals[1], vals[0], steps[0], "training-set size", GREY,
         False),
        (y[1] - 0.5, vals[2], vals[1], steps[1], "cellular novelty", BLUE_D,
         True),
    ):
        ax.annotate("", xy=(a, yi), xytext=(b, yi),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.1,
                                    shrinkA=0, shrinkB=0))
        short = abs(b - a) < 0.28 * (hi - lo)
        ax.annotate(
            f"{txt}  {neg(step)}",
            (a, yi) if short else ((a + b) / 2, yi),
            xytext=(-6, 0) if short else (0, 5),
            textcoords="offset points",
            ha="right" if short else "center",
            va="center" if short else "bottom",
            fontsize=7.5, color=col,
            fontweight="semibold" if bold else "normal")

    ax.set_yticks(y, labs)
    ax.set_xlabel(f"Object-level F1 on the {ns[0]} {line} held-out test images"
                  + (f"  (axis truncated at {lo:.2f})" if truncate else ""))
    ax.set_xlim(lo, hi)
    ax.set_title("Cellular novelty, not training-set size")
    tidy(ax)
    save(fig, outdir, name, src)
    return src


# ---------------------------------------------------------------------------
# B4 — Fig. 6e
# ---------------------------------------------------------------------------

def fig_6e(d: pd.DataFrame, outdir: Path, model: str,
           name="fig6e_precision_recall_f1") -> Optional[pd.DataFrame]:
    """Precision, recall and F1 for the unseen-line model, one scope for all
    four cell lines. Precision and recall come from the pooled TP/FP/FN in
    f1_by_scope.csv, so nothing needs re-running."""
    need = {"unseen_precision", "unseen_recall"}
    if not need.issubset(d.columns) or d["unseen_precision"].isna().all():
        logger.warning("fig6e needs pooled tp/fp/fn in f1_by_scope.csv to "
                       "derive precision and recall; skipped.")
        return None

    d = d.sort_values("unseen_obj_f1", ascending=False).reset_index(drop=True)
    src = d[["cell_line", "model", "scope", "n_images_unseen",
             "unseen_precision", "unseen_recall", "unseen_obj_f1",
             "unseen_tp", "unseen_fp", "unseen_fn"]].rename(columns={
                 "n_images_unseen": "n_images", "unseen_precision": "precision",
                 "unseen_recall": "recall", "unseen_obj_f1": "obj_f1",
                 "unseen_tp": "tp", "unseen_fp": "fp", "unseen_fn": "fn"})

    metrics = [("precision", BLUE_L), ("recall", BLUE_M), ("obj_f1", BLUE_D)]
    x = np.arange(len(src))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for i, (m, col) in enumerate(metrics):
        off = (i - 1) * w
        ax.bar(x + off, src[m], width=w * 0.92, color=col, zorder=2)
        for xi, v in zip(x + off, src[m]):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.6, color=INK, rotation=90)
    ax.set_xticks(x, [f"{l}\n(n = {n})" for l, n
                      in zip(src.cell_line, src.n_images)])
    ax.set_ylabel("Score on the held-out cell line")
    ax.set_ylim(0, 1.15)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"Detection on a cell line the model never saw — {model}")
    tidy(ax, xgrid=False, ygrid=True)
    legend_below(ax, [Patch(facecolor=c, label=m.replace("obj_f1", "F1"))
                      for m, c in metrics], ncol=3, dy=-0.30)
    save(fig, outdir, name, src)
    return src


# ---------------------------------------------------------------------------
# reconciliation
# ---------------------------------------------------------------------------

def reconcile(d6f: pd.DataFrame, d8e: pd.DataFrame,
              g: Optional[pd.DataFrame], label_from: str) -> List[str]:
    fail: List[str] = []
    w = 78
    col = ("cost_of_novelty_displayed" if label_from == "displayed"
           else "cost_of_novelty")

    print()
    print("=" * w)
    print("  RECONCILIATION")
    print("=" * w)

    # 1. the two panels must print the same four numbers
    a = d6f.set_index("cell_line")[col].round(3)
    b = d8e.set_index("cell_line")[col].round(3)
    same = a.reindex(sorted(a.index)).equals(b.reindex(sorted(b.index)))
    print(f"  Fig. 6f and ED Fig. 8e print the same four values ....... "
          f"{'PASS' if same else 'FAIL'}")
    if not same:
        fail.append("Fig. 6f and ED Fig. 8e disagree.")

    # 2. every panel must reproduce its own printed endpoints
    print()
    print(f"  {'cell line':<12}{'in-dist':>9}{'unseen':>9}{'printed':>10}"
          f"{'true gap':>11}{'endpoints':>12}")
    print("  " + "-" * (w - 4))
    for r in d8e.itertuples():
        implied = round(r.in_distribution_obj_f1, 3) - round(r.unseen_obj_f1, 3)
        printed = getattr(r, col)
        ok = round(printed, 3) == round(implied, 3)
        print(f"  {r.cell_line:<12}{r.in_distribution_obj_f1:>9.3f}"
              f"{r.unseen_obj_f1:>9.3f}{printed:>10.3f}"
              f"{r.cost_of_novelty:>11.4f}{implied:>12.3f}"
              + ("" if ok else "   <-- differs"))
        if not ok:
            fail.append(f"{r.cell_line}: ED Fig. 8e prints endpoints that do "
                        f"not reproduce its own gap label.")

    # 3. the decomposition must close against Fig. 6f's NCI-H2170 bar
    if g is not None:
        s1 = float(g.step_as_printed.iloc[1])
        s2 = float(g.step_as_printed.iloc[2])
        h = d6f[d6f.cell_line == "NCI-H2170"]
        if len(h):
            bar = float(h[col].iloc[0])
            closes = round(s1 + s2, 3) == round(bar, 3)
            print()
            print(f"  Fig. 6g:  {s1:.3f} (size) + {s2:.3f} (novelty) "
                  f"= {s1 + s2:.3f}")
            print(f"  Fig. 6f:  NCI-H2170 bar = {bar:.3f}")
            print(f"  The decomposition closes ............................... "
                  f"{'PASS' if closes else 'FAIL'}")
            if not closes:
                fail.append("Fig. 6g's two steps do not sum to Fig. 6f's "
                            "NCI-H2170 bar.")

    # 4. against the manuscript
    print()
    print("  Against the locked values in the manuscript:")
    for r in d6f.itertuples():
        exp = EXPECTED_COST.get(r.cell_line)
        got = round(getattr(r, col), 3)
        mark = "ok" if exp is not None and abs(got - exp) < 5e-4 else "DIFFERS"
        print(f"    {r.cell_line:<12} panel {got:.3f}   manuscript "
              f"{exp if exp is not None else float('nan'):.3f}   {mark}")
        if mark == "DIFFERS":
            fail.append(f"{r.cell_line}: {got:.3f} is not the "
                        f"{exp:.3f} the manuscript expects.")
    for r in d6f.itertuples():
        exp_n = EXPECTED_N.get(r.cell_line)
        if exp_n is not None and int(r.n_images_unseen) != exp_n:
            print(f"    {r.cell_line:<12} n = {int(r.n_images_unseen)}, "
                  f"expected {exp_n}")
            fail.append(f"{r.cell_line}: n = {int(r.n_images_unseen)}, not "
                        f"{exp_n}. Check the scope.")
    if g is not None:
        for key, v in zip(("full", "control", "loco"), g.obj_f1):
            exp = EXPECTED_DECOMP[key]
            if abs(round(float(v), 3) - exp) >= 5e-4:
                print(f"    Fig. 6g {key:<9} {float(v):.3f}, expected "
                      f"{exp:.3f}")
                fail.append(f"Fig. 6g {key} bar is {float(v):.3f}, not "
                            f"{exp:.3f}.")
    return fail


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__)
    ap.add_argument("--by-scope", type=Path,
                    default=Path("outputs/eccount_loco/f1_by_scope.csv"),
                    help="f1_by_scope.csv from rescore_in_distribution_by_"
                         "scope.py, written WITH --loco-run-root.")
    ap.add_argument("--model", default=PEAKS, choices=[PEAKS, THRESH])
    ap.add_argument("--scope", default=TEST, choices=[TEST, ALL],
                    help="Which image set both sides are scored on. "
                         "'test rows' is the like-for-like scope the "
                         "manuscript reports.")
    ap.add_argument("--label-from", default="displayed",
                    choices=["displayed", "full"],
                    help="Whether printed differences are computed from the "
                         "printed endpoints or at full precision.")
    ap.add_argument("--no-truncate", action="store_true",
                    help="Fig. 6g from a zero baseline instead of a truncated "
                         "axis.")
    ap.add_argument("--outdir", type=Path,
                    default=Path("outputs/eccount_loco/figures_partB"))
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)
    style()

    df = load_by_scope(args.by_scope)
    if not has_pooled_counts(df):
        logger.warning("%s has no tp/fp/fn columns — Fig. 6e will be skipped. "
                       "Re-run rescore_in_distribution_by_scope.py, which "
                       "writes them.", args.by_scope)

    d = build(df, args.model, args.scope)
    if d.empty:
        logger.error("No cell line had both an in-distribution and a "
                     "leave-one-out row at scope '%s' for %s.",
                     args.scope, args.model)
        return 1
    if len(d) < len(CELL_LINES):
        logger.warning("Only %d of %d cell lines are present.", len(d),
                       len(CELL_LINES))

    logger.info("Scope: '%s'. Both sides of every difference are scored on "
                "this image set.", args.scope)

    d6f = fig_6f(d, args.outdir, args.model, args.label_from)
    d8e = fig_ed8e(d, args.outdir, args.model, args.label_from)
    g = fig_6g(df, args.outdir, args.model, args.label_from,
               truncate=not args.no_truncate)
    fig_6e(d, args.outdir, args.model)

    fail = reconcile(d6f, d8e, g, args.label_from)

    print()
    print("=" * 78)
    if fail:
        print("  CHECKS FAILED - do not drop this artwork into the figure")
        print("=" * 78)
        for f in fail:
            print("  - " + f)
        print()
        print("  The panels were still written so you can look at them, but "
              "something")
        print("  upstream disagrees with the manuscript. Check the scope of "
              "f1_by_scope.csv")
        print("  before using them.")
        print()
        return 2
    print("  ALL CHECKS PASSED")
    print("=" * 78)
    print(f"  Panels in {args.outdir}")
    print("  Fig. 6f and ED Fig. 8e print identical values; Fig. 6g's two "
          "steps sum to")
    print("  Fig. 6f's NCI-H2170 bar; every panel reproduces its own printed "
          "endpoints.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
