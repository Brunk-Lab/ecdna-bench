#!/usr/bin/env python
"""
loco_story_figures.py — the leave-one-cell-line-out experiment, told in order.

    python scripts/loco_story_figures.py --out-root outputs/eccount_loco

loco_figures.py answers "what was the result". This script answers the three
questions a reader has *before* the result, and then reports the result in a
form where every number is readable rather than inferred from a mark's
position.

    figD1  what the benchmark is made of      cell line x split, images and
                                              annotated objects, as heatmaps
    figD2  what each run held out             the design matrix — five runs x
                                              four cell lines, colour = the
                                              role that line played
    figD3  did every run train properly       train and validation loss for
                                              all five runs, best epoch marked
    figD4  precision, recall and F1           grouped bars on the held-out line
    figD5  every model, every cell line       annotated heatmap; replaces the
                                              dot panel where the baselines
                                              could not be told apart

WHY figD2 EXISTS
----------------
"Leave-one-cell-line-out" is one sentence that describes five different
training sets, and no reader holds five training sets in their head. The
design matrix puts them on one grid: one row per run, one column per cell
line, and a colour for whether that line was trained on, held out and scored,
or sampled down for the size-matched control. Everything else in the story
depends on the reader having understood that grid.

INPUTS
------
    <run>/split_composition.csv    role, cell_line, n_images, gt_objects
                                   (written by train_eccount_loco.py)
    <run>/train_history.csv        per-epoch losses (written by the trainer;
                                   column names are detected, not assumed)
    source_loco_paper_table_*.csv  written by loco_paper_table.py
    the locked per-cell-line CSV   in-distribution values for every model

Any input that is missing causes its own figure to be skipped with a warning.
The others still render.
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

logger = logging.getLogger("loco_story")

# --- palette (same validated instance as loco_figures.py) ------------------
BLUE_100, BLUE_250, BLUE_400 = "#cde2fb", "#86b6ef", "#3987e5"
BLUE_500, BLUE_600, BLUE_700 = "#256abf", "#184f95", "#0d366b"
ORANGE = "#eb6834"
RED = "#e34948"
GREY = "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
INK, INK2 = "#0b0b0b", "#52514e"
SURFACE = "#fcfcfb"

BLUES = LinearSegmentedColormap.from_list(
    "loco_blues", ["#f2f7fe", BLUE_100, BLUE_250, BLUE_400, BLUE_600, BLUE_700])

CELL_LINES = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]
SLUG = {c: c.lower().replace("-", "_") for c in CELL_LINES}
PEAKS = "ecCount (peaks)"
THRESH = "ecCount (threshold mask)"
SPLITS = ["train", "val", "test"]

RUNS: List[Tuple[str, str, str]] = [
    # (run_id, held-out line, label)
    (f"holdout_{SLUG['COLO320DM']}", "COLO320DM", "hold out COLO320DM"),
    (f"holdout_{SLUG['NCI-H2170']}", "NCI-H2170", "hold out NCI-H2170"),
    (f"holdout_{SLUG['NCI-H716']}", "NCI-H716", "hold out NCI-H716"),
    (f"holdout_{SLUG['SNU16']}", "SNU16", "hold out SNU16"),
    (f"holdout_{SLUG['NCI-H2170']}_control", "NCI-H2170",
     "size-matched control\n(NCI-H2170)"),
]


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 8, "axes.labelsize": 8,
        "axes.titlesize": 9, "axes.titleweight": "semibold",
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2,
        "xtick.color": INK2, "ytick.color": INK2,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "grid.color": GRID, "grid.linewidth": 0.7,
        "legend.frameon": False, "legend.fontsize": 7.5,
        "svg.fonttype": "none",
    })


def tidy(ax, xgrid=True, ygrid=False) -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_axisbelow(True)
    ax.xaxis.grid(xgrid)
    ax.yaxis.grid(ygrid)


def legend_below(ax, handles, ncol=2, dy=-0.26) -> None:
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, dy),
              ncol=ncol, frameon=False, handletextpad=0.5,
              columnspacing=1.4, borderpad=0)


def save(fig, outdir: Path, name: str, src: pd.DataFrame) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("png", {"dpi": 300}), ("svg", {})):
        fig.savefig(outdir / f"{name}.{ext}", bbox_inches="tight", **kw)
    src.to_csv(outdir / f"source_{name}.csv", index=False)
    plt.close(fig)
    logger.info("wrote %s.{png,svg} + source_%s.csv", name, name)


def heat_text_colour(v: float, vmax: float) -> str:
    """Ink on light cells, surface on dark ones."""
    return SURFACE if (vmax and v / vmax > 0.55) else INK


def annotated_heatmap(ax, mat: np.ndarray, rows: List[str], cols: List[str],
                      fmt="{:,.0f}", cmap=BLUES, vmax: Optional[float] = None,
                      vmin: float = 0.0, norm_rows=False):
    """Sequential heatmap with every value printed in the cell.

    ``norm_rows`` colours each row by its own maximum, for matrices whose rows
    differ by orders of magnitude (image counts against object counts).
    """
    disp = mat.astype(float).copy()
    if norm_rows:
        with np.errstate(invalid="ignore"):
            disp = disp / np.nanmax(disp, axis=1, keepdims=True)
        vmax_use = 1.0
    else:
        vmax_use = vmax if vmax is not None else np.nanmax(disp)
    im = ax.imshow(disp, cmap=cmap, vmin=vmin, vmax=vmax_use, aspect="auto")
    span = (vmax_use - vmin) or 1.0
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            v = mat[r, c]
            if not np.isfinite(v):
                continue
            shade = (disp[r, c] - vmin) / span
            ax.text(c, r, fmt.format(v), ha="center", va="center",
                    fontsize=7.2,
                    color=SURFACE if shade > 0.55 else INK)
    ax.set_xticks(range(len(cols)), cols)
    ax.set_yticks(range(len(rows)), rows)
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(rows) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.6)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def read_compositions(root: Path) -> Dict[str, pd.DataFrame]:
    out = {}
    for run_id, _, _ in RUNS:
        p = root / run_id / "split_composition.csv"
        if p.is_file():
            out[run_id] = pd.read_csv(p)
        else:
            logger.warning("missing %s", p)
    return out


def detect_history_cols(df: pd.DataFrame) -> Optional[Dict[str, str]]:
    """Find the epoch / train-loss / val-loss / lr columns by name pattern."""
    def find(*pats, exclude=()):
        for c in df.columns:
            lc = str(c).lower()
            if any(re.search(p, lc) for p in pats) and \
               not any(re.search(e, lc) for e in exclude):
                return c
        return None

    cols = {
        "epoch": find(r"^epoch$", r"^ep$", r"epoch"),
        "train": find(r"train.*loss", r"^train$", r"loss.*train"),
        "val": find(r"val.*loss", r"^val$", r"valid", r"loss.*val"),
        "lr": find(r"^lr$", r"learning_?rate"),
    }
    if cols["train"] is None or cols["val"] is None:
        logger.warning("train_history.csv columns not recognised: %s",
                       list(df.columns))
        return None
    return cols


def read_histories(root: Path) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
    out = {}
    for run_id, _, _ in RUNS:
        p = root / run_id / "train_history.csv"
        if not p.is_file():
            logger.warning("missing %s", p)
            continue
        df = pd.read_csv(p)
        cols = detect_history_cols(df)
        if cols:
            out[run_id] = (df, cols)
    return out


# ---------------------------------------------------------------------------
# figD1 — what the benchmark is made of
# ---------------------------------------------------------------------------

def fig_composition(comps: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """Reconstruct the full benchmark from the runs' own split tables.

    Every image appears exactly once as 'eval' in the run that held its cell
    line out, carrying its original split label, so the four leave-one-out
    runs together reproduce the whole benchmark. Nothing is typed in.
    """
    rows = []
    for run_id, line, _ in RUNS:
        if run_id.endswith("_control") or run_id not in comps:
            continue
        ev = comps[run_id]
        ev = ev[(ev.role == "eval") & (ev.cell_line == line)]
        for _, r in ev.iterrows():
            rows.append({"cell_line": line, "n_images": int(r.n_images),
                         "gt_objects": int(r.gt_objects)})
    if not rows:
        logger.warning("figD1 needs split_composition.csv; skipping.")
        return
    tot = pd.DataFrame(rows).groupby("cell_line", as_index=False).sum()

    # split-level counts come from the training sets of the other runs
    per_split = {l: {s: np.nan for s in SPLITS} for l in CELL_LINES}
    for run_id, held, _ in RUNS:
        if run_id.endswith("_control") or run_id not in comps:
            continue
        c = comps[run_id]
        for role, split in (("train", "train"), ("val", "val")):
            for _, r in c[c.role == role].iterrows():
                per_split[r.cell_line][split] = int(r.n_images)
    for l in CELL_LINES:
        known = [per_split[l][s] for s in ("train", "val")]
        t = tot[tot.cell_line == l]
        if len(t) and all(np.isfinite(known)):
            per_split[l]["test"] = int(t.n_images.iloc[0]) - sum(
                int(k) for k in known)

    img = np.array([[per_split[l][s] for s in SPLITS] +
                    [tot[tot.cell_line == l].n_images.iloc[0]
                     if len(tot[tot.cell_line == l]) else np.nan]
                    for l in CELL_LINES], dtype=float)
    obj = np.array([[tot[tot.cell_line == l].gt_objects.iloc[0]
                     if len(tot[tot.cell_line == l]) else np.nan]
                    for l in CELL_LINES], dtype=float)

    src = pd.DataFrame({
        "cell_line": CELL_LINES,
        "train_images": img[:, 0], "val_images": img[:, 1],
        "test_images": img[:, 2], "total_images": img[:, 3],
        "annotated_objects": obj[:, 0],
    })
    src["objects_per_image"] = src.annotated_objects / src.total_images

    fig, axes = plt.subplots(
        1, 3, figsize=(7.8, 2.5),
        gridspec_kw={"width_ratios": [2.3, 0.72, 0.72], "wspace": 0.55})

    annotated_heatmap(axes[0], img, CELL_LINES, ["train", "val", "test",
                                                 "all"])
    axes[0].set_title("a   Images", loc="left")

    annotated_heatmap(axes[1], obj, CELL_LINES, ["objects"], fmt="{:,.0f}")
    axes[1].set_yticks([])
    axes[1].set_title("b   ecDNA objects", loc="left")

    dens = (obj[:, 0] / img[:, 3]).reshape(-1, 1)
    annotated_heatmap(axes[2], dens, CELL_LINES, ["per image"], fmt="{:,.0f}")
    axes[2].set_yticks([])
    axes[2].set_title("c   Per image", loc="left")

    fig.suptitle("What the benchmark is made of", y=1.06, x=0.06, ha="left",
                 fontsize=9.5, fontweight="semibold")
    fig.text(0.06, -0.10,
             "Cells are shaded within each panel; NCI-H2170 supplies "
             f"{img[1, 3] / np.nansum(img[:, 3]):.0%} of the images and "
             f"{obj[1, 0] / np.nansum(obj[:, 0]):.0%} of the annotated "
             "objects.", fontsize=7, color=INK2, ha="left")
    save(fig, outdir, "figD1_benchmark_composition", src)


# ---------------------------------------------------------------------------
# figD2 — the design matrix
# ---------------------------------------------------------------------------

def fig_design(comps: Dict[str, pd.DataFrame], outdir: Path) -> None:
    """One row per run, one column per cell line, colour = role."""
    if not comps:
        logger.warning("figD2 needs split_composition.csv; skipping.")
        return

    TRAINED, SAMPLED, HELD = 0, 1, 2
    role_mat = np.full((len(RUNS), len(CELL_LINES)), np.nan)
    text = np.empty((len(RUNS), len(CELL_LINES)), dtype=object)
    src_rows, n_train, n_eval = [], [], []

    for i, (run_id, held, _) in enumerate(RUNS):
        c = comps.get(run_id)
        if c is None:
            n_train.append(np.nan)
            n_eval.append(np.nan)
            continue
        tr = c[c.role == "train"].groupby("cell_line").n_images.sum()
        va = c[c.role == "val"].groupby("cell_line").n_images.sum()
        ev = c[c.role == "eval"].groupby("cell_line").n_images.sum()
        n_train.append(int(c[c.role == "train"].n_images.sum()))
        n_eval.append(int(c[c.role == "eval"].n_images.sum()))
        control = run_id.endswith("_control")
        for j, line in enumerate(CELL_LINES):
            n_tr, n_va = int(tr.get(line, 0)), int(va.get(line, 0))
            in_eval = int(ev.get(line, 0))
            fit = n_tr + n_va
            if in_eval and not fit:
                role_mat[i, j] = HELD
                text[i, j] = f"HELD OUT\nall {in_eval} images\nscored"
            elif fit and in_eval:
                role_mat[i, j] = SAMPLED
                text[i, j] = (f"{n_tr} train + {n_va} val\n"
                              f"(sampled)\n{in_eval} test scored")
            elif fit:
                role_mat[i, j] = SAMPLED if control else TRAINED
                text[i, j] = (f"{n_tr} train + {n_va} val"
                              + ("\n(sampled)" if control else ""))
            else:
                text[i, j] = "—"
            src_rows.append({"run": run_id, "cell_line": line,
                             "train_images": n_tr, "val_images": n_va,
                             "images_scored": in_eval,
                             "role": ("held out" if in_eval and not fit
                                      else "sampled into training" if control
                                      else "in training" if fit
                                      else "absent")})

    fig, ax = plt.subplots(figsize=(7.6, 3.7))
    colours = {TRAINED: BLUE_250, SAMPLED: BLUE_400, HELD: BLUE_700}
    for i in range(len(RUNS)):
        for j in range(len(CELL_LINES)):
            v = role_mat[i, j]
            face = colours.get(v, "#f2f2ef")
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                       facecolor=face, edgecolor=SURFACE,
                                       linewidth=2.0, zorder=2))
            ax.text(j, i, text[i, j] or "—", ha="center", va="center",
                    fontsize=7.2, zorder=3, linespacing=1.35,
                    color=SURFACE if v == HELD else INK)

    ax.set_xlim(-0.5, len(CELL_LINES) + 1.55)
    ax.set_ylim(len(RUNS) - 0.5, -0.5)
    ax.set_xticks(range(len(CELL_LINES)), CELL_LINES)
    ax.set_yticks(range(len(RUNS)), [lab for _, _, lab in RUNS])
    ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)

    for i, (tr, ev) in enumerate(zip(n_train, n_eval)):
        if np.isfinite(tr):
            ax.text(len(CELL_LINES) + 0.10, i, f"{int(tr):,}", ha="center",
                    va="center", fontsize=7.5, color=INK, fontweight="semibold")
            ax.text(len(CELL_LINES) + 1.00, i, f"{int(ev):,}", ha="center",
                    va="center", fontsize=7.5, color=INK, fontweight="semibold")
    ax.text(len(CELL_LINES) + 0.10, -0.62, "train\nimages\n(total)", ha="center",
            va="center", fontsize=7, color=INK2, linespacing=1.3)
    ax.text(len(CELL_LINES) + 1.00, -0.62, "images\nscored", ha="center",
            va="center", fontsize=7, color=INK2, linespacing=1.3)

    ax.set_title("How the five runs were built", loc="left", pad=26)
    legend_below(ax, [
        Patch(facecolor=BLUE_250, label="trained on this line"),
        Patch(facecolor=BLUE_400, label="sampled into training (control)"),
        Patch(facecolor=BLUE_700, label="held out entirely, then scored"),
    ], ncol=3, dy=-0.14)
    save(fig, outdir, "figD2_loco_design", pd.DataFrame(src_rows))


# ---------------------------------------------------------------------------
# figD3 — training curves
# ---------------------------------------------------------------------------

def fig_training(hists: Dict[str, Tuple[pd.DataFrame, Dict]],
                 outdir: Path, zoom_from: int = 30) -> None:
    """Full curves on the top row, the plateau enlarged on the bottom row.

    The whole question this figure answers — did the run converge, or was it
    still improving when the schedule ended — lives in the last twenty epochs,
    which are invisible on an axis that also has to show a loss of 1.26.
    """
    if not hists:
        logger.warning("figD3 needs train_history.csv; skipping.")
        return
    present = [(rid, held, lab) for rid, held, lab in RUNS if rid in hists]
    n = len(present)
    short = {rid: ("control (NCI-H2170)" if rid.endswith("_control")
                   else held) for rid, held, _ in present}

    fig, axes = plt.subplots(2, n, figsize=(1.72 * n + 0.9, 4.4),
                             sharex="row", gridspec_kw={"hspace": 0.55})
    src_rows = []

    lo = min(float(np.nanmin(h[[c["train"], c["val"]]].to_numpy()))
             for h, c in hists.values())
    hi_ = max(float(np.nanmax(h[[c["train"], c["val"]]].to_numpy()))
              for h, c in hists.values())
    tails = []
    for rid, _, _ in present:
        h, c = hists[rid]
        t = h.iloc[zoom_from:][[c["train"], c["val"]]].to_numpy()
        tails.append((float(np.nanmin(t)), float(np.nanmax(t))))
    zlo = min(t[0] for t in tails)
    zhi = max(t[1] for t in tails)

    for k, (rid, held, _) in enumerate(present):
        h, c = hists[rid]
        ep = (h[c["epoch"]] if c["epoch"] else
              pd.Series(np.arange(1, len(h) + 1)))
        j = int(np.nanargmin(h[c["val"]].to_numpy()))
        best_ep, best_v = float(ep.iloc[j]), float(h[c["val"]].iloc[j])

        for row, ax in enumerate(axes[:, k]):
            ax.plot(ep, h[c["train"]], color=BLUE_250, lw=1.4)
            ax.plot(ep, h[c["val"]], color=BLUE_700, lw=1.4)
            ax.scatter([best_ep], [best_v], s=28, color=ORANGE, zorder=4,
                       edgecolor=SURFACE, linewidth=1.0)
            tidy(ax, xgrid=False, ygrid=True)
            if row == 0:
                ax.set_title(short[rid], fontsize=8, pad=4)
                ax.set_ylim(lo - 0.03 * (hi_ - lo), hi_ + 0.05 * (hi_ - lo))
            else:
                ax.set_xlim(zoom_from, float(ep.iloc[-1]) + 1)
                pad = 0.10 * (zhi - zlo)
                ax.set_ylim(zlo - pad, zhi + pad)
                right = best_ep > zoom_from + 0.55 * (
                    float(ep.iloc[-1]) - zoom_from)
                ax.annotate(f"{best_v:.4f} @ {int(best_ep)}",
                            (best_ep, best_v),
                            xytext=(-7 if right else 7, 9),
                            textcoords="offset points",
                            ha="right" if right else "left",
                            va="bottom", fontsize=7, color=ORANGE,
                            fontweight="semibold")
                ax.set_xlabel("epoch")
            if k > 0:
                ax.set_yticklabels([])

        for _, r in h.iterrows():
            src_rows.append({
                "run": rid, "held_out_cell_line": held,
                "epoch": r[c["epoch"]] if c["epoch"] else np.nan,
                "train_loss": r[c["train"]], "val_loss": r[c["val"]],
                "lr": r[c["lr"]] if c["lr"] else np.nan})

    axes[0, 0].set_ylabel("loss")
    axes[1, 0].set_ylabel("loss (detail)")
    axes[0, 0].annotate("a   all 70 epochs", (0, 1.16), xycoords="axes fraction",
                        fontsize=8.5, fontweight="semibold", color=INK,
                        ha="left", va="bottom")
    axes[1, 0].annotate(f"b   epochs {zoom_from}\u2013 enlarged",
                        (0, 1.06), xycoords="axes fraction", fontsize=8.5,
                        fontweight="semibold", color=INK, ha="left",
                        va="bottom")
    legend_below(axes[1, n // 2], [
        Line2D([], [], color=BLUE_250, lw=1.8, label="training loss"),
        Line2D([], [], color=BLUE_700, lw=1.8, label="validation loss"),
        Line2D([], [], marker="o", ls="", ms=5.5, color=ORANGE,
               label="checkpoint kept (lowest validation loss)"),
    ], ncol=3, dy=-0.34)
    fig.suptitle("Training, one panel per held-out cell line",
                 y=1.10, x=0.02, ha="left", fontsize=9.5,
                 fontweight="semibold")
    save(fig, outdir, "figD3_training_curves", pd.DataFrame(src_rows))


# ---------------------------------------------------------------------------
# figD4 — precision, recall, F1 on the held-out line
# ---------------------------------------------------------------------------

def fig_pr_f1(tab: pd.DataFrame, outdir: Path, model: str) -> None:
    kinds = ["leave-one-out (test-restricted)", "leave-one-out"]
    if not {"precision", "recall"}.issubset(tab.columns):
        logger.warning("figD4 needs precision and recall columns in %s — "
                       "re-run loco_paper_table.py; skipping.",
                       "source_loco_paper_table_*.csv")
        return
    rows = []
    for line in CELL_LINES:
        for k in kinds:
            s = tab[(tab.held_out_cell_line == line) & (tab.model == model)
                    & (tab.run_kind == k)]
            if len(s):
                r = s.iloc[0]
                rows.append({"cell_line": line, "run_kind": k,
                             "n_images": int(r.n_images),
                             "precision": float(r.precision),
                             "recall": float(r.recall),
                             "obj_f1": float(r.obj_f1)})
                break
    if not rows:
        logger.warning("figD4 found no rows for %s; skipping.", model)
        return
    d = pd.DataFrame(rows).sort_values("obj_f1", ascending=False)

    metrics = [("precision", BLUE_250), ("recall", BLUE_400),
               ("obj_f1", BLUE_700)]
    x = np.arange(len(d))
    w = 0.26
    fig, ax = plt.subplots(figsize=(5.8, 2.7))
    for i, (m, col) in enumerate(metrics):
        off = (i - 1) * w
        ax.bar(x + off, d[m], width=w * 0.92, color=col, zorder=2)
        for xi, v in zip(x + off, d[m]):
            ax.annotate(f"{v:.3f}", (xi, v), xytext=(0, 3),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=6.6, color=INK, rotation=90)
    ax.set_xticks(x, [f"{l}\n(n = {n})" for l, n in
                      zip(d.cell_line, d.n_images)])
    ax.set_ylabel("score on the held-out cell line")
    ax.set_ylim(0, 1.13)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_title(f"Detection on a cell line the model never saw — {model}")
    tidy(ax, xgrid=False, ygrid=True)
    legend_below(ax, [Patch(facecolor=c, label=m.replace("obj_f1", "F1"))
                      for m, c in metrics], ncol=3, dy=-0.30)
    save(fig, outdir, "figD4_precision_recall_f1", d)


# ---------------------------------------------------------------------------
# figD5 — every model on every cell line
# ---------------------------------------------------------------------------

def fig_model_heatmap(tab: pd.DataFrame, ind: pd.DataFrame, outdir: Path,
                      model: str) -> None:
    """Every model on every cell line — the panel behind main Fig. 6d.

    The in-distribution rows pool over ALL images of each cell line, so each
    model's score includes the images it trained on. That is deliberate: the
    held-out test split of three of the four lines contains 11-19 images, and
    per-cell-line values on those splits swing by up to 0.11 (Label Engine on
    NCI-H716 moves 0.710 -> 0.599). The scope is recorded in the source CSV so
    the caption can state it.
    """
    ind = ind.copy()
    ind["model"] = ind["model"].astype(str).str.strip()
    models = [m for m in ind["model"].unique()]
    # rank by mean F1 so the reader sees an ordering, best at the top
    order = (ind[ind.cell_line.isin(CELL_LINES)]
             .groupby("model").object_f1.mean().sort_values(ascending=False))
    models = [m for m in order.index]

    mat, rows, src = [], [], []
    for m in models:
        r = []
        for line in CELL_LINES:
            v = ind[(ind.model == m) & (ind.cell_line == line)]["object_f1"]
            val = float(v.iloc[0]) if len(v) else np.nan
            r.append(val)
            n = ind[(ind.model == m) & (ind.cell_line == line)].get("n_images")
            src.append({"model": m, "cell_line": line, "obj_f1": val,
                        "condition": "trained on this cell line",
                        "scope": "all rows of the cell line",
                        "n_images": int(n.iloc[0]) if n is not None and len(n)
                        else np.nan})
        mat.append(r)
        rows.append(m)

    loco = []
    for line in CELL_LINES:
        s = tab[(tab.held_out_cell_line == line) & (tab.model == model)
                & (tab.run_kind == "leave-one-out")]
        val = float(s.iloc[0].obj_f1) if len(s) else np.nan
        loco.append(val)
        src.append({"model": f"{model} — line unseen", "cell_line": line,
                    "obj_f1": val, "condition": "never trained on this line",
                    "scope": (str(s.iloc[0].get("scope", "")) if len(s) else ""),
                    "n_images": int(s.iloc[0].n_images) if len(s) else np.nan})
    mat.append(loco)
    rows.append(f"{model}\nNEVER trained on this line")

    mat = np.array(mat, dtype=float)
    vmin = float(np.floor(np.nanmin(mat) * 20) / 20)   # nearest 0.05 below
    fig, ax = plt.subplots(figsize=(6.0, 0.46 * len(rows) + 1.6))
    im = annotated_heatmap(ax, mat, rows, CELL_LINES, fmt="{:.3f}",
                           vmin=vmin, vmax=1.0)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("object-level F1", fontsize=7.5, color=INK2)
    cb.ax.tick_params(labelsize=7, color=AXIS, labelcolor=INK2)
    cb.outline.set_visible(False)
    ax.axhline(len(rows) - 1.5, color=INK, lw=1.4)
    ax.xaxis.set_ticks_position("top")
    ax.set_title("Object-level F1, every model on every cell line",
                 loc="left", pad=24)
    fig.text(0.5, -0.02,
             "All rows above the line trained on the cell line they are "
             "scored on. The bottom row did not.",
             fontsize=7, color=INK2, ha="center")
    save(fig, outdir, "figD5_model_by_cell_line_heatmap", pd.DataFrame(src))


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Design, training and per-metric figures for the LOCO run.")
    ap.add_argument("--out-root", default="outputs/eccount_loco")
    ap.add_argument("--policy", default="or_matching",
                    choices=["or_matching", "and_matching"])
    ap.add_argument("--model", default=PEAKS, choices=[PEAKS, THRESH])
    ap.add_argument("--in-distribution-csv",
                    default="release/figures/notebook05/"
                            "source_fig6_f1_by_cell_line_heatmap_long.csv")
    ap.add_argument("--figdir", default=None)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)
    style()

    root = Path(args.out_root)
    outdir = Path(args.figdir) if args.figdir else root / "figures"

    comps = read_compositions(root)
    hists = read_histories(root)

    fig_composition(comps, outdir)
    fig_design(comps, outdir)
    fig_training(hists, outdir)

    tab_path = root / f"source_loco_paper_table_{args.policy}.csv"
    if tab_path.is_file():
        tab = pd.read_csv(tab_path)
        fig_pr_f1(tab, outdir, args.model)
        ind_path = Path(args.in_distribution_csv)
        if ind_path.is_file():
            fig_model_heatmap(tab, pd.read_csv(ind_path), outdir, args.model)
        else:
            logger.warning("missing %s; skipping figD5", ind_path)
    else:
        logger.warning("missing %s; skipping figD4 and figD5 — run "
                       "loco_paper_table.py first", tab_path)

    print(f"\nFigures written to {outdir}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
