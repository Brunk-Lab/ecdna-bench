#!/usr/bin/env python
"""
benchmark_river_roi.py — ecCount inside River-predicted ROIs, dual ground truth.

Scores ecCount (peaks) and ecCount (threshold mask) on every image for which a
River-predicted ROI run exists, against two different ground truths:

  gt_river  ecDNA GT intersected with the *predicted* ROI
            (release/river_roi_run/gt_mask_river)
            -> "did ecCount find what was inside the ROI it was given?"

  gt_full   the full manual ecDNA GT, unrestricted
            (ecDNA_Data/bioimage_archive/gt_image)
            -> "what does the whole pipeline recover end to end?"

The gap between the two is the cost of the ROI step, and it is reported
explicitly as an ecDNA-retention fraction per image.

TWO DISCLOSURES THAT MUST TRAVEL WITH THESE NUMBERS
---------------------------------------------------
1.  The ROI model was trained on the 800 benchmark training images. Any
    agreement figure computed over the whole benchmark subset is therefore
    roughly 70 % training data. The honest number is the val+test subset
    (345 images). This script tags every row with `roi_model_saw_image` and
    reports the subsets separately. Supplementary Methods section 11.

2.  Scoring against `gt_river` means the ground truth was filtered by the same
    model under evaluation. Those results are conditional on the ROI, not
    end-to-end, and every output column is labelled `gt_variant` so the two
    can never be silently pooled.

Matching is the canonical operating point used everywhere in the paper:
OR policy, d_max = 20 px, IoU_min = 0.1, alpha = 0.5, 8-connectivity,
minimum object area 3 px. Object F1 is pooled (micro-averaged): TP, FP and FN
are summed across images before F1 is computed, never averaged per image.

Usage
-----
    python scripts/benchmark_river_roi.py --workers 8
    python scripts/benchmark_river_roi.py --limit 40 --workers 1   # smoke test

Outputs land in --out-dir (default release/river_roi_run/benchmark/):
    per_image_metrics_river.csv
    summary_by_cell_line.csv
    summary_by_subset.csv
    roi_retention_per_image.csv
    fig_roi_*.svg / .png
    run_manifest.csv
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ==============================================================================
# Canonical parameters. Do not vary these without changing the paper.
# ==============================================================================

D_MAX = 20.0
IOU_MIN = 0.1
ALPHA = 0.5
POLICY = "OR"
CONNECTIVITY = 8
MIN_AREA = 3

MODELS = {
    "ecCount (peaks)": "eccount_peaks",
    "ecCount (threshold mask)": "eccount_threshold",
}

GT_VARIANTS = {
    "gt_river": "GT restricted to the predicted ROI (conditional on ROI)",
    "gt_full": "Full manual GT (end-to-end)",
}

MODEL_ORDER = ["ecCount (peaks)", "ecCount (threshold mask)"]
MODEL_COLORS = {
    "ecCount (peaks)": "#F58518",
    "ecCount (threshold mask)": "#9C755F",
}
CELL_LINE_COLORS = {
    "NCI-H2170": "#E69F00",
    "SNU16": "#009E73",
    "COLO320DM": "#CC79A7",
    "NCI-H716": "#0072B2",
    "SUM159PT": "#56B4E9",
}
CL_ORDER = ["NCI-H2170", "SNU16", "COLO320DM", "NCI-H716", "SUM159PT"]

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF")


# ==============================================================================
# Small helpers.
# ==============================================================================


def canonical_cell_line(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "unknown"
    s = str(x).strip().upper().replace("-", "").replace("_", "").replace(" ", "")
    return {
        "NCIH2170": "NCI-H2170",
        "SUM159PT": "SUM159PT",
        "SNU16": "SNU16",
        "COLO320DM": "COLO320DM",
        "NCIH716": "NCI-H716",
    }.get(s, str(x).strip())


def infer_cell_line_from_uid(uid: str) -> str:
    u = uid.lower()
    for token, name in (
        ("ncih2170", "NCI-H2170"),
        ("sum159", "SUM159PT"),
        ("snu16", "SNU16"),
        ("colo320", "COLO320DM"),
        ("ncih716", "NCI-H716"),
    ):
        if token in u:
            return name
    return "unknown"


def index_masks(folder: Path) -> dict[str, Path]:
    """Map uid -> path for every image file in a folder, ignoring extension."""
    out: dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_SUFFIXES:
            out.setdefault(p.stem, p)
    return out


def pooled_f1(tp, fp, fn):
    denom = 2 * np.asarray(tp) + np.asarray(fp) + np.asarray(fn)
    return np.where(denom > 0, 2 * np.asarray(tp) / denom, np.nan)


# ==============================================================================
# Per-image work. Runs in a worker process.
# ==============================================================================


def score_one_image(job: dict) -> list[dict]:
    """Score one image for every (model, gt_variant) combination."""
    from ecdna_bench.data.io import load_mask
    from ecdna_bench.evaluation.matching import match_objects
    from ecdna_bench.evaluation.metrics import object_metrics, pixel_metrics
    from ecdna_bench.evaluation.objects import objects_from_mask

    uid = job["uid"]
    rows: list[dict] = []

    try:
        gt_masks = {
            name: load_mask(path)
            for name, path in job["gt_paths"].items()
            if path is not None
        }
        pred_masks = {
            name: load_mask(path)
            for name, path in job["pred_paths"].items()
            if path is not None
        }
    except Exception as exc:  # noqa: BLE001
        return [{
            "uid": uid, "cell_line": job["cell_line"], "error": f"load: {exc}",
        }]

    # Object extraction once per mask, reused across pairings.
    gt_objs, gt_counts = {}, {}
    for name, m in gt_masks.items():
        objs = objects_from_mask(m, min_area=MIN_AREA, connectivity=CONNECTIVITY)
        gt_objs[name] = objs
        gt_counts[name] = len(objs)

    pred_objs, pred_counts = {}, {}
    for name, m in pred_masks.items():
        objs = objects_from_mask(m, min_area=MIN_AREA, connectivity=CONNECTIVITY)
        pred_objs[name] = objs
        pred_counts[name] = len(objs)

    shapes = {m.shape[:2] for m in list(gt_masks.values()) + list(pred_masks.values())}
    shape_ok = len(shapes) <= 1

    for model_name in job["models_present"]:
        for gt_name in job["gts_present"]:
            try:
                res = match_objects(
                    pred_objs[model_name],
                    gt_objs[gt_name],
                    d_max=D_MAX,
                    min_iou=IOU_MIN,
                    alpha=ALPHA,
                    policy=POLICY,
                )
                om = object_metrics(res)
                pm = pixel_metrics(gt_masks[gt_name], pred_masks[model_name])

                pc = pred_counts[model_name]
                gc = gt_counts[gt_name]
                rows.append({
                    "uid": uid,
                    "cell_line": job["cell_line"],
                    "subset": job["subset"],
                    "roi_model_saw_image": job["roi_model_saw_image"],
                    "model": model_name,
                    "gt_variant": gt_name,
                    "shape_ok": shape_ok,
                    "obj_tp": res.tp, "obj_fp": res.fp, "obj_fn": res.fn,
                    "obj_ignored": res.ignored,
                    "obj_precision": om["precision"],
                    "obj_recall": om["recall"],
                    "obj_f1": om["f1"],
                    "pred_count": pc,
                    "gt_count": gc,
                    "count_abs_error": abs(pc - gc),
                    "count_signed_error": pc - gc,
                    "pix_tp": pm["pixel_tp"], "pix_fp": pm["pixel_fp"],
                    "pix_fn": pm["pixel_fn"], "pix_tn": pm["pixel_tn"],
                    "pix_precision": pm["pixel_precision"],
                    "pix_recall": pm["pixel_recall"],
                    "pix_dice": pm["pixel_dice"],
                    "pix_iou": pm["pixel_iou"],
                    "error": "",
                })
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "uid": uid, "cell_line": job["cell_line"],
                    "model": model_name, "gt_variant": gt_name,
                    "error": f"score: {exc}",
                })

    # ROI retention: what fraction of true GT objects survived the ROI step.
    if "gt_full" in gt_counts and "gt_river" in gt_counts:
        n_full = gt_counts["gt_full"]
        n_river = gt_counts["gt_river"]
        rows.append({
            "uid": uid,
            "cell_line": job["cell_line"],
            "subset": job["subset"],
            "roi_model_saw_image": job["roi_model_saw_image"],
            "model": "__retention__",
            "gt_variant": "__retention__",
            "gt_count": n_full,
            "pred_count": n_river,
            "ecdna_retention": (n_river / n_full) if n_full > 0 else np.nan,
            "error": "",
        })

    return rows


# ==============================================================================
# Inventory.
# ==============================================================================


def build_inventory(args) -> pd.DataFrame:
    repo = Path(args.repo_root)
    river = Path(args.river_run)

    pred_dirs = {name: river / "masks" / sub for name, sub in MODELS.items()}
    gt_dirs = {
        "gt_river": river / "gt_mask_river",
        "gt_full": Path(args.gt_full_dir),
    }

    print("Directories")
    for label, d in list(pred_dirs.items()) + list(gt_dirs.items()):
        n = len(index_masks(d))
        print(f"  {label:28s} {n:5d} files  {d}")
        if n == 0:
            print(f"    WARNING: no image files found in {d}")

    pred_idx = {k: index_masks(v) for k, v in pred_dirs.items()}
    gt_idx = {k: index_masks(v) for k, v in gt_dirs.items()}

    # An image is scoreable if it has at least one prediction and one GT.
    uids = set()
    for d in list(pred_idx.values()):
        uids |= set(d)
    uids = sorted(u for u in uids if any(u in g for g in gt_idx.values()))

    # Cell line and split, from the manifests where available.
    meta = {}
    # in_benchmark must come from metadata.csv ALONE. Populating it from both
    # manifests made every one of the 2,986 images match, so everything was
    # labelled "benchmark" — including SUM159PT, which is not in the benchmark.
    bench_uids: set[str] = set()
    for rel in (
        "release/manifests/full_counts_master.csv",
        "release/manifests/metadata.csv",
    ):
        p = repo / rel
        if not p.exists():
            continue
        df = pd.read_csv(p)
        uid_col = next((c for c in ("unique_id", "uid", "image_id") if c in df.columns), None)
        if uid_col is None:
            continue
        for _, r in df.iterrows():
            entry = meta.setdefault(str(r[uid_col]), {})
            if "cell_line" in df.columns and "cell_line" not in entry:
                entry["cell_line"] = canonical_cell_line(r["cell_line"])
            if "split" in df.columns and "split" not in entry:
                entry["split"] = str(r["split"])
            if rel.endswith("metadata.csv"):
                bench_uids |= set(df[uid_col].astype(str))
        print(f"  metadata: {len(df):,} rows from {rel}")

    train_ids = set()
    tp = repo / "release/split_files/train_ids.csv"
    if tp.exists():
        train_ids = set(pd.read_csv(tp).iloc[:, 0].astype(str))
        print(f"  ROI-model training ids: {len(train_ids):,} from {tp.name}")
    else:
        print(f"  WARNING: {tp} not found; roi_model_saw_image will be unknown")

    rows = []
    for uid in uids:
        entry = meta.get(uid, {})
        cl = entry.get("cell_line") or infer_cell_line_from_uid(uid)
        split = entry.get("split", "extension")
        rows.append({
            "uid": uid,
            "cell_line": canonical_cell_line(cl),
            "split": split,
            "in_benchmark": uid in bench_uids,
            "roi_model_saw_image": uid in train_ids,
            **{f"pred_{k}": str(pred_idx[k].get(uid, "")) for k in pred_idx},
            **{f"path_{k}": str(gt_idx[k].get(uid, "")) for k in gt_idx},
        })

    inv = pd.DataFrame(rows)
    inv["subset"] = np.where(
        ~inv["in_benchmark"], "extension (not in benchmark)",
        np.where(inv["roi_model_saw_image"],
                 "benchmark train (SEEN by ROI model)",
                 "benchmark val+test (held out)"),
    )
    return inv


# ==============================================================================
# Aggregation.
# ==============================================================================


def pool(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = (
        df.groupby(keys, observed=False, as_index=False)
        .agg(
            n_images=("uid", "nunique"),
            tp=("obj_tp", "sum"), fp=("obj_fp", "sum"), fn=("obj_fn", "sum"),
            ignored=("obj_ignored", "sum"),
            pix_tp=("pix_tp", "sum"), pix_fp=("pix_fp", "sum"),
            pix_fn=("pix_fn", "sum"),
            count_mae=("count_abs_error", "mean"),
            count_median_ae=("count_abs_error", "median"),
            count_bias=("count_signed_error", "mean"),
            gt_count_mean=("gt_count", "mean"),
            pred_count_mean=("pred_count", "mean"),
        )
    )
    g["object_f1"] = pooled_f1(g["tp"], g["fp"], g["fn"])
    g["object_precision"] = np.where(
        (g["tp"] + g["fp"]) > 0, g["tp"] / (g["tp"] + g["fp"]), np.nan)
    g["object_recall"] = np.where(
        (g["tp"] + g["fn"]) > 0, g["tp"] / (g["tp"] + g["fn"]), np.nan)
    g["pixel_dice"] = pooled_f1(g["pix_tp"], g["pix_fp"], g["pix_fn"])
    g["pixel_iou"] = np.where(
        (g["pix_tp"] + g["pix_fp"] + g["pix_fn"]) > 0,
        g["pix_tp"] / (g["pix_tp"] + g["pix_fp"] + g["pix_fn"]), np.nan)
    g["pixel_precision"] = np.where(
        (g["pix_tp"] + g["pix_fp"]) > 0,
        g["pix_tp"] / (g["pix_tp"] + g["pix_fp"]), np.nan)
    g["pixel_recall"] = np.where(
        (g["pix_tp"] + g["pix_fn"]) > 0,
        g["pix_tp"] / (g["pix_tp"] + g["pix_fn"]), np.nan)
    g["relative_bias_pct"] = np.where(
        g["gt_count_mean"] > 0, 100 * g["count_bias"] / g["gt_count_mean"], np.nan)
    return g


# ==============================================================================
# Figures.
# ==============================================================================


def make_figures(per_image, retention, by_cl, by_subset, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "legend.fontsize": 6.2,
        "axes.linewidth": 0.8, "savefig.dpi": 300, "svg.fonttype": "none",
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
    })

    def save(fig, name):
        for ext in ("svg", "png"):
            fig.savefig(out_dir / f"{name}.{ext}", format=ext,
                        bbox_inches="tight", dpi=150 if ext == "png" else None)
        print(f"  saved {name}.svg / .png")
        plt.close(fig)

    cls = [c for c in CL_ORDER if c in set(by_cl["cell_line"])]
    variants = ["gt_river", "gt_full"]
    vlabel = {"gt_river": "GT inside predicted ROI\n(conditional)",
              "gt_full": "Full GT\n(end to end)"}

    # ---- Fig 1: object F1 by cell line, both GT variants ------------------
    for metric, ylab, fname, ylim in [
        ("object_f1", "Object-level F1", "fig_roi_f1_by_cell_line", (0, 1.05)),
        ("pixel_dice", "Pixel Dice", "fig_roi_dice_by_cell_line", None),
        ("pixel_iou", "Pixel IoU", "fig_roi_iou_by_cell_line", None),
    ]:
        fig, axes = plt.subplots(1, len(variants), figsize=(4.4 * len(variants), 3.0),
                                 sharey=True)
        axes = np.atleast_1d(axes)
        for ax, gv in zip(axes, variants):
            sub = by_cl[by_cl["gt_variant"] == gv]
            x = np.arange(len(cls))
            w = 0.36
            for i, m in enumerate(MODEL_ORDER):
                vals = [
                    float(sub.loc[(sub.cell_line == c) & (sub.model == m), metric].iloc[0])
                    if not sub.loc[(sub.cell_line == c) & (sub.model == m)].empty else np.nan
                    for c in cls
                ]
                ax.bar(x + (i - 0.5) * w, vals, width=w, color=MODEL_COLORS[m],
                       edgecolor="white", linewidth=0.4, label=m)
                for xi, v in zip(x + (i - 0.5) * w, vals):
                    if not np.isnan(v):
                        ax.text(xi, v + 0.015, f"{v:.3f}", ha="center",
                                va="bottom", fontsize=5.4)
            ns = [int(sub.loc[sub.cell_line == c, "n_images"].max()) if not
                  sub.loc[sub.cell_line == c].empty else 0 for c in cls]
            ax.set_xticks(x)
            ax.set_xticklabels([f"{c}\n(n = {n})" for c, n in zip(cls, ns)], fontsize=5.8)
            ax.set_title(vlabel[gv], fontsize=7.4)
            if ylim:
                ax.set_ylim(*ylim)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
            ax.set_axisbelow(True)
        axes[0].set_ylabel(ylab)
        axes[0].legend(loc="lower left", frameon=False, fontsize=5.8, handlelength=1.1)
        fig.suptitle(f"ecCount inside River-predicted ROIs — {ylab}", fontsize=8, y=1.02)
        fig.tight_layout()
        save(fig, fname)

    # ---- Fig 2: count MAE and relative bias -------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    for ax, (metric, ylab) in zip(axes, [("count_mae", "Count MAE (ecDNA per image)"),
                                         ("relative_bias_pct", "Relative bias (% of burden)")]):
        x = np.arange(len(cls))
        w = 0.2
        k = 0
        for gv in variants:
            for m in MODEL_ORDER:
                sub = by_cl[(by_cl.gt_variant == gv) & (by_cl.model == m)]
                vals = [float(sub.loc[sub.cell_line == c, metric].iloc[0])
                        if not sub.loc[sub.cell_line == c].empty else np.nan for c in cls]
                ax.bar(x + (k - 1.5) * w, vals, width=w, color=MODEL_COLORS[m],
                       alpha=1.0 if gv == "gt_river" else 0.55,
                       edgecolor="white", linewidth=0.4,
                       label=f"{m} · {gv}")
                k += 1
        if metric == "relative_bias_pct":
            ax.axhline(0, color="0.2", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(cls, fontsize=5.8, rotation=18, ha="right")
        ax.set_ylabel(ylab)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                   fontsize=5.4, handlelength=1.1)
    fig.suptitle("Counting accuracy inside River-predicted ROIs", fontsize=8, y=1.03)
    fig.tight_layout(rect=[0, 0, 0.80, 1])
    save(fig, "fig_roi_count_mae_bias")

    # ---- Fig 3: ecDNA retention — the bimodality check --------------------
    if not retention.empty:
        fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
        ax = axes[0]
        for c in cls:
            v = retention.loc[retention.cell_line == c, "ecdna_retention"].dropna()
            if len(v) == 0:
                continue
            ax.hist(v, bins=np.linspace(0, 1.0, 26), histtype="step", linewidth=1.3,
                    color=CELL_LINE_COLORS.get(c, "0.4"), label=f"{c} (n = {len(v)})",
                    density=True)
        ax.set_xlabel("Fraction of true ecDNA objects retained inside predicted ROI")
        ax.set_ylabel("Density")
        ax.set_title("ROI retention — a second mode near zero is ROI failure", fontsize=7.4)
        ax.legend(frameon=False, fontsize=5.6)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)

        ax = axes[1]
        data, labels, colours = [], [], []
        for c in cls:
            v = retention.loc[retention.cell_line == c, "ecdna_retention"].dropna()
            if len(v) == 0:
                continue
            data.append(v.values)
            labels.append(f"{c}\n(n = {len(v)})")
            colours.append(CELL_LINE_COLORS.get(c, "0.4"))
        if data:
            bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False)
            for patch, col in zip(bp["boxes"], colours):
                patch.set_facecolor(col)
                patch.set_alpha(0.55)
                patch.set_edgecolor("0.25")
            for k in ("medians", "whiskers", "caps"):
                for art in bp[k]:
                    art.set_color("0.25")
            for i, (v, col) in enumerate(zip(data, colours), start=1):
                jitter = np.random.default_rng(0).uniform(-0.14, 0.14, len(v))
                ax.scatter(i + jitter, v, s=5, color=col, alpha=0.45,
                           edgecolor="none", zorder=3)
            ax.set_xticklabels(labels, fontsize=5.8)
        ax.set_ylabel("ecDNA retention")
        ax.set_ylim(-0.03, 1.05)
        ax.set_title("Per-cell-line retention", fontsize=7.4)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        fig.suptitle("Cost of the ROI step, by cell line", fontsize=8, y=1.03)
        fig.tight_layout()
        save(fig, "fig_roi_retention")

    # ---- Fig 4: count agreement scatter -----------------------------------
    fig, axes = plt.subplots(len(variants), len(MODEL_ORDER),
                             figsize=(4.2 * len(MODEL_ORDER), 3.4 * len(variants)))
    axes = np.atleast_2d(axes)
    hi = 10 ** np.ceil(np.log10(max(
        per_image["gt_count"].max(), per_image["pred_count"].max(), 10) + 1))
    for r, gv in enumerate(variants):
        for c_i, m in enumerate(MODEL_ORDER):
            ax = axes[r, c_i]
            sub = per_image[(per_image.gt_variant == gv) & (per_image.model == m)]
            for c in cls:
                s = sub[sub.cell_line == c]
                ax.scatter(s["gt_count"] + 1, s["pred_count"] + 1, s=6, alpha=0.4,
                           color=CELL_LINE_COLORS.get(c, "0.4"), edgecolor="none",
                           rasterized=True, label=c)
            ax.plot([1, hi], [1, hi], "--", color="0.15", linewidth=0.7)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(1, hi); ax.set_ylim(1, hi)
            mae = sub["count_abs_error"].mean()
            bias = sub["count_signed_error"].mean()
            ax.set_title(f"{m}\n{gv}: MAE = {mae:.1f}, bias = {bias:+.1f}", fontsize=6.8)
            ax.grid(linestyle="--", linewidth=0.35, alpha=0.35)
            ax.set_axisbelow(True)
            if r == len(variants) - 1:
                ax.set_xlabel("Ground-truth count + 1", fontsize=6.5)
            if c_i == 0:
                ax.set_ylabel("Predicted count + 1", fontsize=6.5)
    axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                       fontsize=5.8, markerscale=1.6)
    fig.suptitle("Predicted versus true ecDNA count inside predicted ROIs",
                 fontsize=8, y=1.01)
    fig.tight_layout(rect=[0, 0, 0.88, 1])
    save(fig, "fig_roi_count_agreement")

    # ---- Fig 5: the training-overlap disclosure ---------------------------
    order = ["benchmark train (SEEN by ROI model)",
             "benchmark val+test (held out)",
             "extension (not in benchmark)"]
    present = [s for s in order if s in set(by_subset["subset"])]
    if present:
        fig, ax = plt.subplots(figsize=(6.4, 3.0))
        x = np.arange(len(present))
        w = 0.2
        k = 0
        for gv in variants:
            for m in MODEL_ORDER:
                sub = by_subset[(by_subset.gt_variant == gv) & (by_subset.model == m)]
                vals = [float(sub.loc[sub.subset == s, "object_f1"].iloc[0])
                        if not sub.loc[sub.subset == s].empty else np.nan for s in present]
                ax.bar(x + (k - 1.5) * w, vals, width=w, color=MODEL_COLORS[m],
                       alpha=1.0 if gv == "gt_river" else 0.55,
                       edgecolor="white", linewidth=0.4, label=f"{m} · {gv}")
                k += 1
        ns = [int(by_subset.loc[by_subset.subset == s, "n_images"].max()) for s in present]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{s}\n(n = {n})" for s, n in zip(present, ns)], fontsize=5.6)
        ax.set_ylabel("Object-level F1 (pooled)")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.35)
        ax.set_axisbelow(True)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False,
                  fontsize=5.4, handlelength=1.1)
        ax.set_title("Quote the held-out subset: the ROI model was trained on the "
                     "benchmark training split", fontsize=7.4)
        fig.tight_layout(rect=[0, 0, 0.78, 1])
        save(fig, "fig_roi_training_overlap")


# ==============================================================================
# Main.
# ==============================================================================


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-root",
                    default="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench")
    ap.add_argument("--river-run", default=None,
                    help="default: <repo-root>/release/river_roi_run")
    ap.add_argument("--gt-full-dir",
                    default="/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/"
                            "bioimage_archive/gt_image")
    ap.add_argument("--out-dir", default=None,
                    help="default: <river-run>/benchmark")
    ap.add_argument("--workers", type=int, default=1,
                    help="Set this to match --cpus-per-task. SLURM allocates the "
                         "cores you asked for, not the node's.")
    ap.add_argument("--limit", type=int, default=0, help="Score only N images.")
    ap.add_argument("--checkpoint-every", type=int, default=200)
    ap.add_argument("--resume", action="store_true",
                    help="Skip uids already present in the checkpoint.")
    ap.add_argument("--inventory-only", action="store_true")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    args.river_run = args.river_run or str(repo / "release" / "river_roi_run")
    args.out_dir = args.out_dir or str(Path(args.river_run) / "benchmark")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    src = repo / "src"
    sys.path.insert(0, str(src if src.exists() else repo))

    print("=" * 78)
    print("ecCount inside River-predicted ROIs — dual ground truth")
    print("=" * 78)
    print(f"repo      : {repo}")
    print(f"river run : {args.river_run}")
    print(f"full GT   : {args.gt_full_dir}")
    print(f"out dir   : {out_dir}")
    print(f"matching  : {POLICY}, d_max={D_MAX:g} px, IoU_min={IOU_MIN}, "
          f"alpha={ALPHA}, conn={CONNECTIVITY}, min_area={MIN_AREA}")
    print()

    inv = build_inventory(args)
    if inv.empty:
        print("\nNothing to score: no uid had both a prediction and a ground truth.")
        return 1

    inv.to_csv(out_dir / "inventory.csv", index=False)
    print(f"\nScoreable images: {len(inv):,}")
    print(inv.groupby(["cell_line", "subset"]).size().to_string())

    if args.inventory_only:
        print("\n--inventory-only: stopping before scoring.")
        return 0

    ckpt = out_dir / "per_image_metrics_river.partial.csv"
    done: set[str] = set()
    if args.resume and ckpt.exists():
        done = set(pd.read_csv(ckpt)["uid"].astype(str))
        print(f"Resuming: {len(done):,} uids already scored.")

    work = inv[~inv["uid"].isin(done)]
    if args.limit:
        work = work.head(args.limit)
    print(f"To score now: {len(work):,} images\n")

    jobs = []
    for _, r in work.iterrows():
        preds = {m: (Path(r[f"pred_{m}"]) if r[f"pred_{m}"] else None) for m in MODELS}
        gts = {g: (Path(r[f"path_{g}"]) if r[f"path_{g}"] else None) for g in GT_VARIANTS}
        jobs.append({
            "uid": r["uid"], "cell_line": r["cell_line"], "subset": r["subset"],
            "roi_model_saw_image": bool(r["roi_model_saw_image"]),
            "pred_paths": preds, "gt_paths": gts,
            "models_present": [m for m, p in preds.items() if p is not None],
            "gts_present": [g for g, p in gts.items() if p is not None],
        })

    rows: list[dict] = []
    t0 = time.time()

    def flush():
        if not rows:
            return
        df = pd.DataFrame(rows)
        df.to_csv(ckpt, mode="a", header=not ckpt.exists(), index=False)
        rows.clear()

    if args.workers <= 1:
        for i, job in enumerate(jobs, start=1):
            rows.extend(score_one_image(job))
            if i % args.checkpoint_every == 0:
                flush()
                el = time.time() - t0
                print(f"  {i:5d}/{len(jobs)}  {el:7.1f}s  "
                      f"eta {el / i * (len(jobs) - i):7.1f}s", flush=True)
        flush()
    else:
        print(f"Using {args.workers} worker processes.")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(score_one_image, j): j["uid"] for j in jobs}
            for i, fut in enumerate(as_completed(futs), start=1):
                try:
                    rows.extend(fut.result())
                except Exception:  # noqa: BLE001
                    print(f"  worker failed on {futs[fut]}")
                    traceback.print_exc()
                if i % args.checkpoint_every == 0:
                    flush()
                    el = time.time() - t0
                    print(f"  {i:5d}/{len(jobs)}  {el:7.1f}s  "
                          f"eta {el / i * (len(jobs) - i):7.1f}s", flush=True)
        flush()

    print(f"\nScoring finished in {time.time() - t0:.1f}s")

    allrows = pd.read_csv(ckpt)

    # An empty error string round-trips through CSV as NaN, so normalise before
    # testing it. Getting this wrong silently discards every good row.
    if "error" in allrows.columns:
        err_col = allrows["error"].fillna("").astype(str).str.strip()
    else:
        err_col = pd.Series("", index=allrows.index, dtype=str)
    allrows["error"] = err_col

    errors = allrows[err_col.str.len() > 0]
    if not errors.empty:
        errors.to_csv(out_dir / "errors.csv", index=False)
        print(f"WARNING: {len(errors)} rows carried an error; see errors.csv")

    retention = allrows[allrows["model"] == "__retention__"].copy()
    per_image = allrows[
        (allrows["model"] != "__retention__") & (err_col.str.len() == 0)
    ].copy()

    if per_image.empty:
        raise RuntimeError(
            "No scoreable rows survived. Check errors.csv and the mask "
            "directories before interpreting anything."
        )

    per_image.to_csv(out_dir / "per_image_metrics_river.csv", index=False)
    retention.to_csv(out_dir / "roi_retention_per_image.csv", index=False)

    by_cl = pool(per_image, ["cell_line", "model", "gt_variant"])
    by_subset = pool(per_image, ["subset", "model", "gt_variant"])
    by_cl_subset = pool(per_image, ["cell_line", "subset", "model", "gt_variant"])
    overall = pool(per_image, ["model", "gt_variant"])

    by_cl.to_csv(out_dir / "summary_by_cell_line.csv", index=False)
    by_subset.to_csv(out_dir / "summary_by_subset.csv", index=False)
    by_cl_subset.to_csv(out_dir / "summary_by_cell_line_and_subset.csv", index=False)
    overall.to_csv(out_dir / "summary_overall.csv", index=False)

    cols = ["model", "gt_variant", "n_images", "object_f1", "object_precision",
            "object_recall", "pixel_dice", "pixel_iou", "count_mae", "count_bias",
            "relative_bias_pct"]
    print("\n── Overall ──")
    print(overall[cols].round(3).to_string(index=False))
    print("\n── By cell line ──")
    print(by_cl[["cell_line"] + cols].round(3).to_string(index=False))
    print("\n── By subset (the training-overlap disclosure) ──")
    print(by_subset[["subset"] + cols].round(3).to_string(index=False))

    if not retention.empty:
        ret = (retention.groupby("cell_line")["ecdna_retention"]
               .agg(n="size", mean="mean", median="median",
                    q10=lambda s: s.quantile(0.10),
                    frac_below_50pct=lambda s: float((s < 0.5).mean()))
               .reset_index())
        ret.to_csv(out_dir / "roi_retention_summary.csv", index=False)
        print("\n── ROI retention (fraction of true ecDNA kept inside predicted ROI) ──")
        print(ret.round(3).to_string(index=False))
        print("\nA high frac_below_50pct with a high median is the bimodal ROI "
              "failure that must be disclosed.")

    if not args.no_figures:
        print("\nFigures:")
        make_figures(per_image, retention, by_cl, by_subset, out_dir)

    manifest = pd.DataFrame([
        {"file": p.name, "size_kb": round(p.stat().st_size / 1024, 1)}
        for p in sorted(out_dir.iterdir()) if p.is_file()
    ])
    manifest.to_csv(out_dir / "run_manifest.csv", index=False)

    (out_dir / "run_params.json").write_text(json.dumps({
        "policy": POLICY, "d_max": D_MAX, "iou_min": IOU_MIN, "alpha": ALPHA,
        "connectivity": CONNECTIVITY, "min_area": MIN_AREA,
        "repo_root": str(repo), "river_run": args.river_run,
        "gt_full_dir": args.gt_full_dir, "n_images": int(per_image["uid"].nunique()),
        "disclosures": [
            "gt_river results are conditional on the predicted ROI, not end-to-end.",
            "The ROI model was trained on the benchmark training split; quote the "
            "held-out subset for agreement figures.",
        ],
    }, indent=2), encoding="utf-8")

    print(f"\nDone. {len(manifest)} files in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
