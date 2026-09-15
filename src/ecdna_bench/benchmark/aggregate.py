"""
ecdna_bench.benchmark.aggregate
=================================
Aggregate per-image benchmark metrics into group-level summaries.

Four aggregation axes
---------------------
* Overall (one row per model)
* By split (train / val / test)
* By cell line
* By count bin [0–9, 10–49, 50–149, 150–299, 300+]

Public API
----------
``aggregate_benchmark(per_image_df) -> AggregateResult``
``write_frozen_results(agg, out_dir)``   → ``release/frozen_results/`` CSVs

Count metrics computed here (not in run.py) so they aggregate correctly
(you cannot average per-image MAE; you must recompute from all values).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ecdna_bench.benchmark.registry import MODEL_ORDER
from ecdna_bench.evaluation.metrics import (
    _mdape,
    _safe_corr,
    _trimmed_mae,
)

# pearsonr / spearmanr are passed as callables into _safe_corr
from scipy.stats import pearsonr, spearmanr

logger = logging.getLogger(__name__)

__all__ = ["AggregateResult", "aggregate_benchmark", "write_frozen_results"]

# Frozen ordered categories
SPLIT_ORDER:   List[str] = ["ALL", "train", "val", "test"]
DENSITY_ORDER: List[str] = ["0-9", "10-49", "50-149", "150-299", "300+"]


def _count_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    y_true = sub["gt_count"].values.astype(float)
    y_pred = sub["pred_count"].values.astype(float)
    n = len(y_true)
    if n == 0:
        nan = float("nan")
        return dict(count_mae=nan, count_trimmed_mae=nan, count_mdape=nan,
                    count_rmse=nan, count_bias=nan,
                    count_pearson_r=nan, count_spearman_rho=nan)
    return {
        "count_mae":          float(np.mean(np.abs(y_pred - y_true))),
        "count_trimmed_mae":  _trimmed_mae(y_true, y_pred, trim_percent=5.0),
        "count_mdape":        _mdape(y_true, y_pred, ignore_zero_gt=True),
        "count_rmse":         float(np.sqrt(np.mean((y_pred - y_true) ** 2))),
        "count_bias":         float(np.mean(y_pred - y_true)),
        "count_pearson_r":    _safe_corr(pearsonr,  y_true, y_pred),
        "count_spearman_rho": _safe_corr(spearmanr, y_true, y_pred),
    }


# ---------------------------------------------------------------------------
# Group summarisation
# ---------------------------------------------------------------------------

def _summarize_group(sub: pd.DataFrame) -> Dict[str, object]:
    tp  = int(sub["obj_tp"].sum());  fp  = int(sub["obj_fp"].sum())
    fn  = int(sub["obj_fn"].sum());  ign = int(sub["obj_ignored"].sum())
    n_pred = int(sub["pred_count"].sum()); n_gt = int(sub["gt_count"].sum())

    obj_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    obj_rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom    = 2 * tp + fp + fn
    obj_f1   = 2 * tp / denom if denom > 0 else 0.0

    pix_tp = int(sub["pix_tp"].sum()); pix_fp = int(sub["pix_fp"].sum())
    pix_fn = int(sub["pix_fn"].sum()); pix_tn = int(sub["pix_tn"].sum())
    pix_prec = pix_tp / (pix_tp + pix_fp) if (pix_tp + pix_fp) > 0 else 0.0
    pix_rec  = pix_tp / (pix_tp + pix_fn) if (pix_tp + pix_fn) > 0 else 0.0
    pix_f1   = 2*pix_prec*pix_rec / (pix_prec+pix_rec) if (pix_prec+pix_rec) > 0 else 0.0
    pix_iou  = pix_tp / (pix_tp + pix_fp + pix_fn) if (pix_tp + pix_fp + pix_fn) > 0 else 0.0
    pix_dice = 2*pix_tp / (2*pix_tp + pix_fp + pix_fn) if (2*pix_tp+pix_fp+pix_fn) > 0 else 0.0

    return {
        "n_images":     len(sub),
        "obj_tp": tp, "obj_fp": fp, "obj_fn": fn, "obj_ignored": ign,
        "pred_count": n_pred, "gt_count": n_gt,
        "obj_precision": obj_prec, "obj_recall": obj_rec, "obj_f1": obj_f1,
        "pix_tp": pix_tp, "pix_fp": pix_fp, "pix_fn": pix_fn, "pix_tn": pix_tn,
        "pix_precision": pix_prec, "pix_recall": pix_rec, "pix_f1": pix_f1,
        "pix_iou": pix_iou, "pix_dice": pix_dice,
        **_count_metrics(sub),
    }


def _agg_overall(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        sub = df[df["model"] == model]
        if len(sub) == 0: continue
        rows.append({"model": model, **_summarize_group(sub)})
    out = pd.DataFrame(rows)
    if len(out):
        out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
        out = out.sort_values("model").reset_index(drop=True)
    return out


def _agg_by_group(df: pd.DataFrame, col: str, cat_order: Optional[List[str]] = None) -> pd.DataFrame:
    rows = []
    for model in MODEL_ORDER:
        sub_m = df[df["model"] == model]
        if len(sub_m) == 0: continue
        groups = (cat_order or sorted(sub_m[col].dropna().astype(str).unique()))
        for g in groups:
            sub = sub_m[sub_m[col].astype(str) == str(g)]
            if len(sub) == 0: continue
            rows.append({"model": model, col: g, **_summarize_group(sub)})
    out = pd.DataFrame(rows)
    if len(out):
        out["model"] = pd.Categorical(out["model"], categories=MODEL_ORDER, ordered=True)
        if cat_order:
            out[col] = pd.Categorical(out[col], categories=cat_order, ordered=True)
        out = out.sort_values(["model", col]).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Public dataclass and entry point
# ---------------------------------------------------------------------------

@dataclass
class AggregateResult:
    """All group-level summary DataFrames for one matching mode.

    Attributes
    ----------
    mode:
        ``"or"`` or ``"and"``.
    overall:
        One row per model.
    by_split:
        One row per (model, split).
    by_cell_line:
        One row per (model, cell_line).
    by_density:
        One row per (model, density_bin).
    ranked:
        ``overall`` sorted by obj_f1 desc, count_mae asc, pix_dice desc.
    """
    mode:         str
    overall:      pd.DataFrame
    by_split:     pd.DataFrame
    by_cell_line: pd.DataFrame
    by_density:   pd.DataFrame
    ranked:       pd.DataFrame


def aggregate_benchmark(
    per_image_df: pd.DataFrame,
    mode:         str = "or",
) -> AggregateResult:
    """Aggregate a per-image metrics DataFrame into group-level summaries.

    Parameters
    ----------
    per_image_df:
        Output of ``run_benchmark`` for one matching mode.
    mode:
        ``"or"`` or ``"and"`` (informational; not used in computation).

    Returns
    -------
    AggregateResult
    """
    overall    = _agg_overall(per_image_df)
    by_split   = _agg_by_group(per_image_df, "split",       cat_order=SPLIT_ORDER[1:])
    by_cell    = _agg_by_group(per_image_df, "cell_line")
    by_density = _agg_by_group(per_image_df, "density_bin", cat_order=DENSITY_ORDER)

    ranked = overall.sort_values(
        by=["obj_f1", "count_mae", "pix_dice"],
        ascending=[False, True, False],
    ).reset_index(drop=True)

    return AggregateResult(
        mode         = mode,
        overall      = overall,
        by_split     = by_split,
        by_cell_line = by_cell,
        by_density   = by_density,
        ranked       = ranked,
    )


def write_frozen_results(
    agg:     AggregateResult,
    out_dir: Path,
) -> Dict[str, Path]:
    """Write aggregate summary CSVs to the ``frozen_results`` release directory.

    Writes five files inside ``{out_dir}/{mode}_matching/``:
      - ``summary_overall.csv``       — one row per model
      - ``summary_by_split.csv``      — one row per (model, split)
      - ``summary_by_cell_line.csv``  — one row per (model, cell_line)
      - ``summary_by_density_bin.csv``— one row per (model, density_bin)
      - ``ranked_by_obj_f1.csv``      — summary_overall sorted by obj_f1 desc

    The per-image metrics file (``per_image_metrics.csv``) is written by
    ``benchmark.run.run_benchmark`` and is not touched here.

    Returns
    -------
    dict ``{csv_name: path}``
    """
    out_dir = Path(out_dir)
    mode_dir = out_dir / f"{agg.mode}_matching"
    mode_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, Path] = {}

    def _write(df: pd.DataFrame, name: str) -> Path:
        p = mode_dir / name
        df.to_csv(p, index=False)
        logger.info("Wrote %s (%d rows)", p, len(df))
        written[name] = p
        return p

    _write(agg.overall,      "summary_overall.csv")
    _write(agg.by_split,     "summary_by_split.csv")
    _write(agg.by_cell_line, "summary_by_cell_line.csv")
    _write(agg.by_density,   "summary_by_density_bin.csv")
    _write(agg.ranked,       "ranked_by_obj_f1.csv")

    return written