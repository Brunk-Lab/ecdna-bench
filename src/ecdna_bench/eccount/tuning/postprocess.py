"""
ecdna_bench.eccount.tuning.postprocess
========================================
Grid search over post-processing hyperparameters.

108-config grid (paper §10 / §13d):
    smooth_sigma      ∈ {0.0, 0.5, 1.0}
    threshold_abs     ∈ {0.25, 0.35, 0.45}
    peak_min_distance ∈ {1, 2, 3, 4}

3 × 3 × 4 = 36 configs × 3 nms_min_distance values ∈ {1, 2, 3} = 108 total.

Frozen paper values:
    smooth_sigma=0.5, threshold_abs=0.35, peak_min_distance=2, nms_min_distance=2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ecdna_bench.eccount.postprocess import PostprocessConfig, detect_points_from_map

__all__ = ["POSTPROCESS_GRID", "PostprocessRunResult", "sweep_postprocess_grid",
           "aggregate_postprocess_results"]

# Grid axes
_SMOOTH_SIGMAS:      List[float] = [0.0, 0.5, 1.0]
_THRESHOLD_ABS:      List[float] = [0.25, 0.35, 0.45]
_PEAK_MIN_DISTANCES: List[int]   = [1, 2, 3, 4]
_NMS_MIN_DISTANCES:  List[int]   = [1, 2, 3]

# Frozen paper values
_PAPER = dict(smooth_sigma=0.5, threshold_abs=0.35, peak_min_distance=2, nms_min_distance=2)


def _build_grid() -> List[Dict]:
    configs = []
    for sm in _SMOOTH_SIGMAS:
        for thr in _THRESHOLD_ABS:
            for pmd in _PEAK_MIN_DISTANCES:
                for nmd in _NMS_MIN_DISTANCES:
                    configs.append(dict(
                        smooth_sigma=sm, threshold_abs=thr,
                        peak_min_distance=pmd, nms_min_distance=nmd,
                    ))
    return configs


POSTPROCESS_GRID: List[Dict] = _build_grid()
assert len(POSTPROCESS_GRID) == 108, f"Expected 108 configs, got {len(POSTPROCESS_GRID)}"


@dataclass
class PostprocessRunResult:
    """Result of one post-processing config evaluated on a set of images."""
    smooth_sigma:       float
    threshold_abs:      float
    peak_min_distance:  int
    nms_min_distance:   int
    # Aggregate metrics over all evaluated images
    mean_pred_count:    float
    mean_gt_count:      float
    mae:                float
    f1:                 Optional[float]  = None   # object-level F1 if GS available
    extra:              Optional[dict]   = None


def sweep_postprocess_grid(
    prob_maps:       List,   # list of np.ndarray (H, W)
    roi_masks:       Optional[List] = None,
    gt_counts:       Optional[List[int]] = None,
    configs:         Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Evaluate all postprocess configs on a list of probability maps.

    Parameters
    ----------
    prob_maps:
        List of (H, W) float32 probability maps (one per image).
    roi_masks:
        Optional list of (H, W) binary masks (same length as prob_maps).
    gt_counts:
        Optional list of gold-standard counts for MAE computation.
    configs:
        List of config dicts.  Defaults to ``POSTPROCESS_GRID`` (108 configs).

    Returns
    -------
    pd.DataFrame
        One row per config, sorted by MAE ascending.
    """
    import numpy as np

    configs = configs or POSTPROCESS_GRID
    n       = len(prob_maps)

    rows = []
    for cfg_dict in configs:
        pp_cfg = PostprocessConfig(
            smooth_sigma      = cfg_dict["smooth_sigma"],
            threshold_abs     = cfg_dict["threshold_abs"],
            peak_min_distance = cfg_dict["peak_min_distance"],
            nms_min_distance  = cfg_dict["nms_min_distance"],
        )

        pred_counts = []
        for i, pm in enumerate(prob_maps):
            roi = roi_masks[i] if roi_masks else None
            peaks = detect_points_from_map(pm, roi_mask=roi, config=pp_cfg)
            pred_counts.append(len(peaks))

        mean_pred = float(np.mean(pred_counts))
        if gt_counts:
            mae = float(np.mean(np.abs(np.array(pred_counts) - np.array(gt_counts))))
            mean_gt = float(np.mean(gt_counts))
        else:
            mae     = float("nan")
            mean_gt = float("nan")

        rows.append({
            "smooth_sigma":      cfg_dict["smooth_sigma"],
            "threshold_abs":     cfg_dict["threshold_abs"],
            "peak_min_distance": cfg_dict["peak_min_distance"],
            "nms_min_distance":  cfg_dict["nms_min_distance"],
            "mean_pred_count":   mean_pred,
            "mean_gt_count":     mean_gt,
            "mae":               mae,
            "is_paper_value":    all(
                cfg_dict[k] == _PAPER[k] for k in _PAPER
            ),
        })

    df = pd.DataFrame(rows)
    if not df.empty and "mae" in df.columns:
        df = df.sort_values("mae").reset_index(drop=True)
    return df


def aggregate_postprocess_results(results: List[PostprocessRunResult]) -> pd.DataFrame:
    """Aggregate a list of ``PostprocessRunResult`` into a DataFrame."""
    rows = [
        {
            "smooth_sigma":      r.smooth_sigma,
            "threshold_abs":     r.threshold_abs,
            "peak_min_distance": r.peak_min_distance,
            "nms_min_distance":  r.nms_min_distance,
            "mean_pred_count":   r.mean_pred_count,
            "mean_gt_count":     r.mean_gt_count,
            "mae":               r.mae,
            "f1":                r.f1 if r.f1 is not None else float("nan"),
            "is_paper_value":    all([
                r.smooth_sigma == _PAPER["smooth_sigma"],
                r.threshold_abs == _PAPER["threshold_abs"],
                r.peak_min_distance == _PAPER["peak_min_distance"],
                r.nms_min_distance == _PAPER["nms_min_distance"],
            ]),
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("mae").reset_index(drop=True)
    return df
