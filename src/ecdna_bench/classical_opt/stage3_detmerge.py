"""
ecdna_bench.classical_opt.stage3_detmerge
==========================================
Stage 3: Bayesian optimisation over detection + merge parameters.

Given the fixed preprocessing combo and parameters from Stage 2, this
module runs BO over:
    threshold_factor, morph_close_kernel, min_area, max_area, merge_distance

The first BO probe is always the Stage-2 baseline evaluation (fixed
detection params from Stage 2), so the Stage-3 result is always at least
as good as the Stage-2 result.

Design rules
------------
* Pure library API — no ``argparse``, no global paths.
* Resumable from partial CSV exactly like Stage 2.
* Output: one per-cell-line JSON file that the ``freeze`` module reads.
* The JSON format is identical to the Stage-2 JSON but adds a
  ``"best_theta"`` key for the detection/merge parameters.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ecdna_bench.classical.preprocess import (
    rgb_to_gray_preserve,
    mask_with_roi,
    apply_chain,
    _unpack_flat_params,
)
from ecdna_bench.classical.detect import (
    detect_objects,
    merge_close_objects,
    label_objects_hsv,
)
from ecdna_bench.evaluation.objects import objects_from_mask
from ecdna_bench.evaluation.matching import match_objects

logger = logging.getLogger(__name__)

__all__ = ["Stage3Config", "run_stage3", "Stage3Result"]

# ---------------------------------------------------------------------------
# Stage-2 baseline detection params (frozen, used as starting point for BO)
# ---------------------------------------------------------------------------
_S2_DETECT = dict(
    threshold_factor   = 1.0,
    morph_close_kernel = 5,
    min_area           = 3,
    max_area           = 900,
)
_S2_MERGE_DIST = 5.0

# Stage-3 matching params (same as Stage 2 internal matching)
_MATCH_MAX_DIST = 20.0
_MATCH_IOU_MIN  = 0.1
_MATCH_ALPHA    = 0.5
_MATCH_POLICY   = "OR"
_MIN_GT_AREA    = 3

# Stage-3 BO search space (detection + merge only)
_PB_DET_MERGE: Dict[str, Tuple[float, float]] = {
    "threshold_factor":   (0.6,   1.6),
    "morph_close_kernel": (1.0,  15.0),
    "min_area":           (1.0,  30.0),
    "max_area":         (100.0, 1000.0),
    "merge_distance":     (0.0,  20.0),
}
_INT_PARAMS = {"morph_close_kernel", "min_area", "max_area"}
_ODD_PARAMS = {"morph_close_kernel"}


def _oddize(v: int) -> int:
    v = max(1, int(round(v)))
    return v if v % 2 == 1 else v + 1


def _sanitize_det_theta(theta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in theta.items():
        fv = float(v)
        if k in _INT_PARAMS:
            iv = int(round(fv))
            if k in _ODD_PARAMS:
                iv = _oddize(iv)
            out[k] = float(max(1, iv))
        else:
            out[k] = fv
    return out


# ---------------------------------------------------------------------------
# Per-image worker
# ---------------------------------------------------------------------------

def _eval_one_image(args: Tuple) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, ignored)."""
    import cv2

    (uid, rgb_path, roi_path, gt_path,
     combo, preproc_param_map, hsv_params,
     det_params, merge_dist, match_params) = args

    try:
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else None
        if gt is None:
            return (0, 0, 0, 0)
        gt_objs = objects_from_mask(gt, min_area=_MIN_GT_AREA)
        n_gt    = len(gt_objs)
    except Exception:
        return (0, 0, 0, 0)

    try:
        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            return (0, 0, n_gt, 0)
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path else None

        rgb_m = mask_with_roi(rgb, roi, mode="zero") if roi is not None else rgb
        gray  = rgb_to_gray_preserve(rgb_m, mode="max")
        enhanced = apply_chain(gray, list(combo), preproc_param_map)

        raw    = detect_objects(enhanced, **det_params)
        merged = merge_close_objects(raw, merge_distance=merge_dist)
        labelled, _ = label_objects_hsv(
            merged, rgb_m,
            white_value_threshold      = hsv_params["white_value_threshold"],
            white_saturation_threshold = hsv_params["white_saturation_threshold"],
        )
        pred_objs = [o for o in labelled if o.get("label") == "ecDNA"]

        res = match_objects(
            pred_objs = pred_objs,
            gt_objs   = gt_objs,
            max_dist  = match_params["max_dist"],
            min_iou   = match_params["min_iou"],
            alpha     = match_params["alpha"],
            policy    = match_params["policy"],
        )
        return (res.tp, res.fp, res.fn, res.ignored_count)
    except Exception:
        return (0, 0, n_gt, 0)


def _eval_split(
    sample_tuples: List[Tuple],
    combo:              Sequence[str],
    preproc_param_map:  Dict[str, Dict[str, Any]],
    hsv_params:         Dict[str, float],
    det_params:         Dict[str, Any],
    merge_dist:         float,
    executor:           ProcessPoolExecutor,
    n_workers:          int,
) -> Dict[str, float]:
    match_params = dict(
        max_dist = _MATCH_MAX_DIST,
        min_iou  = _MATCH_IOU_MIN,
        alpha    = _MATCH_ALPHA,
        policy   = _MATCH_POLICY,
    )
    tasks = [
        (uid, rgb, roi, gt,
         combo, preproc_param_map, hsv_params,
         det_params, merge_dist, match_params)
        for uid, rgb, roi, gt in sample_tuples
    ]
    n  = len(tasks)
    cs = max(1, min(32, n // max(1, 12 * n_workers)))
    tp = fp = fn = ign = 0
    for (tpi, fpi, fni, igni) in executor.map(_eval_one_image, tasks, chunksize=cs):
        tp += tpi; fp += fpi; fn += fni; ign += igni

    denom = 2 * tp + fp + fn
    f1    = (2 * tp / denom) if denom > 0 else 0.0
    prec  = tp / (tp + fp)   if (tp + fp) > 0 else 0.0
    rec   = tp / (tp + fn)   if (tp + fn) > 0 else 0.0
    return dict(tp=tp, fp=fp, fn=fn, ignored_pred=ign,
                precision=prec, recall=rec, f1=f1, n_images=n)


# ---------------------------------------------------------------------------
# Config + public API
# ---------------------------------------------------------------------------

@dataclass
class Stage3Config:
    """Parameters for one Stage-3 run."""
    out_dir:      Path
    partials_dir: Path
    json_dir:     Path
    init_points:  int  = 10
    n_iter:       int  = 40
    force:        bool = False
    max_workers:  int  = max(1, (os.cpu_count() or 1))


@dataclass
class Stage3Result:
    """Return value of ``run_stage3``."""
    cell_line:            str
    combo:                Tuple[str, ...]
    best_theta:           Dict[str, float]   # {threshold_factor, morph_close_kernel, ...}
    fixed_preproc_params: Dict[str, Any]     # flat format
    fixed_hsv_params:     Dict[str, float]
    best_train_f1:        float
    best_val_f1:          float
    best_test_f1:         float
    history_csv:          Path
    json_path:            Path


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))


def run_stage3(
    train_samples:        List[Any],
    val_samples:          List[Any],
    test_samples:         List[Any],
    cell_line:            str,
    combo:                Sequence[str],
    fixed_preproc_params: Dict[str, Any],   # flat, e.g. {"clahe__clipLimit": 0.4}
    fixed_hsv_params:     Dict[str, float],
    cfg:                  Stage3Config,
) -> Stage3Result:
    """Run Stage-3 detection+merge BO for one (cell_line, combo) pair.

    Parameters
    ----------
    train_samples, val_samples, test_samples:
        Sample objects for the respective splits.
        BO objective is micro-F1 on **train**; best probe selected by
        **val_f1** (Decision D2); **test_f1** reported as held-out result.
    cell_line:
        Cell-line name.
    combo:
        Preprocessing order (from Stage 1).
    fixed_preproc_params:
        Flat preprocessing params from Stage 2.
    fixed_hsv_params:
        HSV thresholds from Stage 2.
    cfg:
        ``Stage3Config``.

    Returns
    -------
    Stage3Result
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as e:
        raise ImportError(
            "bayesian-optimization is required for Stage 3. "
            "Install it with: pip install bayesian-optimization"
        ) from e

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.partials_dir.mkdir(parents=True, exist_ok=True)
    cfg.json_dir.mkdir(parents=True, exist_ok=True)

    combo = tuple(combo)
    cid   = "__".join(combo)
    safe  = _safe_name(cell_line)

    cl_partial_dir = cfg.partials_dir / safe
    cl_partial_dir.mkdir(parents=True, exist_ok=True)
    partial_csv = cl_partial_dir / f"{safe}__{cid}__stage3_partial.csv"
    json_path   = cfg.json_dir / f"{safe}.json"

    # Unpack preproc params for apply_chain
    preproc_param_map = _unpack_flat_params(fixed_preproc_params)

    def _to_tuples(samples: List[Any]) -> List[Tuple]:
        return [
            (
                str(s.uid),
                str(s.rgb_path),
                str(s.roi_path)     if getattr(s, "roi_path",     None) is not None else "",
                str(s.gt_mask_path) if getattr(s, "gt_mask_path", None) is not None else "",
            )
            for s in samples
        ]

    train_tuples = _to_tuples(train_samples)
    val_tuples   = _to_tuples(val_samples)
    test_tuples  = _to_tuples(test_samples)

    total_expected = cfg.init_points + cfg.n_iter

    # --- Check resume ---
    history_rows: List[Dict[str, Any]] = []
    if partial_csv.exists() and not cfg.force:
        try:
            df_partial = pd.read_csv(partial_csv)
            if len(df_partial) >= total_expected:
                logger.info("Stage-3 | %s %s already complete (%d rows).",
                            cell_line, cid, len(df_partial))
                sel_col  = "val_f1" if "val_f1" in df_partial.columns else "test_f1"
                best_row  = df_partial.loc[df_partial[sel_col].astype(float).idxmax()]
                theta     = {k: float(best_row[k]) for k in _PB_DET_MERGE if k in best_row}
                theta     = _sanitize_det_theta(theta)
                return _make_result(
                    cell_line, combo, theta, fixed_preproc_params,
                    fixed_hsv_params, best_row, partial_csv, json_path,
                    preproc_param_map, cid,
                )
            history_rows = df_partial.to_dict("records")
            logger.info("Stage-3 | %s %s: resuming from %d rows.",
                        cell_line, cid, len(history_rows))
        except Exception as exc:
            logger.warning("Stage-3 | Could not load partial: %s", exc)
            history_rows = []

    n_workers = cfg.max_workers

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        def _objective(**theta_raw: float) -> float:
            theta   = _sanitize_det_theta(theta_raw)
            det_p   = {
                "threshold_factor":   theta["threshold_factor"],
                "morph_close_kernel": int(theta["morph_close_kernel"]),
                "min_area":           int(theta["min_area"]),
                "max_area":           int(theta["max_area"]),
            }
            m       = _eval_split(train_tuples, combo, preproc_param_map,
                                  fixed_hsv_params, det_p,
                                  theta["merge_distance"], executor, n_workers)
            val_m   = _eval_split(val_tuples,   combo, preproc_param_map,
                                  fixed_hsv_params, det_p,
                                  theta["merge_distance"], executor, n_workers)
            test_m  = _eval_split(test_tuples,  combo, preproc_param_map,
                                  fixed_hsv_params, det_p,
                                  theta["merge_distance"], executor, n_workers)

            row = {f"train_{k}": v for k, v in m.items()}
            row.update({f"val_{k}":  v for k, v in val_m.items()})
            row.update({f"test_{k}": v for k, v in test_m.items()})
            row.update({"cell_line": cell_line, "combo_id": cid})
            row.update({k: theta[k] for k in _PB_DET_MERGE})
            history_rows.append(row)
            tmp = str(partial_csv) + ".tmp"
            pd.DataFrame(history_rows).to_csv(tmp, index=False)
            os.replace(tmp, str(partial_csv))
            return float(m["f1"])

        optimizer = BayesianOptimization(
            f            = _objective,
            pbounds      = _PB_DET_MERGE,
            verbose      = 0,
            random_state = 42,
        )

        # Seed from existing history
        already_done = len(history_rows)
        for row in history_rows:
            theta_row = {k: float(row[k]) for k in _PB_DET_MERGE if k in row and pd.notna(row[k])}
            if theta_row:
                optimizer.register(params=theta_row,
                                   target=float(row.get("train_f1", 0.0)))

        # Always evaluate the Stage-2 baseline first (if not already in history)
        if already_done == 0:
            baseline_theta = {
                "threshold_factor":   float(_S2_DETECT["threshold_factor"]),
                "morph_close_kernel": float(_S2_DETECT["morph_close_kernel"]),
                "min_area":           float(_S2_DETECT["min_area"]),
                "max_area":           float(_S2_DETECT["max_area"]),
                "merge_distance":     _S2_MERGE_DIST,
            }
            try:
                optimizer.probe(params=baseline_theta, lazy=False)
            except Exception as exc:
                logger.warning("Stage-3 | baseline probe failed: %s", exc)

        remaining_init = max(0, cfg.init_points - max(0, already_done - 1))
        remaining_iter = max(0, cfg.n_iter - max(0, already_done - 1 - cfg.init_points))
        logger.info("Stage-3 | %s %s | init=%d iter=%d (from %d existing)",
                    cell_line, cid, remaining_init, remaining_iter, already_done)

        if remaining_init + remaining_iter > 0:
            optimizer.maximize(init_points=remaining_init, n_iter=remaining_iter)

    df_h    = pd.DataFrame(history_rows)
    sel_col = "val_f1" if "val_f1" in df_h.columns else "test_f1"
    best_r  = df_h.loc[df_h[sel_col].astype(float).idxmax()]
    theta   = {k: float(best_r[k]) for k in _PB_DET_MERGE if k in best_r}
    theta   = _sanitize_det_theta(theta)

    return _make_result(
        cell_line, combo, theta, fixed_preproc_params,
        fixed_hsv_params, best_r, partial_csv, json_path,
        preproc_param_map, cid,
    )


def _make_result(
    cell_line:            str,
    combo:                Tuple[str, ...],
    theta:                Dict[str, float],
    fixed_preproc_params: Dict[str, Any],
    fixed_hsv_params:     Dict[str, float],
    best_row:             Any,
    partial_csv:          Path,
    json_path:            Path,
    preproc_param_map:    Dict[str, Dict[str, Any]],
    cid:                  str,
) -> Stage3Result:
    payload = {
        cid: {
            "combo":                list(combo),
            "fixed_preproc_params": fixed_preproc_params,
            "fixed_hsv_params":     fixed_hsv_params,
            "best_theta":           theta,
            "best_train_f1":        float(best_row.get("train_f1", 0.0)),
            "best_val_f1":          float(best_row.get("val_f1",   0.0)),
            "best_test_f1":         float(best_row.get("test_f1",  0.0)),
        }
    }
    _write_json_atomic(json_path, payload)

    return Stage3Result(
        cell_line            = cell_line,
        combo                = combo,
        best_theta           = theta,
        fixed_preproc_params = fixed_preproc_params,
        fixed_hsv_params     = fixed_hsv_params,
        best_train_f1        = float(best_row.get("train_f1", 0.0)),
        best_val_f1          = float(best_row.get("val_f1",   0.0)),
        best_test_f1         = float(best_row.get("test_f1",  0.0)),
        history_csv          = partial_csv,
        json_path            = json_path,
    )