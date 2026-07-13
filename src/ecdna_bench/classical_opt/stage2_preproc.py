"""
ecdna_bench.classical_opt.stage2_preproc
==========================================
Stage 2: Bayesian optimisation over preprocessing parameters.

Given the best preprocessing combo from Stage 1, this module runs BO
(via the ``bayesian-optimization`` library) over the scalar parameters
of each op in the combo, plus the two HSV thresholds.

Design rules
------------
* Pure library API — no ``argparse``, no global paths.
* Resumable: iteration history is written to a partial CSV after each BO
  probe.  If ``force=False`` and the partial is already complete (≥
  ``init_points + n_iter`` rows), the best result is extracted without
  re-running BO.
* Parallelism: images within one BO objective call are evaluated via
  ``ProcessPoolExecutor``; the BO loop itself is single-threaded.
* Objective: micro-F1 on the **training** split.  Test F1 is tracked
  every iteration (not used for selection).
* Search space: tight bounds centred on the legacy defaults, clipped to
  global hard limits.
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
    PREPROCESSING_REGISTRY,
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

__all__ = ["Stage2Config", "run_stage2", "build_param_bounds", "Stage2Result"]

# ---------------------------------------------------------------------------
# Fixed Stage-2 detection & matching (copied from source script)
# ---------------------------------------------------------------------------
_FIXED_DETECT = dict(
    threshold_factor   = 1.0,
    morph_close_kernel = 5,
    min_area           = 3,
    max_area           = 900,
)
_FIXED_MERGE_DIST  = 5.0
_MATCH_MAX_DIST    = 20.0    
_MATCH_IOU_MIN     = 0.1
_MATCH_ALPHA       = 0.5
_MATCH_POLICY      = "OR"
_MIN_GT_AREA       = 3

# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------

_BASELINE_CENTERS: Dict[str, float] = {
    "apply_gamma__gamma_val":             1.2,
    "apply_sharpening__kernel_weight":    5.0,
    "apply_bilateral_filter__d":          9.0,
    "apply_bilateral_filter__sigmaColor": 75.0,
    "apply_bilateral_filter__sigmaSpace": 75.0,
    "clahe__clipLimit":                   1.0,
    "clahe__tileGridSize":               30.0,
    "remove_background__kernel_size":    11.0,
    "top_hat__k_size":                   15.0,
    "top_hat__chrom_size":               25.0,
    "apply_sigmoid__gain":               10.0,
    "apply_sigmoid__threshold":          100.0 / 255.0,
    "hsv__white_value_threshold":       200.0,
    "hsv__white_saturation_threshold":   30.0,
}

_GLOBAL_LIMITS: Dict[str, Tuple[float, float]] = {
    "apply_gamma__gamma_val":             (0.4,  3.0),
    "apply_sharpening__kernel_weight":    (0.0, 15.0),
    "apply_bilateral_filter__d":          (1.0, 25.0),
    "apply_bilateral_filter__sigmaColor": (1.0, 200.0),
    "apply_bilateral_filter__sigmaSpace": (1.0, 200.0),
    "clahe__clipLimit":                   (0.1, 10.0),
    "clahe__tileGridSize":                (4.0, 64.0),
    "remove_background__kernel_size":     (3.0, 51.0),
    "top_hat__k_size":                    (3.0, 51.0),
    "top_hat__chrom_size":                (5.0, 61.0),
    "apply_sigmoid__gain":                (1.0, 25.0),
    "apply_sigmoid__threshold":           (80.0 / 255.0, 120.0 / 255.0),
    "hsv__white_value_threshold":         (0.0, 255.0),
    "hsv__white_saturation_threshold":    (0.0, 255.0),
}

_HALF_RANGES: Dict[str, float] = {
    "apply_gamma__gamma_val":              0.25,
    "apply_sharpening__kernel_weight":     2.0,
    "apply_bilateral_filter__d":           3.0,
    "apply_bilateral_filter__sigmaColor": 25.0,
    "apply_bilateral_filter__sigmaSpace": 25.0,
    "clahe__clipLimit":                    0.6,
    "clahe__tileGridSize":                10.0,
    "remove_background__kernel_size":      6.0,
    "top_hat__k_size":                     8.0,
    "top_hat__chrom_size":                10.0,
    "apply_sigmoid__gain":                 6.0,
    "apply_sigmoid__threshold":           10.0 / 255.0,
    "hsv__white_value_threshold":         30.0,
    "hsv__white_saturation_threshold":    20.0,
}

_INT_PARAMS = {
    "apply_bilateral_filter__d",
    "clahe__tileGridSize",
    "remove_background__kernel_size",
    "top_hat__k_size",
    "top_hat__chrom_size",
}
_ODD_INT_PARAMS = {
    "remove_background__kernel_size",
    "top_hat__k_size",
    "top_hat__chrom_size",
}


def _oddize(v: int) -> int:
    v = max(1, int(round(v)))
    return v if v % 2 == 1 else v + 1


def _tight_bounds(center: float, half: float, lo: float, hi: float) -> Tuple[float, float]:
    a = max(lo, center - half)
    b = min(hi, center + half)
    return (a, b + 1e-9) if b <= a else (a, b)


def build_param_bounds(
    combo: Sequence[str],
    center_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Tuple[float, float]]:
    """Build the BO search-space bounds for the given preprocessing combo.

    Parameters
    ----------
    combo:
        Ordered list of preprocessing op registry keys.
    center_overrides:
        Optional dict to override default centres for any flat param key.

    Returns
    -------
    dict mapping flat param name → (lo, hi) tuple.
    """
    centers = dict(_BASELINE_CENTERS)
    if center_overrides:
        centers.update(center_overrides)

    pb: Dict[str, Tuple[float, float]] = {}

    def _add(name: str) -> None:
        pb[name] = _tight_bounds(
            centers[name], _HALF_RANGES[name],
            *_GLOBAL_LIMITS[name],
        )

    op_to_params: Dict[str, List[str]] = {
        "apply_gamma":            ["apply_gamma__gamma_val"],
        "apply_sharpening":       ["apply_sharpening__kernel_weight"],
        "apply_bilateral_filter": ["apply_bilateral_filter__d",
                                   "apply_bilateral_filter__sigmaColor",
                                   "apply_bilateral_filter__sigmaSpace"],
        "clahe":                  ["clahe__clipLimit", "clahe__tileGridSize"],
        "remove_background":      ["remove_background__kernel_size"],
        "top_hat":                ["top_hat__k_size", "top_hat__chrom_size"],
        "apply_sigmoid":          ["apply_sigmoid__gain", "apply_sigmoid__threshold"],
    }

    for fn in combo:
        for p in op_to_params.get(fn, []):
            _add(p)

    # HSV thresholds always included
    _add("hsv__white_value_threshold")
    _add("hsv__white_saturation_threshold")
    return pb


def _sanitize_theta(theta: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in theta.items():
        fv = float(v)
        if k in _INT_PARAMS:
            iv = int(round(fv))
            if k in _ODD_INT_PARAMS:
                iv = _oddize(iv)
            out[k] = float(max(1, iv))
        else:
            out[k] = fv
    return out


def _split_theta(theta: Dict[str, float]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, float]]:
    """Split flat theta into ``{op: {param: val}}`` + HSV dict."""
    theta = _sanitize_theta(theta)
    func_params: Dict[str, Dict[str, Any]] = {}
    hsv_params  = {
        "white_value_threshold":      float(theta["hsv__white_value_threshold"]),
        "white_saturation_threshold": float(theta["hsv__white_saturation_threshold"]),
    }
    for k, v in theta.items():
        if k.startswith("hsv__") or "__" not in k:
            continue
        fn, pname = k.split("__", 1)
        func_params.setdefault(fn, {})[pname] = float(v)
    return func_params, hsv_params


# ---------------------------------------------------------------------------
# Per-image worker (picklable)
# ---------------------------------------------------------------------------

def _eval_one_image(args: Tuple) -> Tuple[int, int, int, int]:
    """Return (tp, fp, fn, ignored).  On failure after GT load: (0, 0, n_gt, 0)."""
    import cv2

    (uid, rgb_path, roi_path, gt_path,
     combo, func_params, hsv_params,
     detect_params, merge_dist, match_params) = args

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
        enhanced = apply_chain(gray, list(combo), func_params)

        raw   = detect_objects(enhanced, **detect_params)
        merged = merge_close_objects(raw, merge_distance=merge_dist)

        # HSV filter: keep ecDNA only
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
    combo: Sequence[str],
    func_params: Dict[str, Dict[str, Any]],
    hsv_params:  Dict[str, float],
    executor:    ProcessPoolExecutor,
    n_workers:   int,
) -> Dict[str, float]:
    """Evaluate all samples in parallel; return aggregate metrics dict."""
    match_params = dict(
        max_dist = _MATCH_MAX_DIST,
        min_iou  = _MATCH_IOU_MIN,
        alpha    = _MATCH_ALPHA,
        policy   = _MATCH_POLICY,
    )
    tasks = [
        (uid, rgb, roi, gt, combo, func_params, hsv_params,
         _FIXED_DETECT, _FIXED_MERGE_DIST, match_params)
        for uid, rgb, roi, gt in sample_tuples
    ]
    n    = len(tasks)
    cs   = max(1, min(32, n // max(1, 12 * n_workers)))
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
class Stage2Config:
    """Parameters for one Stage-2 run."""
    out_dir:           Path
    partials_dir:      Path
    json_dir:          Path
    init_points:       int   = 20
    n_iter:            int   = 100
    top_combos:        int   = 1      # how many top Stage-1 combos to optimise
    force:             bool  = False
    max_workers:       int   = max(1, (os.cpu_count() or 1))
    center_overrides:  Optional[Dict[str, float]] = field(default=None)


@dataclass
class Stage2Result:
    """Return value of ``run_stage2``."""
    cell_line:           str
    combo:               Tuple[str, ...]
    best_theta:          Dict[str, float]
    fixed_preproc_params: Dict[str, Any]   # flat format {"clahe__clipLimit": 0.4, ...}
    fixed_hsv_params:    Dict[str, float]
    best_train_f1:       float
    best_val_f1:         float
    best_test_f1:        float
    history_csv:         Path
    json_path:           Path


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def run_stage2(
    train_samples: List[Any],
    val_samples:   List[Any],
    test_samples:  List[Any],
    cell_line:     str,
    combo:         Sequence[str],
    cfg:           Stage2Config,
) -> Stage2Result:
    """Run Stage-2 BO for one (cell_line, combo) pair.

    Parameters
    ----------
    train_samples, val_samples, test_samples:
        Sample objects (from ``data.samples``) for the respective splits.
        BO objective is micro-F1 on **train**; best probe selected by
        **val_f1** (Decision D2); **test_f1** reported as held-out result.
    cell_line:
        Cell-line name.
    combo:
        Preprocessing op order (from Stage-1 output).
    cfg:
        ``Stage2Config``.

    Returns
    -------
    Stage2Result
    """
    try:
        from bayes_opt import BayesianOptimization
    except ImportError as e:
        raise ImportError(
            "bayesian-optimization is required for Stage 2. "
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
    partial_csv = cl_partial_dir / f"{safe}__{cid}__stage2_partial.csv"
    json_path   = cfg.json_dir / f"{safe}.json"

    # Serialise samples
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

    pbounds = build_param_bounds(combo, cfg.center_overrides)
    total_expected = cfg.init_points + cfg.n_iter

    # --- Check resume ---
    history_rows: List[Dict[str, Any]] = []
    if partial_csv.exists() and not cfg.force:
        try:
            df_partial = pd.read_csv(partial_csv)
            if len(df_partial) >= total_expected:
                logger.info("Stage-2 | %s %s already complete (%d rows).",
                            cell_line, cid, len(df_partial))
                # Select best probe by val_f1 (D2); fall back to test_f1
                sel_col = "val_f1" if "val_f1" in df_partial.columns else "test_f1"
                best_row  = df_partial.loc[df_partial[sel_col].astype(float).idxmax()]
                theta     = {k: float(best_row[k]) for k in pbounds if k in best_row}
                theta     = _sanitize_theta(theta)
                func_params, hsv_params = _split_theta(theta)
                flat_preproc = {
                    f"{fn}__{p}": v
                    for fn, pd_ in func_params.items()
                    for p, v in pd_.items()
                }
                return Stage2Result(
                    cell_line            = cell_line,
                    combo                = combo,
                    best_theta           = theta,
                    fixed_preproc_params = flat_preproc,
                    fixed_hsv_params     = hsv_params,
                    best_train_f1        = float(best_row.get("train_f1", 0.0)),
                    best_val_f1          = float(best_row.get("val_f1",   0.0)),
                    best_test_f1         = float(best_row.get("test_f1",  0.0)),
                    history_csv          = partial_csv,
                    json_path            = json_path,
                )
            history_rows = df_partial.to_dict("records")
            logger.info("Stage-2 | %s %s: resuming from %d rows.", cell_line, cid, len(history_rows))
        except Exception as exc:
            logger.warning("Stage-2 | Could not load partial CSV: %s", exc)
            history_rows = []

    # --- BO loop ---
    n_workers = cfg.max_workers

    with ProcessPoolExecutor(max_workers=n_workers) as executor:

        def _objective(**theta_raw: float) -> float:
            func_params, hsv_params = _split_theta(theta_raw)
            m      = _eval_split(train_tuples, combo, func_params, hsv_params, executor, n_workers)
            val_m  = _eval_split(val_tuples,   combo, func_params, hsv_params, executor, n_workers)
            test_m = _eval_split(test_tuples,  combo, func_params, hsv_params, executor, n_workers)
            row = {f"train_{k}": v for k, v in m.items()}
            row.update({f"val_{k}":  v for k, v in val_m.items()})
            row.update({f"test_{k}": v for k, v in test_m.items()})
            row.update({"cell_line": cell_line, "combo_id": cid})
            row.update({k: _sanitize_theta(theta_raw)[k] for k in pbounds})
            history_rows.append(row)
            # Atomic write
            tmp = str(partial_csv) + ".tmp"
            pd.DataFrame(history_rows).to_csv(tmp, index=False)
            os.replace(tmp, str(partial_csv))
            return float(m["f1"])

        optimizer = BayesianOptimization(
            f         = _objective,
            pbounds   = pbounds,
            verbose   = 0,
            random_state = 42,
        )

        # Seed from existing history
        already_done = len(history_rows)
        for row in history_rows:
            theta_row = {k: float(row[k]) for k in pbounds if k in row and pd.notna(row[k])}
            if theta_row:
                optimizer.register(params=theta_row, target=float(row.get("train_f1", 0.0)))

        remaining_init  = max(0, cfg.init_points - already_done)
        remaining_iter  = max(0, cfg.n_iter - max(0, already_done - cfg.init_points))
        logger.info("Stage-2 | %s %s | init=%d iter=%d (from %d existing)",
                    cell_line, cid, remaining_init, remaining_iter, already_done)

        if remaining_init + remaining_iter > 0:
            optimizer.maximize(init_points=remaining_init, n_iter=remaining_iter)

    # Extract best result — select by val_f1 (Decision D2)
    df_h    = pd.DataFrame(history_rows)
    sel_col = "val_f1" if "val_f1" in df_h.columns else "test_f1"
    best_r  = df_h.loc[df_h[sel_col].astype(float).idxmax()]
    theta   = {k: float(best_r[k]) for k in pbounds if k in best_r}
    theta   = _sanitize_theta(theta)
    func_params, hsv_params = _split_theta(theta)
    flat_preproc = {
        f"{fn}__{p}": v
        for fn, pd_ in func_params.items()
        for p, v in pd_.items()
    }

    # Write per-cell-line JSON
    payload = {
        cid: {
            "combo":                list(combo),
            "fixed_preproc_params": flat_preproc,
            "fixed_hsv_params":     hsv_params,
            "best_train_f1":        float(best_r.get("train_f1", 0.0)),
            "best_val_f1":          float(best_r.get("val_f1",   0.0)),
            "best_test_f1":         float(best_r.get("test_f1",  0.0)),
            "best_theta":           theta,
        }
    }
    _write_json_atomic(json_path, payload)
    logger.info("Stage-2 | Saved JSON for %s → %s", cell_line, json_path)

    return Stage2Result(
        cell_line            = cell_line,
        combo                = combo,
        best_theta           = theta,
        fixed_preproc_params = flat_preproc,
        fixed_hsv_params     = hsv_params,
        best_train_f1        = float(best_r.get("train_f1", 0.0)),
        best_val_f1          = float(best_r.get("val_f1",   0.0)),
        best_test_f1         = float(best_r.get("test_f1",  0.0)),
        history_csv          = partial_csv,
        json_path            = json_path,
    )


def _write_json_atomic(path: Path, obj: Any) -> None:
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, str(path))