"""
ecdna_bench.benchmark.run
==========================
Per-image benchmark evaluation across all models × OR/AND matching modes.

Design rules
------------
* ``run_benchmark`` is the single public entry point.  It is a pure library
  function: no argparse, no global state, no disk I/O except writing the
  per-image CSV outputs and an errors CSV.
* The evaluation logic (CC extraction, pairwise precompute, Hungarian
  matching, pixel metrics) lives in ``ecdna_bench.evaluation.*``.  This
  module is an orchestrator, not a reimplementation.
* Both OR and AND modes are always evaluated in a single pass per image
  (pairwise matrices are shared, so the second mode is essentially free).
* Parallelism: ``ProcessPoolExecutor`` dispatches one task per
  (image, model) pair.  The worker function ``_eval_one`` is module-level
  and picklable.
* Idempotent: if ``{out_dir}/{mode}_matching/per_image_metrics.csv`` already
  exists and has the right number of rows, the run is skipped (unless
  ``force=True``).
* Frozen eval params (from §2 of REWRITE_PLAN.md):
    d_max=20, iou_min=0.1, alpha=0.5, min_area=3.
"""

from __future__ import annotations

import logging
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ecdna_bench.benchmark.registry import MODEL_ORDER, MODEL_REGISTRY
from ecdna_bench.evaluation.matching import (
    precompute_pairwise,
    resolve_matching_from_pairwise,
)
from ecdna_bench.evaluation.metrics import object_metrics_from_counts

logger = logging.getLogger(__name__)

__all__ = ["EvalConfig", "run_benchmark"]


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EvalConfig:
    """Evaluation hyperparameters for the benchmark.

    All defaults are the frozen paper values (§2 of REWRITE_PLAN.md).
    """
    d_max:           float = 20.0   # frozen
    iou_min:         float = 0.1    # frozen
    alpha:           float = 0.5    # frozen
    min_area:        int   = 3      # frozen CC filter
    pixel_threshold: float = 0.5    # binarization threshold for pixel metrics
    modes:           List[str] = field(default_factory=lambda: ["or", "and"])
    n_workers:       int   = 8
    force:           bool  = False


# ---------------------------------------------------------------------------
# Pure worker function (module-level → picklable)
# ---------------------------------------------------------------------------

def _assign_density_bin(gt_count: int) -> str:
    """Assign one of the frozen density bins."""
    if gt_count < 10:   return "0-9"
    if gt_count < 50:   return "10-49"
    if gt_count < 150:  return "50-149"
    if gt_count < 300:  return "150-299"
    return "300+"


def _load_gray(path) -> Optional[np.ndarray]:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    return img


def _extract_objects(mask: np.ndarray, min_area: int) -> List[Dict[str, Any]]:
    """Extract connected components as object dicts (centroid, bbox, area, mask)."""
    from skimage.measure import label, regionprops
    binary  = (mask > 0).astype(np.uint8)
    labeled = label(binary, connectivity=2)
    objs    = []
    for p in regionprops(labeled):
        if p.area < min_area:
            continue
        y1, x1, y2, x2 = p.bbox
        objs.append({
            "centroid": p.centroid,
            "bbox":     (int(y1), int(x1), int(y2), int(x2)),
            "area":     int(p.area),
            "mask":     (labeled == p.label).astype(np.uint8),
        })
    return objs


def _pixel_confusion(gt_mask: np.ndarray, pred_mask: np.ndarray,
                     threshold: float = 0.5) -> Dict[str, int]:
    gt_bin   = (gt_mask   > (threshold * 255)).astype(bool)
    pred_bin = (pred_mask > (threshold * 255)).astype(bool)
    return {
        "tp": int((gt_bin  &  pred_bin).sum()),
        "fp": int((~gt_bin &  pred_bin).sum()),
        "fn": int((gt_bin  & ~pred_bin).sum()),
        "tn": int((~gt_bin & ~pred_bin).sum()),
    }


def _pixel_metrics(conf: Dict[str, int]) -> Dict[str, float]:
    tp, fp, fn, tn = conf["tp"], conf["fp"], conf["fn"], conf["tn"]
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    iou  = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "iou": iou, "dice": dice}


def _eval_one(args: tuple) -> Dict[str, Any]:
    """Worker: evaluate one (uid, model) pair under both matching modes.

    Returns ``{mode: row_dict}``.
    Raises on GT load failure; returns all-zero prediction on missing pred mask.
    """
    (uid, split, cell_line, gt_path_str, pred_path_str,
     model_name, d_max, iou_min, alpha, min_area, pixel_thr) = args

    gt_path = Path(gt_path_str)
    if not gt_path.is_file():
        raise FileNotFoundError(f"GT mask not found: {gt_path}")

    gt_mask = _load_gray(gt_path)
    if gt_mask is None:
        raise RuntimeError(f"Cannot read GT mask: {gt_path}")

    pred_path = Path(pred_path_str) if pred_path_str else None
    pred_exists = 0
    if pred_path and pred_path.is_file():
        pred_mask = _load_gray(pred_path)
        pred_exists = 1 if pred_mask is not None else 0
    else:
        pred_mask = None

    if pred_mask is None:
        pred_mask = np.zeros_like(gt_mask)

    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Shape mismatch uid={uid} model={model_name}: "
            f"GT={gt_mask.shape} PRED={pred_mask.shape}"
        )

    gt_objs   = _extract_objects(gt_mask,   min_area)
    pred_objs = _extract_objects(pred_mask, min_area)

    # Precompute pairwise geometry once; resolve for each mode cheaply.
    pw = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=d_max)
    pix_conf = _pixel_confusion(gt_mask, pred_mask, pixel_thr)
    pix_m    = _pixel_metrics(pix_conf)

    gt_count   = len(gt_objs)
    pred_count = len(pred_objs)
    abs_err    = abs(pred_count - gt_count)
    ape        = float("nan") if gt_count == 0 else 100.0 * abs_err / gt_count

    out: Dict[str, Any] = {}
    for mode in ("or", "and"):
        result = resolve_matching_from_pairwise(
            pw,
            d_max=d_max,
            min_iou=iou_min,
            alpha=alpha,
            policy=mode.upper(),
        )
        obj_m = object_metrics_from_counts(
            tp=result.tp, fp=result.fp, fn=result.fn, ignored=result.ignored
        )

        # Debug candidate counts derived from PairwiseTensors.
        dist_ok    = np.isfinite(pw.dist) & (pw.dist <= d_max)
        overlap_ok = pw.hit & (pw.overlap >= iou_min)
        if mode == "and":
            valid = dist_ok & overlap_ok
        else:
            valid = dist_ok | overlap_ok
        cand_pairs = int(valid.sum())
        cand_preds = int(valid.any(axis=1).sum()) if valid.size > 0 else 0
        cand_gts   = int(valid.any(axis=0).sum()) if valid.size > 0 else 0

        out[mode] = {
            "uid":             uid,
            "split":           split,
            "cell_line":       cell_line,
            "density_bin":     _assign_density_bin(gt_count),
            "model":           model_name,
            "pred_exists":     pred_exists,
            "gt_path":         gt_path_str,
            "pred_path":       pred_path_str or "",
            # Object metrics
            "obj_tp":          result.tp,
            "obj_fp":          result.fp,
            "obj_fn":          result.fn,
            "obj_ignored":     result.ignored,
            "obj_precision":   obj_m["precision"],
            "obj_recall":      obj_m["recall"],
            "obj_f1":          obj_m["f1"],
            # Count metrics
            "pred_count":      pred_count,
            "gt_count":        gt_count,
            "count_abs_error": float(abs_err),
            "count_ape":       ape,
            # Pixel metrics
            "pix_tp":          pix_conf["tp"],
            "pix_fp":          pix_conf["fp"],
            "pix_fn":          pix_conf["fn"],
            "pix_tn":          pix_conf["tn"],
            "pix_precision":   pix_m["precision"],
            "pix_recall":      pix_m["recall"],
            "pix_f1":          pix_m["f1"],
            "pix_iou":         pix_m["iou"],
            "pix_dice":        pix_m["dice"],
            # Debug
            "cand_pairs":      cand_pairs,
            "cand_preds":      cand_preds,
            "cand_gts":        cand_gts,
        }

    return out


# ---------------------------------------------------------------------------
# Public benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    model_mask_dirs: Dict[str, Path],
    metadata_df:     pd.DataFrame,
    eval_cfg:        EvalConfig,
    out_dir:         Path,
) -> Dict[str, pd.DataFrame]:
    """Evaluate all models against GT on all images, for all matching modes.

    Parameters
    ----------
    model_mask_dirs:
        ``{model_key: harmonized_mask_dir}`` — one entry per model.
    metadata_df:
        DataFrame with columns ``unique_id``, ``split``, ``cell_line``,
        ``gt_fullpath``, ``ecDNA_gt``.
    eval_cfg:
        ``EvalConfig`` with frozen defaults.
    out_dir:
        Root directory for outputs.  Writes:
        ``{out_dir}/or_matching/per_image_metrics.csv``
        ``{out_dir}/and_matching/per_image_metrics.csv``
        ``{out_dir}/errors.csv``

    Returns
    -------
    dict ``{"or": df_or, "and": df_and}`` — per-image metrics DataFrames.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Idempotency check
    if not eval_cfg.force:
        expected_rows = len(metadata_df) * len(model_mask_dirs)
        all_done = True
        for mode in eval_cfg.modes:
            csv_path = out_dir / f"{mode}_matching" / "per_image_metrics.csv"
            if not csv_path.exists():
                all_done = False
                break
            try:
                n = len(pd.read_csv(csv_path))
                if n < expected_rows:
                    all_done = False
                    break
            except Exception:
                all_done = False
                break

        if all_done:
            logger.info("All per_image_metrics CSVs exist and look complete. Skipping (use force=True).")
            result = {}
            for mode in eval_cfg.modes:
                csv_path = out_dir / f"{mode}_matching" / "per_image_metrics.csv"
                result[mode] = pd.read_csv(csv_path)
            return result

    # Build task list
    tasks = []
    for _, row in metadata_df.iterrows():
        uid       = str(row["unique_id"])
        split     = str(row.get("split", "unknown"))
        cell_line = str(row.get("cell_line", "unknown"))
        gt_path   = str(row["gt_fullpath"])

        for model_key, mask_dir in model_mask_dirs.items():
            spec     = MODEL_REGISTRY[model_key]
            pred_png = Path(mask_dir) / f"{uid}.png"
            tasks.append((
                uid, split, cell_line, gt_path,
                str(pred_png),
                spec.name,
                eval_cfg.d_max, eval_cfg.iou_min, eval_cfg.alpha,
                eval_cfg.min_area, eval_cfg.pixel_threshold,
            ))

    logger.info("Benchmark: %d tasks (%d images × %d models)",
                len(tasks), len(metadata_df), len(model_mask_dirs))

    # Dispatch
    per_mode_rows: Dict[str, List[Dict]] = {m: [] for m in ["or", "and"]}
    error_rows: List[Dict] = []

    if eval_cfg.n_workers <= 1:
        for t in tasks:
            try:
                res = _eval_one(t)
                for mode in ["or", "and"]:
                    per_mode_rows[mode].append(res[mode])
            except Exception as exc:
                uid, _, _, _, _, model_name = t[:6]
                error_rows.append({"uid": uid, "model": model_name,
                                   "error": str(exc), "traceback": traceback.format_exc()})
    else:
        with ProcessPoolExecutor(max_workers=eval_cfg.n_workers) as ex:
            futures = {ex.submit(_eval_one, t): t for t in tasks}
            for fut in as_completed(futures):
                t = futures[fut]
                uid, _, _, _, _, model_name = t[:6]
                try:
                    res = fut.result()
                    for mode in ["or", "and"]:
                        per_mode_rows[mode].append(res[mode])
                except Exception as exc:
                    error_rows.append({"uid": uid, "model": model_name,
                                       "error": str(exc), "traceback": traceback.format_exc()})
                    logger.error("uid=%s model=%s: %s", uid, model_name, exc)

    # Write error log
    pd.DataFrame(error_rows).to_csv(out_dir / "errors.csv", index=False)
    if error_rows:
        logger.warning("%d evaluation errors logged to %s/errors.csv",
                       len(error_rows), out_dir)

    if not per_mode_rows["or"]:
        raise RuntimeError("No evaluations completed. Check errors.csv.")

    # Write and return per-image CSVs
    result: Dict[str, pd.DataFrame] = {}
    model_order_names = [MODEL_REGISTRY[k].name for k in MODEL_REGISTRY
                         if k in model_mask_dirs]

    for mode in eval_cfg.modes:
        mode_dir = out_dir / f"{mode}_matching"
        mode_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(per_mode_rows[mode])
        if len(df) > 0:
            # Sort by model order then UID for deterministic output
            known_order = [n for n in MODEL_ORDER if n in df["model"].values]
            df["model"] = pd.Categorical(df["model"], categories=known_order, ordered=True)
            df = df.sort_values(["model", "uid"]).reset_index(drop=True)
        csv_path = mode_dir / "per_image_metrics.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Wrote %s (%d rows)", csv_path, len(df))
        result[mode] = df

    return result