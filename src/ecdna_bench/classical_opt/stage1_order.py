"""
ecdna_bench.classical_opt.stage1_order
========================================
Stage 1: exhaustive preprocessing-order search.

For each cell line, enumerate every order-sensitive combination
(permutation) of the 7 preprocessing ops in the registry, length 3-4 by
current defaults. Evaluate each combo on the training split with fixed
detection and matching parameters, then rank by micro-averaged F1.

This version is hardened for dense NCI-H2170 images:
* object-count and match-pair caps prevent pathological combos from OOMing;
* force=True recomputes partials instead of silently reusing old partial CSVs;
* multiprocessing workers are recycled aggressively to release OpenCV/numpy memory;
* progress logging includes cap-hit counts.
"""

from __future__ import annotations

import gc
import itertools
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
import ast
import numpy as np
import pandas as pd

from ecdna_bench.classical.preprocess import (
    rgb_to_gray_preserve,
    mask_with_roi,
    apply_chain,
)
from ecdna_bench.classical.detect import (
    detect_objects,
    merge_close_objects,
    label_objects_hsv,
)
from ecdna_bench.evaluation.objects import objects_from_mask
from ecdna_bench.evaluation.matching import match_objects

logger = logging.getLogger(__name__)

__all__ = ["Stage1Config", "Stage1Result", "run_stage1", "generate_combos", "combo_id"]

# ---------------------------------------------------------------------------
# Frozen training-time constants
# ---------------------------------------------------------------------------
_TRAIN_D_MAX: float = 20.0
_TRAIN_IOU_MIN: float = 0.1
_TRAIN_ALPHA: float = 0.5
_TRAIN_POLICY: str = "OR"
_MIN_COMBO_LEN: int = 3
_MAX_COMBO_LEN: int = 4
_MAX_TRAIN_IMAGES: int = 100
_DEFAULT_WORKERS: int = max(1, (os.cpu_count() or 1))

# Fixed detection params used during Stage-1 order search.
_FIXED_DETECT = dict(
    threshold_factor=1.0,
    morph_close_kernel=5,
    min_area=3,
    max_area=900,
)
_FIXED_MERGE_DIST = 5.0
_FIXED_HSV = dict(white_value_threshold=200.0, white_saturation_threshold=40.0)
_MIN_GT_AREA = 3

# ---------------------------------------------------------------------------
# Stage-1 OOM safety controls
# ---------------------------------------------------------------------------
# These are intentionally environment-variable driven so you can tune them from
# SLURM without editing code. A combo/image that exceeds these caps receives a
# valid zero-F1 penalty row and the job continues.
_MAX_OBJECTS = int(os.environ.get("ECDNA_OPT_MAX_OBJECTS", "5000"))
_MAX_MATCH_PAIRS = int(os.environ.get("ECDNA_OPT_MAX_MATCH_PAIRS", "5000000"))
_MAX_TASKS_PER_CHILD = int(os.environ.get("ECDNA_STAGE1_MAX_TASKS_PER_CHILD", "1"))


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Stage1Config:
    """Parameters for one Stage-1 run."""

    out_dir: Path
    partials_dir: Path
    min_combo_len: int = _MIN_COMBO_LEN
    max_combo_len: int = _MAX_COMBO_LEN
    max_train_images: int = _MAX_TRAIN_IMAGES
    max_combos: Optional[int] = None
    force: bool = False
    max_workers: int = _DEFAULT_WORKERS
    seed: int = 42


@dataclass
class Stage1Result:
    """Return value of ``run_stage1``."""

    cell_line: str
    ranking: pd.DataFrame
    best_combo: Tuple[str, ...]
    best_f1: float
    rank_csv: Path


# ---------------------------------------------------------------------------
# Combo helpers
# ---------------------------------------------------------------------------
def generate_combos(
    func_names: Optional[Sequence[str]] = None,
    min_len: int = _MIN_COMBO_LEN,
    max_len: int = _MAX_COMBO_LEN,
    max_combos: Optional[int] = None,
) -> List[Tuple[str, ...]]:
    """Enumerate all order-sensitive permutations of preprocessing ops."""
    if func_names is None:
        func_names = [
            "apply_gamma",
            "apply_sharpening",
            "apply_bilateral_filter",
            "clahe",
            "remove_background",
            "top_hat",
            "apply_sigmoid",
        ]

    result: List[Tuple[str, ...]] = []
    for k in range(min_len, max_len + 1):
        for combo in itertools.permutations(func_names, k):
            result.append(combo)
            if max_combos is not None and len(result) >= max_combos:
                return result
    return result


def combo_id(combo: Sequence[str]) -> str:
    return "__".join(combo)


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", s)


def _partial_csv_path(partials_dir: Path, cell_line: str, combo: Sequence[str]) -> Path:
    cl_dir = partials_dir / _safe_name(cell_line)
    cl_dir.mkdir(parents=True, exist_ok=True)
    return cl_dir / f"{_safe_name(cell_line)}__{combo_id(combo)}.csv"


def _rank_csv_path(out_dir: Path, cell_line: str, min_len: int, max_len: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"len{min_len}" if min_len == max_len else f"len{min_len}-{max_len}"
    return out_dir / f"{_safe_name(cell_line)}__stage1_order_{suffix}_f1_results.csv"


def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(path) + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, str(path))


# ---------------------------------------------------------------------------
# Penalty rows for pathological combos
# ---------------------------------------------------------------------------
def _penalty_row(
    uid: str,
    cell_line: str,
    n_gt: int,
    n_pred: int,
    reason: str,
    n_chrom: int = 0,
    n_ecdna: int = 0,
) -> Dict[str, Any]:
    """Return a valid zero-F1 row for a pathological sample/combo."""
    fp_penalty = min(max(int(n_pred), 1), _MAX_OBJECTS)
    return {
        "uid": uid,
        "cell_line": cell_line,
        "tp": 0,
        "fp": fp_penalty,
        "fn": int(n_gt),
        "ignored_pred": 0,
        "f1": 0.0,
        "n_pred": int(n_pred),
        "n_gt": int(n_gt),
        "n_chrom": int(n_chrom),
        "n_ecdna": int(n_ecdna),
        "object_cap_hit": 1,
        "cap_reason": reason,
        "object_cap": _MAX_OBJECTS,
        "max_match_pairs": _MAX_MATCH_PAIRS,
    }


# ---------------------------------------------------------------------------
# Per-sample evaluation
# ---------------------------------------------------------------------------
def _eval_sample(
    uid: str,
    cell_line: str,
    rgb_path: str,
    roi_path: Optional[str],
    gt_path: Optional[str],
    combo: Tuple[str, ...],
) -> Optional[Dict[str, Any]]:
    """Evaluate one sample under one combo.

    Pathological object explosions are converted into zero-F1 penalty rows
    before HSV classification or Hungarian matching can allocate large arrays.
    """
    import cv2

    try:
        try:
            cv2.setNumThreads(0)
        except Exception:
            pass

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            return None

        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE) if roi_path else None
        gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path else None
        if gt is None:
            return None

        gt_objs = objects_from_mask(gt, min_area=_MIN_GT_AREA)
        n_gt = len(gt_objs)

        if n_gt > _MAX_OBJECTS:
            logger.warning(
                "Stage-1 cap | uid=%s combo=%s | n_gt=%d > cap=%d; penalizing",
                uid,
                combo_id(combo),
                n_gt,
                _MAX_OBJECTS,
            )
            return _penalty_row(uid, cell_line, n_gt=n_gt, n_pred=0, reason="gt_object_cap")

        rgb_m = mask_with_roi(rgb, roi, mode="zero") if roi is not None else rgb
        gray = rgb_to_gray_preserve(rgb_m, mode="max")
        enhanced = apply_chain(gray, list(combo))

        raw_objs = detect_objects(enhanced, **_FIXED_DETECT)
        n_raw = len(raw_objs)
        if n_raw > _MAX_OBJECTS:
            logger.warning(
                "Stage-1 cap | uid=%s combo=%s | raw_objs=%d > cap=%d; penalizing",
                uid,
                combo_id(combo),
                n_raw,
                _MAX_OBJECTS,
            )
            return _penalty_row(uid, cell_line, n_gt=n_gt, n_pred=n_raw, reason="raw_object_cap")

        merged = merge_close_objects(raw_objs, merge_distance=_FIXED_MERGE_DIST)
        n_merged = len(merged)
        if n_merged > _MAX_OBJECTS:
            logger.warning(
                "Stage-1 cap | uid=%s combo=%s | merged_objs=%d > cap=%d; penalizing",
                uid,
                combo_id(combo),
                n_merged,
                _MAX_OBJECTS,
            )
            return _penalty_row(uid, cell_line, n_gt=n_gt, n_pred=n_merged, reason="merged_object_cap")

        labelled, counts = label_objects_hsv(
            merged,
            rgb_m,
            white_value_threshold=_FIXED_HSV["white_value_threshold"],
            white_saturation_threshold=_FIXED_HSV["white_saturation_threshold"],
        )
        pred_objs = [o for o in labelled if o.get("label") == "ecDNA"]
        n_pred = len(pred_objs)

        if n_pred > _MAX_OBJECTS:
            logger.warning(
                "Stage-1 cap | uid=%s combo=%s | pred_objs=%d > cap=%d; penalizing",
                uid,
                combo_id(combo),
                n_pred,
                _MAX_OBJECTS,
            )
            return _penalty_row(
                uid,
                cell_line,
                n_gt=n_gt,
                n_pred=n_pred,
                reason="pred_object_cap",
                n_chrom=counts.get("chromosome", 0),
                n_ecdna=counts.get("ecDNA", n_pred),
            )

        n_pairs = n_pred * n_gt
        if n_pairs > _MAX_MATCH_PAIRS:
            logger.warning(
                "Stage-1 cap | uid=%s combo=%s | match_pairs=%d > cap=%d "
                "(n_pred=%d, n_gt=%d); penalizing",
                uid,
                combo_id(combo),
                n_pairs,
                _MAX_MATCH_PAIRS,
                n_pred,
                n_gt,
            )
            return _penalty_row(
                uid,
                cell_line,
                n_gt=n_gt,
                n_pred=n_pred,
                reason="match_pair_cap",
                n_chrom=counts.get("chromosome", 0),
                n_ecdna=counts.get("ecDNA", n_pred),
            )

        result = match_objects(
            pred_objs=pred_objs,
            gt_objs=gt_objs,
            max_dist=_TRAIN_D_MAX,
            min_iou=_TRAIN_IOU_MIN,
            alpha=_TRAIN_ALPHA,
            policy=_TRAIN_POLICY,
        )

        tp = int(result.tp)
        fp = int(result.fp)
        fn = int(result.fn)
        ignored = int(result.ignored_count)
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0

        return {
            "uid": uid,
            "cell_line": cell_line,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "ignored_pred": ignored,
            "f1": float(f1),
            "n_pred": n_pred,
            "n_gt": n_gt,
            "n_chrom": counts.get("chromosome", 0),
            "n_ecdna": counts.get("ecDNA", 0),
            "object_cap_hit": 0,
            "cap_reason": "",
            "object_cap": _MAX_OBJECTS,
            "max_match_pairs": _MAX_MATCH_PAIRS,
        }

    except Exception as exc:
        logger.debug("_eval_sample uid=%s combo=%s error: %s", uid, combo_id(combo), exc)
        return None


# ---------------------------------------------------------------------------
# Per-combo worker task
# ---------------------------------------------------------------------------
def _eval_combo_task(args: Tuple) -> Dict[str, Any]:
    """Evaluate one combo over all Stage-1 samples and write one partial CSV."""
    cell_line, combo, sample_tuples, partial_path_str, force = args
    partial_path = Path(partial_path_str)
    cid = combo_id(combo)

    if force and partial_path.exists():
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass

    if partial_path.exists():
        return {"combo_id": cid, "status": "skipped", "n_rows": 0, "cap_hits": 0}

    rows: List[Dict[str, Any]] = []
    cap_hits = 0
    for uid, rgb_path, roi_path, gt_path in sample_tuples:
        r = _eval_sample(uid, cell_line, rgb_path, roi_path, gt_path, combo)
        if r is not None:
            rows.append(r)
            cap_hits += int(r.get("object_cap_hit", 0) or 0)

    if rows:
        _atomic_write_csv(pd.DataFrame(rows), partial_path)

    # Explicitly release large lists in recycled child processes.
    del rows
    gc.collect()
    return {"combo_id": cid, "status": "done", "n_rows": len(sample_tuples), "cap_hits": cap_hits}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def _aggregate(
    cell_line: str,
    combos: List[Tuple[str, ...]],
    partials_dir: Path,
    out_dir: Path,
    min_len: int,
    max_len: int,
) -> Optional[pd.DataFrame]:
    """Read partial CSVs for a cell line and write a sorted ranking CSV."""
    rows = []
    for combo in combos:
        ppath = _partial_csv_path(partials_dir, cell_line, combo)
        if not ppath.exists():
            continue
        try:
            df = pd.read_csv(ppath)
        except Exception as exc:
            logger.warning("Could not read partial %s: %s", ppath, exc)
            continue
        if df.empty:
            continue

        for col, default in [
            ("ignored_pred", 0),
            ("object_cap_hit", 0),
            ("cap_reason", ""),
            ("n_pred", 0),
            ("n_gt", 0),
        ]:
            if col not in df.columns:
                df[col] = default

        tp = int(df["tp"].sum())
        fp = int(df["fp"].sum())
        fn = int(df["fn"].sum())
        ign = int(df["ignored_pred"].sum())
        denom = 2 * tp + fp + fn
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        cap_hits = int(df["object_cap_hit"].fillna(0).sum())
        cap_reasons = (
            df.loc[df["object_cap_hit"].fillna(0).astype(int) > 0, "cap_reason"]
            .astype(str)
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "cell_line": cell_line,
                "combo_id": combo_id(combo),
                "combo": list(combo),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "ignored_pred_total": ign,
                "ignored_pred_per_img": float(df["ignored_pred"].mean()),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "n_images": int(len(df)),
                "mean_uid_f1": float(df["f1"].mean()),
                "object_cap_hits": cap_hits,
                "object_cap_hit_rate": float(cap_hits / len(df)) if len(df) else 0.0,
                "cap_reasons_json": json.dumps(cap_reasons, sort_keys=True),
                "max_n_pred": int(pd.to_numeric(df["n_pred"], errors="coerce").fillna(0).max()),
                "max_n_gt": int(pd.to_numeric(df["n_gt"], errors="coerce").fillna(0).max()),
            }
        )

    if not rows:
        logger.warning("No partials for cell_line=%s; skipping aggregation.", cell_line)
        return None

    out_df = pd.DataFrame(rows).sort_values(
        ["f1", "object_cap_hits", "fp"], ascending=[False, True, True]
    )
    out_path = _rank_csv_path(out_dir, cell_line, min_len, max_len)
    _atomic_write_csv(out_df, out_path)
    logger.info("Saved Stage-1 ranking for %s -> %s", cell_line, out_path)
    return out_df


def _parse_combo_from_row(row: Any) -> Tuple[str, ...]:
    combo_val = row.get("combo", None)

    if isinstance(combo_val, (list, tuple)):
        return tuple(str(x) for x in combo_val)

    if isinstance(combo_val, str) and combo_val.strip():
        try:
            parsed = ast.literal_eval(combo_val)
            if isinstance(parsed, (list, tuple)):
                return tuple(str(x) for x in parsed)
        except Exception:
            pass

    combo_id_val = row.get("combo_id", "")
    return tuple(str(combo_id_val).split("__"))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_stage1(samples: List[Any], cell_line: str, cfg: Stage1Config) -> Stage1Result:
    """Run Stage-1 order optimization for one cell line."""
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.partials_dir.mkdir(parents=True, exist_ok=True)

    rank_path = _rank_csv_path(cfg.out_dir, cell_line, cfg.min_combo_len, cfg.max_combo_len)
    if rank_path.exists() and not cfg.force:
        logger.info("Stage-1 ranking already exists for %s; skipping (use force=True).", cell_line)
        ranking = pd.read_csv(rank_path)
        best = ranking.iloc[0]
        return Stage1Result(
            cell_line=cell_line,
            ranking=ranking,
            best_combo=_parse_combo_from_row(best),
            best_f1=float(best["f1"]),
            rank_csv=rank_path,
        )

    rng = np.random.default_rng(cfg.seed)
    if len(samples) > cfg.max_train_images:
        idx = rng.choice(len(samples), cfg.max_train_images, replace=False)
        samples = [samples[i] for i in sorted(idx)]

    logger.info(
        "Stage-1 | cell_line=%s | n_train=%d | workers=%d | object_cap=%d | match_pair_cap=%d | maxtasksperchild=%d",
        cell_line,
        len(samples),
        cfg.max_workers,
        _MAX_OBJECTS,
        _MAX_MATCH_PAIRS,
        _MAX_TASKS_PER_CHILD,
    )

    sample_tuples = [
        (
            str(s.uid),
            str(s.rgb_path),
            str(s.roi_path) if getattr(s, "roi_path", None) is not None else "",
            str(s.gt_mask_path) if getattr(s, "gt_mask_path", None) is not None else "",
        )
        for s in samples
    ]

    combos = generate_combos(
        min_len=cfg.min_combo_len,
        max_len=cfg.max_combo_len,
        max_combos=cfg.max_combos,
    )
    logger.info("Stage-1 | %d combos to evaluate", len(combos))

    tasks = [
        (
            cell_line,
            combo,
            sample_tuples,
            str(_partial_csv_path(cfg.partials_dir, cell_line, combo)),
            bool(cfg.force),
        )
        for combo in combos
    ]

    n_done = 0
    total_cap_hits = 0

    if cfg.max_workers <= 1:
        for task in tasks:
            result = _eval_combo_task(task)
            n_done += 1
            total_cap_hits += int(result.get("cap_hits", 0))
            if n_done % 50 == 0 or n_done == len(tasks):
                logger.info(
                    "Stage-1 | %s | %d / %d combos done | cap_hits=%d",
                    cell_line,
                    n_done,
                    len(tasks),
                    total_cap_hits,
                )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=int(cfg.max_workers),
            maxtasksperchild=max(1, int(_MAX_TASKS_PER_CHILD)),
        ) as pool:
            for result in pool.imap_unordered(_eval_combo_task, tasks, chunksize=1):
                n_done += 1
                total_cap_hits += int(result.get("cap_hits", 0))
                if n_done % 50 == 0 or n_done == len(tasks):
                    logger.info(
                        "Stage-1 | %s | %d / %d combos done | cap_hits=%d",
                        cell_line,
                        n_done,
                        len(tasks),
                        total_cap_hits,
                    )

    ranking = _aggregate(
        cell_line=cell_line,
        combos=combos,
        partials_dir=cfg.partials_dir,
        out_dir=cfg.out_dir,
        min_len=cfg.min_combo_len,
        max_len=cfg.max_combo_len,
    )

    if ranking is None or ranking.empty:
        raise RuntimeError(f"Stage-1 produced no results for cell_line={cell_line!r}")

    best = ranking.iloc[0]
    return Stage1Result(
        cell_line=cell_line,
        ranking=ranking,
        best_combo=_parse_combo_from_row(best),
        best_f1=float(best["f1"]),
        rank_csv=rank_path,
    )
