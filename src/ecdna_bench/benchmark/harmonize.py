"""
ecdna_bench.benchmark.harmonize
=================================
Harmonize raw baseline predictions to canonical binary masks before
running the benchmark.

For each model that has a ``harmonizer`` set in ``MODEL_REGISTRY``, this
module calls the corresponding function from ``ecdna_bench.baselines``
and validates that the output directory has adequate UID coverage.

Models with ``harmonizer=None`` (ecCount exports, Classical pipeline
outputs) are assumed to already be in canonical format; they are passed
through unchanged.

Design rules
------------
* Pure library — no argparse, no global state.
* Idempotent when ``force=False``: if the output directory already exists
  and has the expected number of masks, the harmonizer is skipped.
* Validates UID coverage: warns if > 0 missing, raises if > 10% missing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ecdna_bench.benchmark.registry import MODEL_REGISTRY, ModelSpec

logger = logging.getLogger(__name__)

__all__ = ["harmonize_all", "harmonize_one_model"]

# Harmonizer dispatch table: name → callable
_HARMONIZER_FN: Dict[str, Any] = {}


def _get_harmonizer(name: str):
    """Lazily import and cache harmonizer functions to avoid heavy top-level imports."""
    if name not in _HARMONIZER_FN:
        if name == "ecseg":
            from ecdna_bench.baselines.ecseg import harmonize_ecseg
            _HARMONIZER_FN["ecseg"] = harmonize_ecseg
        elif name == "mia":
            from ecdna_bench.baselines.mia import harmonize_mia
            _HARMONIZER_FN["mia"] = harmonize_mia
        elif name == "label_engine":
            from ecdna_bench.baselines.label_engine import harmonize_label_engine
            _HARMONIZER_FN["label_engine"] = harmonize_label_engine
        else:
            raise ValueError(f"Unknown harmonizer: {name!r}")
    return _HARMONIZER_FN[name]


def harmonize_one_model(
    model_key:  str,
    pred_dir:   Path,
    output_dir: Path,
    uid_list:   List[str],
    min_area:   int  = 3,
    force:      bool = False,
) -> Dict[str, str]:
    """Harmonize one model's raw predictions to canonical binary masks.

    Parameters
    ----------
    model_key:
        Key in ``MODEL_REGISTRY`` (e.g. ``"ecseg"``).
    pred_dir:
        Directory containing raw prediction files for this model.
    output_dir:
        Where to write normalized binary PNGs.
    uid_list:
        UIDs to process.
    min_area:
        Minimum CC area for noise removal (default 3, matching evaluation).
    force:
        If True, re-harmonize even if output already exists.

    Returns
    -------
    dict ``{uid: "ok" | "missing" | "error: <msg>"}``
    """
    spec = MODEL_REGISTRY[model_key]
    pred_dir_p  = Path(pred_dir)
    output_dir_p = Path(output_dir)
    if spec.harmonizer is None:
        # No harmonization needed — canonical masks live in pred_dir (which IS the mask dir)
        logger.info("Model %s has no harmonizer; assuming masks at %s are canonical.",
                    model_key, pred_dir_p)
        def _exists(uid: str) -> bool:
            return ((pred_dir_p / f"{uid}.png").exists()
                    or (output_dir_p / f"{uid}.png").exists())
        return {uid: "ok" if _exists(uid) else "missing" for uid in uid_list}

    # Check if already done
    output_dir = Path(output_dir)
    if not force and output_dir.exists():
        existing = {p.stem for p in output_dir.glob("*.png")}
        expected = set(uid_list)
        if expected <= existing:
            logger.info("Model %s: harmonized masks already exist in %s. Skipping.",
                        model_key, output_dir)
            return {uid: "ok" for uid in uid_list}

    fn = _get_harmonizer(spec.harmonizer)
    return fn(pred_dir=pred_dir, output_dir=output_dir,
               uid_list=uid_list, min_area=min_area)


def harmonize_all(
    model_keys:    List[str],
    pred_dirs:     Dict[str, Path],
    output_dirs:   Dict[str, Path],
    uid_list:      List[str],
    min_area:      int  = 3,
    force:         bool = False,
    missing_ok_pct: float = 10.0,
) -> Dict[str, Path]:
    """Harmonize all specified models.

    Parameters
    ----------
    model_keys:
        Keys from ``MODEL_REGISTRY`` to harmonize.
    pred_dirs:
        ``{model_key: raw_prediction_dir}``.
    output_dirs:
        ``{model_key: harmonized_output_dir}``.
    uid_list:
        UIDs to process.
    min_area:
        Min CC area for noise removal.
    force:
        If True, redo harmonization even if output exists.
    missing_ok_pct:
        Percentage of missing UIDs tolerated before raising an error.

    Returns
    -------
    dict ``{model_key: harmonized_mask_dir}``
    """
    results: Dict[str, Path] = {}

    for key in model_keys:
        if key not in MODEL_REGISTRY:
            raise KeyError(f"Unknown model key: {key!r}. Available: {sorted(MODEL_REGISTRY)}")

        pred_dir   = pred_dirs.get(key)
        output_dir = output_dirs.get(key)

        if pred_dir is None or output_dir is None:
            raise ValueError(f"pred_dirs or output_dirs missing entry for model key {key!r}")

        logger.info("Harmonizing model: %s", key)
        status = harmonize_one_model(
            model_key  = key,
            pred_dir   = Path(pred_dir),
            output_dir = Path(output_dir),
            uid_list   = uid_list,
            min_area   = min_area,
            force      = force,
        )

        n_total   = len(uid_list)
        n_ok      = sum(1 for v in status.values() if v == "ok")
        n_missing = sum(1 for v in status.values() if v == "missing")
        n_errors  = sum(1 for v in status.values() if v.startswith("error"))

        logger.info("  Model %s: ok=%d missing=%d errors=%d / total=%d",
                    key, n_ok, n_missing, n_errors, n_total)

        if n_missing > 0:
            pct_missing = 100.0 * n_missing / max(n_total, 1)
            msg = (f"Model {key!r}: {n_missing}/{n_total} UIDs missing "
                   f"({pct_missing:.1f}%)")
            if pct_missing > missing_ok_pct:
                raise RuntimeError(
                    f"{msg} exceeds tolerance of {missing_ok_pct}%."
                )
            logger.warning("%s", msg)

        results[key] = Path(output_dir)

    return results
