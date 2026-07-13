"""
ecdna_bench.cli.sensitivity
==============================
Run sensitivity sweep over d_max × IoU_min × OR/AND matching policies.

Usage
-----
    # single model (legacy)
    python -m ecdna_bench.cli.sensitivity --config configs/benchmark.yaml --model eccount_peaks

    # all six models in the registry
    python -m ecdna_bench.cli.sensitivity --config configs/benchmark.yaml --model all

The sweep evaluates the chosen model(s) across a grid of matching parameters
and writes:

* ``{frozen_results_dir}/sensitivity/sensitivity_sweep.csv``
* ``{frozen_results_dir}/sensitivity/sensitivity_sweep_per_image.csv``

When ``--model all`` is passed, both CSVs contain rows for every model in
``MODEL_REGISTRY`` (one ``model`` column distinguishes them). The aggregate
CSV is consumed by ``figures/supp/sfig6_sensitivity.py``.

Performance note
----------------
Pairwise geometry is computed once per image using
``max_precompute_dist = max(_D_MAX_GRID)``. All parameter combinations then
reuse those tensors through ``sweep_matching_grid``.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from ecdna_bench.cli._common import (
    get_path,
    load_config,
    require_file,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical sweep grid — Supplementary sensitivity analysis.
# ---------------------------------------------------------------------------
_D_MAX_GRID = [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
_IOU_MIN_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
_POLICIES = ["or", "and"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sensitivity sweep over matching parameters.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/benchmark.yaml"),
    )
    p.add_argument(
        "--model",
        default="eccount_peaks",
        help=(
            "Model key to sweep, or 'all' to sweep every model in "
            "MODEL_REGISTRY."
        ),
    )
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all"],
    )
    p.add_argument(
        "--n-workers",
        type=int,
        default=4,
        help="Reserved for future parallel version. Current implementation is serial.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Redo sweep even if output already exists.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def _load_gray(path: Path) -> Optional[np.ndarray]:
    """Load a grayscale image with OpenCV."""
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _build_items_for_model(
    *,
    model_key: str,
    spec,
    df: pd.DataFrame,
    mask_dir: Path,
    max_precompute_dist: float,
    split_label: str,
    extract_objects_fn,
    precompute_pairwise_fn,
) -> List[tuple]:
    """Build the (uid, pairwise_tensors, extra) list for one model."""
    file_idx: Dict[str, Path] = {
        p.stem.lower(): p for p in mask_dir.glob("*.png")
    }

    items: List[tuple] = []
    n_images = len(df)

    for img_idx, (_, row) in enumerate(df.iterrows()):
        uid = str(row["unique_id"])
        gt_path = Path(str(row.get("gt_fullpath", "")))
        pred_path = file_idx.get(uid.lower()) or mask_dir / f"{uid}.png"

        gt = _load_gray(gt_path)
        if gt is None:
            logger.warning(
                "[%s] Cannot load GT for uid=%s — skipping.", model_key, uid
            )
            continue

        pred = _load_gray(pred_path)
        if pred is None:
            pred = np.zeros_like(gt)

        if gt.shape != pred.shape:
            logger.warning(
                "[%s] Shape mismatch for uid=%s — skipping. GT=%s PRED=%s",
                model_key,
                uid,
                gt.shape,
                pred.shape,
            )
            continue

        gt_objs = extract_objects_fn(gt, min_area=3)
        pred_objs = extract_objects_fn(pred, min_area=3)

        pw = precompute_pairwise_fn(
            pred_objs,
            gt_objs,
            max_precompute_dist=max_precompute_dist,
        )

        items.append(
            (
                uid,
                pw,
                {
                    "model": spec.name,
                    "model_key": model_key,
                    "split": split_label,
                    "cell_line": str(row.get("cell_line", "unknown")),
                    "uid": uid,
                },
            )
        )

        if (img_idx + 1) % 25 == 0 or (img_idx + 1) == n_images:
            logger.info(
                "[%s] Precomputed %d / %d images",
                model_key,
                img_idx + 1,
                n_images,
            )

    return items


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv = get_path(cfg, "consistency_csv")
    frozen_results_dir = (
        get_path(cfg, "frozen_results_dir", required=False)
        or resolve_path("release/frozen_results")
    )
    require_file(consistency_csv, "consistency CSV")

    from ecdna_bench.benchmark.registry import MODEL_REGISTRY
    from ecdna_bench.benchmark.run import _extract_objects
    from ecdna_bench.evaluation.matching import precompute_pairwise
    from ecdna_bench.evaluation.sensitivity import sweep_matching_grid

    # ------------------------------------------------------------------
    # Resolve which model keys to run.
    # ------------------------------------------------------------------
    if args.model == "all":
        model_keys = list(MODEL_REGISTRY.keys())
    else:
        if args.model not in MODEL_REGISTRY:
            logger.error(
                "Unknown model key: %r. Available: %s (or 'all')",
                args.model,
                sorted(MODEL_REGISTRY),
            )
            return
        model_keys = [args.model]

    logger.info("Sweeping models: %s", model_keys)

    # ------------------------------------------------------------------
    # Output paths — single combined CSV per output type.
    # ------------------------------------------------------------------
    out_dir = frozen_results_dir / "sensitivity"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_dir / "sensitivity_sweep.csv"
    tidy_csv = out_dir / "sensitivity_sweep_per_image.csv"

    if out_csv.exists() and tidy_csv.exists() and not args.force:
        logger.info(
            "Sensitivity sweep already exists: %s and %s  (use --force to redo)",
            out_csv,
            tidy_csv,
        )
        return

    # ------------------------------------------------------------------
    # Load and filter the consistency CSV once for all models.
    # ------------------------------------------------------------------
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]

    if args.split != "all":
        df = df[df["split"].astype(str) == args.split]

    logger.info(
        "Sensitivity sweep: %d images, models=%s, split=%s",
        len(df),
        model_keys,
        args.split,
    )
    logger.info(
        "Grid: d_max=%s  iou_min=%s  policies=%s",
        _D_MAX_GRID,
        _IOU_MIN_GRID,
        _POLICIES,
    )

    max_precompute_dist = float(max(_D_MAX_GRID))

    # ------------------------------------------------------------------
    # Build (uid, pw, extra) items for every model, then sweep all at once.
    # ------------------------------------------------------------------
    all_items: List[tuple] = []

    for model_key in model_keys:
        spec = MODEL_REGISTRY[model_key]
        mask_dir = (
            get_path(cfg, spec.mask_dir_key, required=False)
            or resolve_path(f"release/harmonized_masks/{model_key}")
        )
        if not mask_dir.exists():
            logger.warning(
                "Mask dir not found for %s: %s — skipping this model.",
                model_key,
                mask_dir,
            )
            continue

        logger.info("[%s] Mask dir: %s", model_key, mask_dir)

        items = _build_items_for_model(
            model_key=model_key,
            spec=spec,
            df=df,
            mask_dir=mask_dir,
            max_precompute_dist=max_precompute_dist,
            split_label=args.split,
            extract_objects_fn=_extract_objects,
            precompute_pairwise_fn=precompute_pairwise,
        )
        logger.info(
            "[%s] Prepared %d images for sweep.", model_key, len(items)
        )
        all_items.extend(items)

    if not all_items:
        logger.error("No images to sweep across any model. Aborting.")
        return

    # ------------------------------------------------------------------
    # Per-image sweep across all (model × image) entries.
    # one row per model × uid × d_max × iou_min × policy.
    # ------------------------------------------------------------------
    tidy_df = sweep_matching_grid(
        all_items,
        d_max_grid=_D_MAX_GRID,
        min_iou_grid=_IOU_MIN_GRID,
        policies=[p.upper() for p in _POLICIES],
        alpha=0.5,
    )

    tidy_df.to_csv(tidy_csv, index=False)
    logger.info(
        "Per-image sensitivity sweep written to %s (%d rows)",
        tidy_csv,
        len(tidy_df),
    )

    # ------------------------------------------------------------------
    # Aggregate: one row per model × split × policy × d_max × iou_min.
    # ------------------------------------------------------------------
    if tidy_df.empty:
        logger.warning("No valid images were swept. Writing empty summary.")
        sweep_df = pd.DataFrame(
            columns=[
                "model",
                "split",
                "policy",
                "d_max",
                "iou_min",
                "tp",
                "fp",
                "fn",
                "ignored",
                "precision",
                "recall",
                "f1",
            ]
        )
    else:
        tidy_df["policy"] = tidy_df["policy"].astype(str).str.lower()

        group_cols = ["model", "split", "policy", "d_max", "min_iou"]

        sweep_df = (
            tidy_df.groupby(group_cols, as_index=False)[
                ["tp", "fp", "fn", "ignored"]
            ]
            .sum()
            .rename(columns={"min_iou": "iou_min"})
        )

        tp = sweep_df["tp"].astype(float)
        fp = sweep_df["fp"].astype(float)
        fn = sweep_df["fn"].astype(float)

        sweep_df["precision"] = np.where(
            (tp + fp) > 0,
            tp / (tp + fp),
            0.0,
        )
        sweep_df["recall"] = np.where(
            (tp + fn) > 0,
            tp / (tp + fn),
            0.0,
        )
        sweep_df["f1"] = np.where(
            (2 * tp + fp + fn) > 0,
            (2 * tp) / (2 * tp + fp + fn),
            0.0,
        )

    sweep_df.to_csv(out_csv, index=False)
    logger.info(
        "Sensitivity sweep written to %s (%d rows, %d unique models)",
        out_csv,
        len(sweep_df),
        sweep_df["model"].nunique() if not sweep_df.empty else 0,
    )


if __name__ == "__main__":
    main()