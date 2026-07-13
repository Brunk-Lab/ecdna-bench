"""
ecdna_bench.cli.build_metadata
================================
Build the master metadata CSV and counts CSV from the raw file tree.

Usage
-----
    python -m ecdna_bench.cli.build_metadata --config configs/default.yaml

Outputs
-------
* ``{paths.metadata_csv}``   — per-image metadata (1,145 rows for the benchmark subset)
* ``{paths.counts_csv}``     — uid / cell_line / split / ecDNA_gt / coord_count_npy

Count columns in counts_master.csv
------------------------------------
ecDNA_gt
    The CANONICAL evaluation count: 8-connectivity connected components of
    the rendered GT mask with min_area=3 px.  This matches the evaluation
    framework exactly and is what every model in the benchmark is compared
    against.  See PROJECT_RULES.md §2 and §7.

coord_count_npy
    The annotation-point count read from the sibling NPY/NPZ coordinate
    files (``gt_coords/`` directory).  Released for transparency.  Always
    ``>= ecDNA_gt``; differences arise from the diamond-merge effect.

count_source
    Always ``"gt_mask_cc_conn8_min3"`` to document the canonical definition.

Required config keys (under ``paths``)
----------------------------------------
* ``rgb_dir``        — root of RGB image files
* ``gt_mask_dir``    — root of rendered GT binary mask files
* ``split_dir``      — directory containing train_ids.csv / val_ids.csv / test_ids.csv
* ``metadata_csv``   — output path for metadata CSV
* ``counts_csv``     — output path for counts CSV

Optional config keys (under ``paths``)
----------------------------------------
* ``dapi_dir``       — root of DAPI image files
* ``roi_mask_dir``   — root of ROI mask files
* ``mia_mask_dir``   — root of MIA pre-computed mask files
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ecdna_bench.cli._common import (
    get_path,
    load_config,
    require_dir,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build master metadata CSV and counts CSV for ecdna-bench.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Path to YAML config file.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing metadata CSV even if it already exists.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    rgb_dir     = get_path(cfg, "rgb_dir")
    gt_mask_dir = get_path(cfg, "gt_mask_dir")
    split_dir   = get_path(cfg, "split_dir")

    metadata_out = resolve_path(
        cfg.get("paths", {}).get("metadata_csv", "metadata.csv")
    )
    counts_out = resolve_path(
        cfg.get("paths", {}).get("counts_csv", "counts_master.csv")
    )

    optional = {
        "dapi_dir":     get_path(cfg, "dapi_dir",     required=False),
        "roi_mask_dir": get_path(cfg, "roi_mask_dir", required=False),
        "mia_mask_dir": get_path(cfg, "mia_mask_dir", required=False),
    }

    # ------------------------------------------------------------------
    # Guard: skip if already built (unless --force)
    # ------------------------------------------------------------------
    if metadata_out.exists() and not args.force:
        logger.info(
            "metadata CSV already exists: %s  (use --force to rebuild)",
            metadata_out,
        )
        return

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    logger.info("Building master metadata ...")
    logger.info("  rgb_dir      = %s", rgb_dir)
    logger.info("  gt_mask_dir  = %s", gt_mask_dir)
    logger.info("  split_dir    = %s", split_dir)
    logger.info("  metadata_out = %s", metadata_out)
    logger.info("  counts_out   = %s", counts_out)

    from ecdna_bench.data.metadata import build_metadata

    meta_df, counts_df = build_metadata(
        rgb_dir      = rgb_dir,
        gt_mask_dir  = gt_mask_dir,
        split_dir    = split_dir,
        dapi_dir     = optional["dapi_dir"],
        roi_mask_dir = optional["roi_mask_dir"],
        mia_mask_dir = optional["mia_mask_dir"],
        metadata_out = metadata_out,
        counts_out   = counts_out,
    )

    logger.info("Done — %d rows written to %s", len(meta_df),   metadata_out)
    logger.info("       %d rows written to %s", len(counts_df), counts_out)

    # Quick sanity check: confirm ecDNA_gt is the mask-CC count.
    import pandas as pd
    n_coord_source = (counts_df["count_source"] == "gt_mask_cc_conn8_min3").sum()
    logger.info(
        "count_source check: %d/%d rows have 'gt_mask_cc_conn8_min3' (expected all)",
        n_coord_source,
        len(counts_df),
    )
    if "coord_count_npy" in counts_df.columns:
        n_diff = (counts_df["coord_count_npy"] != counts_df["ecDNA_gt"]).sum()
        logger.info(
            "diamond-merge: %d/%d images have coord_count_npy > ecDNA_gt",
            int(n_diff),
            len(counts_df),
        )


if __name__ == "__main__":
    main()