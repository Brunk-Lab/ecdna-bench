"""
ecdna_bench.cli.run_qc
========================
File audit + count-mask consistency check.

Usage
-----
    python -m ecdna_bench.cli.run_qc --config configs/default.yaml

    # Include the slower GT morphology audit (adds diamond-merge statistics):
    python -m ecdna_bench.cli.run_qc --config configs/default.yaml \\
        --include-gt-morphology

Outputs
-------
* ``{paths.consistency_csv}``
    Per-image QC results (64+ columns). All 1,145 benchmark images are
    expected to pass ``file_audit_pass=True`` and
    ``count_mask_consistent=True``.

* ``{paths.consistency_csv}.file_audit_summary.txt``
    Human-readable file-audit pass rates.

* ``{paths.consistency_csv}.count_mask_summary.txt``
    Human-readable count/mask consistency rates.  When
    ``--include-gt-morphology`` is used, this file also contains
    DIAMOND-MERGE STATISTICS showing how many images have
    ``coord_count_npy > ecDNA_gt`` and by how much.

What this checks
-----------------
1. File audit (run_file_audit)
   Every RGB / DAPI / ROI / GT file exists, is readable, and the four
   modalities have matching spatial dimensions.

2. Count-mask consistency (run_count_mask_consistency)
   ``ecDNA_gt == 0  iff  gt_mask is empty``.
   After a correct build_metadata run, ecDNA_gt is the GT-mask CC count,
   so this check is sign-based and all 1,145 images pass.

3. GT morphology audit (run_gt_morphology_audit, optional)
   Per-image CC counts with min_area=3 filter, area statistics, and the
   diamond-merge discrepancy (coord_count_npy - ecDNA_gt).  Source for
   the Supplementary Figure on annotation uncertainty.

Required config keys (under ``paths``)
----------------------------------------
* ``metadata_csv``      — path to metadata.csv (output of build_metadata)
* ``counts_csv``        — path to counts_master.csv (output of build_metadata)
* ``data_root``         — root directory for resolving relative image paths
* ``consistency_csv``   — output path for the QC results CSV
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ecdna_bench.cli._common import (
    get_path,
    load_config,
    require_file,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run file audit + count-mask consistency QC.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing QC CSV even if it already exists.",
    )
    p.add_argument(
        "--include-gt-morphology",
        action="store_true",
        default=False,
        help=(
            "Also run the GT morphology audit (per-image CC counts, area stats, "
            "and diamond-merge discrepancy between coord_count_npy and ecDNA_gt). "
            "Slower (~15 min for 1,145 images) but required for the "
            "Supplementary Figure on annotation uncertainty."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    # ------------------------------------------------------------------
    # Resolve paths
    # ------------------------------------------------------------------
    metadata_csv = get_path(cfg, "metadata_csv")
    counts_csv   = get_path(cfg, "counts_csv")

    consistency_out = resolve_path(
        cfg.get("paths", {}).get(
            "consistency_csv",
            "dl_master_metadata_stage1_step3_consistency.csv",
        )
    )

    data_root = get_path(cfg, "data_root", required=False)

    require_file(metadata_csv, "metadata CSV")
    require_file(counts_csv,   "counts CSV")

    # ------------------------------------------------------------------
    # Guard: skip if already built (unless --force)
    # ------------------------------------------------------------------
    if consistency_out.exists() and not args.force:
        logger.info(
            "QC CSV already exists: %s  (use --force to rerun)",
            consistency_out,
        )
        return

    # ------------------------------------------------------------------
    # Run QC
    # ------------------------------------------------------------------
    logger.info("Running QC on %s ...", metadata_csv)
    if args.include_gt_morphology:
        logger.info("GT morphology audit enabled (diamond-merge statistics will be computed)")

    from ecdna_bench.data.qc import run_qc

    result_df = run_qc(
        metadata_csv          = metadata_csv,
        counts_csv            = counts_csv,
        data_root             = data_root,
        output_csv            = consistency_out,
        include_gt_morphology = args.include_gt_morphology,
    )

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    if result_df.empty:
        logger.warning("QC produced an empty DataFrame — check inputs.")
        return

    n = len(result_df)

    if "file_audit_pass" in result_df.columns:
        n_audit = int(result_df["file_audit_pass"].astype(bool).sum())
        logger.info("File audit:          %d / %d pass", n_audit, n)
        if n_audit < n:
            logger.warning(
                "%d images FAILED the file audit — inspect %s for details",
                n - n_audit,
                consistency_out,
            )

    if "count_mask_consistent" in result_df.columns:
        n_cmc = int(result_df["count_mask_consistent"].astype(bool).sum())
        logger.info("Count/mask consist.: %d / %d pass", n_cmc, n)
        if n_cmc < n:
            logger.warning(
                "%d images FAILED count_mask_consistency — "
                "these should NOT enter ecCount training",
                n - n_cmc,
            )

    if "gt_components_match_ecDNA_gt" in result_df.columns:
        n_match = int(result_df["gt_components_match_ecDNA_gt"].astype(bool).sum())
        logger.info("gt_n_components == ecDNA_gt: %d / %d", n_match, n)
        if n_match < n:
            logger.warning(
                "%d images have gt_n_components != ecDNA_gt — "
                "this indicates build_metadata was run with different parameters",
                n - n_match,
            )

    if "coord_gt_cc_match" in result_df.columns:
        n_coord_match = int(result_df["coord_gt_cc_match"].astype(bool).sum())
        logger.info(
            "Diamond-merge: coord_count_npy == ecDNA_gt in %d / %d images "
            "(%d have coord > CC)",
            n_coord_match, n, n - n_coord_match,
        )

    logger.info("QC results written to %s", consistency_out)


if __name__ == "__main__":
    main()