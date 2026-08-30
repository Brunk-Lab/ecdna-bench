"""
ecdna_bench.cli.benchmark
===========================
Run the full cross-model benchmark evaluation.

Usage
-----
    python -m ecdna_bench.cli.benchmark --config configs/default.yaml

Pipeline
--------
1. harmonize_all  — ensure all mask dirs are in canonical format
2. run_benchmark  — evaluate all models × OR/AND policies
3. aggregate      — compute group summaries
4. write_frozen_results — write release/frozen_results/ CSVs

Required config keys (under ``paths``)
---------------------------------------
* ``consistency_csv``
* ``frozen_results_dir``
* One ``{model_key}_masks`` key per model you want to include.

Optional
--------
* ``benchmark.n_workers``  (default 8)
* ``benchmark.force``      (default false)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ecdna_bench.cli._common import (
    get_path, load_config, require_file, resolve_path, setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full cross-model benchmark evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--models", nargs="*", default=None,
                   help="Model keys to include (default: all in registry).")
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-harmonize", action="store_true",
                   help="Skip harmonization step (masks assumed ready).")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Override frozen_results_dir from config.")               
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv    = get_path(cfg, "consistency_csv")
    frozen_results_dir = args.output_dir or \
                         get_path(cfg, "frozen_results_dir", required=False) or \
                         resolve_path("release/frozen_results")
    harmonized_root    = get_path(cfg, "harmonized_masks_dir", required=False) or \
                         resolve_path("release/harmonized_masks")
    require_file(consistency_csv, "consistency CSV")

    from ecdna_bench.benchmark.registry import MODEL_REGISTRY, MODEL_ORDER
    from ecdna_bench.benchmark.harmonize import harmonize_all
    from ecdna_bench.benchmark.run import EvalConfig, run_benchmark
    from ecdna_bench.benchmark.aggregate import aggregate_benchmark, write_frozen_results

    import pandas as pd
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]
    uid_list = df["unique_id"].astype(str).tolist()

    model_keys = args.models or list(MODEL_REGISTRY.keys())
    n_workers  = args.n_workers or cfg.get("benchmark", {}).get("n_workers", 8)

    # Build mask_dirs from config paths
    mask_dirs = {}
    for key in model_keys:
        spec = MODEL_REGISTRY.get(key)
        if spec is None:
            logger.warning("Unknown model key %r — skipping.", key)
            continue
        p = get_path(cfg, spec.mask_dir_key, required=False)
        if p is None:
            p = harmonized_root / key
        mask_dirs[key] = p

    # 1. Harmonize
    if not args.skip_harmonize:
        logger.info("Step 1: harmonizing masks ...")
        harmonize_all(
            model_keys  = list(mask_dirs.keys()),
            pred_dirs   = mask_dirs,
            output_dirs = mask_dirs,
            uid_list    = uid_list,
            force       = args.force,
        )

    # 2. Run benchmark
    logger.info("Step 2: running benchmark evaluation (%d images × %d models) ...",
                len(df), len(mask_dirs))
    eval_cfg = EvalConfig(n_workers=n_workers, force=args.force)   # all frozen defaults + workers
    out_dir  = frozen_results_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    per_image_dfs = run_benchmark(
        model_mask_dirs = mask_dirs,
        metadata_df     = df,
        eval_cfg        = eval_cfg,
        out_dir         = out_dir,
    )

    # 3 & 4. Aggregate + write frozen results
    for policy in ("or", "and"):
        pf = per_image_dfs.get(policy)
        if pf is None or pf.empty:
            logger.warning("No rows for policy=%s — skipping aggregation.", policy)
            continue
        logger.info("Step 3: aggregating %s policy ...", policy.upper())
        agg = aggregate_benchmark(pf, mode=policy)
        write_frozen_results(agg, frozen_results_dir)
        logger.info("%s policy: overall models=%d rows, by_density rows=%d",
                    policy.upper(), len(agg.overall), len(agg.by_density))

    logger.info("benchmark complete — results in %s", frozen_results_dir)


if __name__ == "__main__":
    main()