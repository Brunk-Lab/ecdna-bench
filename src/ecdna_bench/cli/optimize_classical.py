"""
ecdna_bench.cli.optimize_classical
=====================================
Run Stage 1 / 2 / 3 Bayesian optimisation for the classical pipeline.

Usage
-----
    python -m ecdna_bench.cli.optimize_classical --stage all --config configs/default.yaml
    python -m ecdna_bench.cli.optimize_classical --stage 1 --cell-line NCIH2170

After Stage 3 completes, automatically calls ``freeze`` to write
``configs/classical/stage3_frozen_params.json``.

Stages
------
1  Exhaustive preprocessing-order search (→ per-cell-line ranking CSV)
2  BO over preprocessing parameters (→ per-cell-line Stage-2 JSON)
3  BO over detection + merge parameters (→ per-cell-line Stage-3 JSON)
   freeze  Merge per-cell-line JSONs → frozen config JSON

Image budgets
-------------
Stage 1 *ranks* 1,050 preprocessing orderings — between-ordering variance
dominates within-ordering image variance, so a small subsample (~25 images)
is statistically sufficient and dramatically faster than using the full
training set.  Stages 2 and 3 *fit* parameters to the data, so they need
the full training cap.  Two separate flags express this:
    --stage1-max-train   (default  25)  - used by Stage 1 only
    --stage23-max-train  (default 100)  - used by Stages 2 and 3
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path
from typing import List, Optional

from ecdna_bench.cli._common import (
    get_path,
    load_config,
    require_file,
    resolve_path,
    setup_logging,
)

logger = logging.getLogger(__name__)

_ALL_CELL_LINES = ["NCI-H2170", "SNU16", "COLO320DM", "NCI-H716"]
_DEFAULT_STAGE1_MAX_TRAIN = 25
_DEFAULT_STAGE23_MAX_TRAIN = 100
_DEFAULT_STAGE1_WORKERS = 4
_DEFAULT_STAGE23_MAX_VAL = None
_DEFAULT_STAGE23_MAX_TEST = None

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Classical pipeline Bayesian optimization.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument(
        "--stage",
        choices=["1", "2", "3", "freeze", "all"],
        default="all",
        help="Which stage(s) to run.",
    )
    p.add_argument("--cell-line", nargs="+", default=None, help="Cell line(s) to optimize.")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument("--force", action="store_true", help="Rerun even if output already exists.")
    p.add_argument("--n-workers", type=int, default=None, help="Worker count for Stage 2/3.")
    p.add_argument(
        "--stage1-workers",
        type=int,
        default=None,
        help="Worker count for Stage 1 only. Defaults to env ECDNA_STAGE1_WORKERS or 4.",
    )
    p.add_argument(
        "--stage1-max-train",
        type=int,
        default=_DEFAULT_STAGE1_MAX_TRAIN,
        help="Max training images per cell line used by Stage 1.",
    )
    p.add_argument(
        "--stage23-max-train",
        type=int,
        default=_DEFAULT_STAGE23_MAX_TRAIN,
        help="Max training images per cell line used by Stages 2 and 3.",
    )
    p.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="DEPRECATED: sets both --stage1-max-train and --stage23-max-train.",
    )
    p.add_argument(
        "--stage23-max-val",
        type=int,
        default=_DEFAULT_STAGE23_MAX_VAL,
        help="Max validation images per cell line used by Stages 2 and 3. Default: no cap.",
    )
    p.add_argument(
        "--stage23-max-test",
        type=int,
        default=_DEFAULT_STAGE23_MAX_TEST,
        help="Max test images per cell line evaluated during Stages 2 and 3. Default: no cap.",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed for training subsampling.")
    return p.parse_args()


def _load_samples(
    consistency_csv: Path,
    cell_line: str,
    split: str,
    max_train: Optional[int] = None,
    max_samples: Optional[int] = None,
    seed: int = 42,
):
    """Load samples for one cell line and split.

    Training can be capped through max_train for backward compatibility.
    Any split can be capped through max_samples.
    """
    import pandas as pd
    from types import SimpleNamespace

    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]
    df = df[df["cell_line"] == cell_line]
    df = df[df["split"] == split]

    samples = []
    for _, row in df.iterrows():
        roi_value = str(row.get("roi_fullpath", "")) if "roi_fullpath" in row else ""
        s = SimpleNamespace(
            uid=str(row["unique_id"]),
            rgb_path=Path(str(row["rgb_fullpath"])),
            roi_path=Path(roi_value) if roi_value not in ("", "nan", "None") else None,
            gt_mask_path=Path(str(row["gt_fullpath"])),
            cell_line=str(row["cell_line"]),
            split=str(row["split"]),
        )
        samples.append(s)

    cap = max_samples
    if split == "train" and max_train is not None:
        cap = max_train if cap is None else min(cap, max_train)

    if cap is not None and len(samples) > cap:
        rng = random.Random(seed)
        original_n = len(samples)
        samples = rng.sample(samples, cap)
        logger.info(
            "_load_samples | %s | %s capped at %d from %d",
            cell_line,
            split,
            cap,
            original_n,
        )

    return samples


def _stage1_len_suffix(cfg: dict) -> str:
    s1_cfg = cfg.get("classical_opt", {}).get("stage1", {})
    min_len = int(s1_cfg.get("min_sequence_length", 3))
    max_len = int(s1_cfg.get("max_sequence_length", 4))
    return f"len{min_len}" if min_len == max_len else f"len{min_len}-{max_len}"


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv = get_path(cfg, "consistency_csv")
    opt_out_dir = get_path(cfg, "classical_opt_out_dir", required=False) or resolve_path(
        cfg.get("paths", {}).get("classical_opt_out_dir", "outputs/classical_opt")
    )
    frozen_json_out = get_path(cfg, "frozen_params_json", required=False) or resolve_path(
        "configs/classical/stage3_frozen_params.json"
    )
    require_file(consistency_csv, "consistency CSV")

    cell_lines: List[str] = list(args.cell_line or _ALL_CELL_LINES)
    stages = ["1", "2", "3", "freeze"] if args.stage == "all" else [args.stage]

    stage1_max_train = args.max_train if args.max_train is not None else args.stage1_max_train
    stage23_max_train = args.max_train if args.max_train is not None else args.stage23_max_train
    seed = int(args.seed)
    stage23_max_val = args.stage23_max_val
    stage23_max_test = args.stage23_max_test
    
    n_workers = int(args.n_workers or cfg.get("classical_opt", {}).get("n_workers", 8))
    stage1_workers = int(
        args.stage1_workers
        or os.environ.get("ECDNA_STAGE1_WORKERS", _DEFAULT_STAGE1_WORKERS)
    )
    stage1_workers = max(1, stage1_workers)

    logger.info("Optimising cell lines : %s", cell_lines)
    logger.info("Stages                : %s", stages)
    logger.info("Stage 1 max train     : %d (seed=%d)", stage1_max_train, seed)
    logger.info("Stage 2/3 max train   : %d (seed=%d)", stage23_max_train, seed)
    logger.info("Stage 1 workers       : %d", stage1_workers)
    logger.info("Stage 2/3 workers     : %d", n_workers)
    logger.info("Stage 1 object cap    : %s", os.environ.get("ECDNA_OPT_MAX_OBJECTS", "5000"))
    logger.info("Stage 1 match-pair cap: %s", os.environ.get("ECDNA_OPT_MAX_MATCH_PAIRS", "5000000"))
    logger.info("Stage 2/3 max val     : %s", stage23_max_val)
    logger.info("Stage 2/3 max test    : %s", stage23_max_test)
    logger.info(
        "Stage 1 tasks/child  : %s",
        os.environ.get("ECDNA_STAGE1_MAX_TASKS_PER_CHILD", "1"),
    )

    from ecdna_bench.classical_opt.stage1_order import Stage1Config, run_stage1
    from ecdna_bench.classical_opt.stage2_preproc import Stage2Config, run_stage2
    from ecdna_bench.classical_opt.stage3_detmerge import Stage3Config, run_stage3
    from ecdna_bench.classical_opt.freeze import freeze_best_params

    partials = opt_out_dir / "partials"
    json_dir = opt_out_dir / "stage3_jsons"

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------
    if "1" in stages:
        logger.info("=== Stage 1: preprocessing order search ===")
        for cl in cell_lines:
            train = _load_samples(
                consistency_csv,
                cl,
                "train",
                max_train=stage1_max_train,
                seed=seed,
            )
            if not train:
                logger.warning("No train samples for %s; skipping Stage 1.", cl)
                continue

            effective_workers = min(stage1_workers, len(train))
            logger.info(
                "Stage 1 | %s | %d train samples | %d workers",
                cl,
                len(train),
                effective_workers,
            )
            s1_cfg_cl = Stage1Config(
                out_dir=opt_out_dir / "stage1",
                partials_dir=partials / "stage1",
                max_workers=effective_workers,
                max_train_images=stage1_max_train,
                seed=seed,
                force=args.force,
            )
            result = run_stage1(train, cl, s1_cfg_cl)
            logger.info(
                "Stage 1 | %s | best_combo=%s f1=%.4f",
                cl,
                result.best_combo,
                result.best_f1,
            )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    if "2" in stages:
        logger.info("=== Stage 2: preprocessing param BO ===")
        s2_cfg = Stage2Config(
            out_dir=opt_out_dir / "stage2",
            partials_dir=partials / "stage2",
            json_dir=opt_out_dir / "stage2_jsons",
            max_workers=n_workers,
            force=args.force,
        )
        len_suffix = _stage1_len_suffix(cfg)
        for cl in cell_lines:
            cl_safe = cl.replace("-", "_")
            rank_csv = opt_out_dir / "stage1" / f"{cl_safe}__stage1_order_{len_suffix}_f1_results.csv"
            if not rank_csv.exists():
                logger.warning(
                    "Stage 1 ranking not found for %s (looked for %s); skipping Stage 2.",
                    cl,
                    rank_csv,
                )
                continue

            import pandas as pd

            ranking = pd.read_csv(rank_csv)
            best_combo_str = str(ranking.iloc[0]["combo_id"])
            combo = tuple(best_combo_str.split("__"))

            train = _load_samples(
                consistency_csv,
                cl,
                "train",
                max_train=stage23_max_train,
                seed=seed,
            )
            val = _load_samples(
                consistency_csv,
                cl,
                "val",
                max_samples=stage23_max_val,
                seed=seed + 1,
            )
            test = _load_samples(
                consistency_csv,
                cl,
                "test",
                max_samples=stage23_max_test,
                seed=seed + 2,
            )
            logger.info("Stage 2 | %s | combo=%s | %d train / %d val samples",
                        cl, combo, len(train), len(val))
            run_stage2(train, val, test, cl, combo, s2_cfg)

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------
    if "3" in stages:
        logger.info("=== Stage 3: detect+merge BO ===")
        s3_cfg = Stage3Config(
            out_dir=opt_out_dir / "stage3",
            partials_dir=partials / "stage3",
            json_dir=json_dir,
            max_workers=n_workers,
            force=args.force,
        )
        import json as json_mod

        for cl in cell_lines:
            cl_safe = cl.replace("-", "_")
            s2_json = opt_out_dir / "stage2_jsons" / f"{cl_safe}.json"
            if not s2_json.exists():
                logger.warning("Stage 2 JSON not found for %s; skipping Stage 3.", cl)
                continue
            with open(s2_json) as f:
                s2_data = json_mod.load(f)
            best_combo_id = next(iter(s2_data))
            payload = s2_data[best_combo_id]
            combo = tuple(payload["combo"])
            preproc = payload["fixed_preproc_params"]
            hsv = payload["fixed_hsv_params"]

            train = _load_samples(
                consistency_csv,
                cl,
                "train",
                max_train=stage23_max_train,
                seed=seed,
            )
            val = _load_samples(
                consistency_csv,
                cl,
                "val",
                max_samples=stage23_max_val,
                seed=seed + 1,
            )
            test = _load_samples(
                consistency_csv,
                cl,
                "test",
                max_samples=stage23_max_test,
                seed=seed + 2,
            )
            logger.info("Stage 3 | %s | combo=%s | %d train / %d val samples",
                        cl, combo, len(train), len(val))
            run_stage3(train, val, test, cl, combo, preproc, hsv, s3_cfg)

    # ------------------------------------------------------------------
    # Freeze
    # ------------------------------------------------------------------
    if "freeze" in stages:
        logger.info("=== Freeze: writing frozen params JSON ===")
        if not json_dir.exists():
            logger.error("Stage-3 JSON dir not found: %s", json_dir)
            return
        frozen = freeze_best_params(json_dir, frozen_json_out, cell_lines=cell_lines)
        logger.info("Frozen params written to %s (%d cell lines)", frozen_json_out, len(frozen))

    logger.info("optimize_classical complete.")


if __name__ == "__main__":
    main()