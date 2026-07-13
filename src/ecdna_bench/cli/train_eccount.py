"""
ecdna_bench.cli.train_eccount
================================
Train the ecCount U-Net.

Usage
-----
    python -m ecdna_bench.cli.train_eccount --config configs/default.yaml
    python -m ecdna_bench.cli.train_eccount --config configs/default.yaml --resume auto

NOTE: There is no separate ``configs/eccount.yaml``.  All ecCount
parameters live under the ``eccount:`` block in ``configs/default.yaml``.

Config keys read
----------------
All training hyperparameters are nested under ``eccount.train.*``:

    eccount:
      input_size: [1024, 1224]          → image_h / image_w
      targets:
        sigma: 1.0                      → target sigma
      loss:
        pos_weight: 20.0
        bce_weight: 1.0
        dice_weight: 1.0
        dice_smooth: 1.0e-6
      augmentation:
        hflip_p: 0.5
        vflip_p: 0.5
        brightness_p: 0.2
        brightness_range: [0.9, 1.1]
      train:
        batch_size: 2
        num_workers: 4
        lr: 1.0e-4
        weight_decay: 0.0
        epochs: 70
        scheduler: plateau
        scheduler_patience: 5
        scheduler_factor: 0.5
        seed: 42
        amp: false          ← paper results are fp32; do NOT set true

Path keys (resolved from ``paths:`` section in paths.local.yaml):
    paths:
      consistency_csv: ...
      eccount_out_dir: ...   (optional; default: outputs/eccount_training/)

After training, add to paths.local.yaml:
    paths:
      eccount_checkpoint: <eccount_out_dir>/best_model.pt
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
        description="Train the ecCount U-Net.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--config", type=Path,
        # Default is default.yaml — there is no eccount.yaml in this repo.
        default=Path("configs/default.yaml"),
        help="Path to YAML config.  Use configs/default.yaml (the only config file).",
    )
    p.add_argument(
        "--resume", default=None,
        help="Override resume behaviour: 'auto', 'never', or an explicit checkpoint path.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv = get_path(cfg, "consistency_csv")
    out_dir         = get_path(cfg, "eccount_out_dir", required=False) or \
                      resolve_path("outputs/eccount_training")
    require_file(consistency_csv, "consistency CSV")

    # ------------------------------------------------------------------
    # Read all ecCount config blocks at the correct YAML nesting level.
    #
    # Hierarchy: cfg["eccount"]["train"]["lr"]   ← correct
    #            cfg["eccount"]["lr"]             ← WRONG (one level too high)
    # ------------------------------------------------------------------
    eccount_cfg = cfg.get("eccount", {})
    train_sub   = eccount_cfg.get("train",       {})
    target_sub  = eccount_cfg.get("targets",     {})
    loss_sub    = eccount_cfg.get("loss",         {})
    aug_sub     = eccount_cfg.get("augmentation", {})
    input_size  = eccount_cfg.get("input_size",  [1024, 1224])

    image_h = int(input_size[0])
    image_w = int(input_size[1])

    # Training hyperparameters — from eccount.train.*
    max_epochs   = int(train_sub.get("epochs",       70))
    batch_size   = int(train_sub.get("batch_size",   2))
    lr           = float(train_sub.get("lr",         1e-4))
    weight_decay = float(train_sub.get("weight_decay", 0.0))
    device       = str(train_sub.get("device",       "cuda"))
    n_workers    = int(train_sub.get("num_workers",
                       train_sub.get("n_workers",    4)))
    seed         = int(train_sub.get("seed",         42))
    # IMPORTANT: amp defaults to False in the YAML (paper used fp32).
    # TrainConfig defaults to use_amp=True, so we MUST read this from YAML.
    use_amp      = bool(train_sub.get("amp",         False))
    scheduler    = str(train_sub.get("scheduler",    "plateau"))
    sched_patience = int(train_sub.get("scheduler_patience", 5))
    sched_factor   = float(train_sub.get("scheduler_factor", 0.5))

    # Soft-target sigma — from eccount.targets.*
    sigma = float(target_sub.get("sigma", 1.0))

    # Loss — from eccount.loss.*
    from ecdna_bench.eccount.losses import LossConfig
    loss_cfg = LossConfig(
        pos_weight  = float(loss_sub.get("pos_weight",  20.0)),
        bce_weight  = float(loss_sub.get("bce_weight",  1.0)),
        dice_weight = float(loss_sub.get("dice_weight", 1.0)),
        smooth      = float(loss_sub.get("dice_smooth",
                            loss_sub.get("smooth",      1e-6))),
    )

    from ecdna_bench.eccount.train import TrainConfig
    from ecdna_bench.eccount.model import ModelConfig

    train_cfg = TrainConfig(
        out_dir          = out_dir,
        max_epochs       = max_epochs,
        batch_size       = batch_size,
        lr               = lr,
        weight_decay     = weight_decay,
        image_h          = image_h,
        image_w          = image_w,
        sigma            = sigma,
        device           = device,
        num_workers      = n_workers,
        seed             = seed,
        use_amp          = use_amp,
        scheduler        = scheduler,
        plateau_patience = sched_patience,
        plateau_factor   = sched_factor,
        model_cfg        = ModelConfig(),
        loss_cfg         = loss_cfg,
        resume           = args.resume if args.resume is not None
                           else str(train_sub.get("resume", "auto")),
    )

    logger.info("Training ecCount:")
    logger.info("  epochs=%d  batch=%d  lr=%.0e  device=%s  amp=%s",
                train_cfg.max_epochs, train_cfg.batch_size,
                train_cfg.lr, train_cfg.device, train_cfg.use_amp)
    logger.info("  sigma=%.1f  scheduler=%s (patience=%d factor=%.1f)  seed=%d",
                train_cfg.sigma, train_cfg.scheduler,
                train_cfg.plateau_patience, train_cfg.plateau_factor, train_cfg.seed)
    logger.info("  out_dir: %s", out_dir)

    # ------------------------------------------------------------------
    # Build datasets
    # ------------------------------------------------------------------
    import pandas as pd
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]

    from ecdna_bench.eccount.dataset import EcCountDataset, DatasetConfig
    from ecdna_bench.eccount.targets import SoftTargetConfig

    target_cfg = SoftTargetConfig(sigma=sigma)

    ds_train_cfg = DatasetConfig(
        training               = True,
        hflip_prob             = float(aug_sub.get("hflip_p",        0.5)),
        vflip_prob             = float(aug_sub.get("vflip_p",        0.5)),
        brightness_jitter_prob = float(aug_sub.get("brightness_p",  0.2)),
        brightness_jitter_range= tuple(aug_sub.get("brightness_range", (0.9, 1.1))),
    )
    ds_val_cfg = DatasetConfig(training=False)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df   = df[df["split"] == "val"  ].reset_index(drop=True)

    logger.info("Train: %d images | Val: %d images", len(train_df), len(val_df))

    train_ds = EcCountDataset(train_df, ds_train_cfg, target_cfg)
    val_ds   = EcCountDataset(val_df,   ds_val_cfg,   target_cfg)

    # ------------------------------------------------------------------
    # Run training
    # ------------------------------------------------------------------
    from ecdna_bench.eccount.train import train_eccount
    result = train_eccount(train_cfg, train_ds, val_ds)

    logger.info("Training complete.")
    logger.info("  Best epoch:    %d",    result.best_epoch)
    logger.info("  Best val loss: %.4f",  result.best_val_loss)
    logger.info("  Best ckpt:     %s",    result.best_ckpt)
    logger.info(
        "\nNext step — add this to configs/paths.local.yaml:\n"
        "  eccount_checkpoint: %s",
        result.best_ckpt,
    )


if __name__ == "__main__":
    main()