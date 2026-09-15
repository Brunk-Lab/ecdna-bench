"""
ecdna_bench.eccount.train
==========================
Resumable training loop for ecCount.

Frozen defaults (from §2 of REWRITE_PLAN.md)
--------------------------------------------
* Adam lr=1e-4, weight_decay=0, batch_size=2, epochs=70
* ReduceLROnPlateau(patience=5, factor=0.5)
* Best checkpoint of the released model: epoch 49, val_loss 0.6421
* Augmentation: hflip 0.5, vflip 0.5, brightness [0.9,1.1] @ p=0.2

Design rules
------------
* Pure library API — no argparse, no hard-coded paths.
* All hyperparameters are in ``TrainConfig`` (with frozen defaults).
* Resumable: ``train_eccount`` looks for ``last_model.pt`` in
  ``out_dir`` on startup and resumes automatically unless
  ``resume="never"``.
* Saves two checkpoints each epoch: ``last_model.pt`` (always) and
  ``best_model.pt`` (when val loss improves).
* Writes ``train_history.csv`` (one row per epoch).
* SIGTERM/SIGINT → safe checkpoint + clean exit.
"""

from __future__ import annotations

import csv
import logging
import os
import random
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ecdna_bench.eccount.dataset import DatasetConfig, EcCountDataset
from ecdna_bench.eccount.losses import LossConfig, build_loss
from ecdna_bench.eccount.model import ModelConfig, build_model
from ecdna_bench.eccount.targets import SoftTargetConfig

logger = logging.getLogger(__name__)

__all__ = ["TrainConfig", "TrainResult", "train_eccount"]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    """All ecCount training hyperparameters.

    All defaults are the frozen published values.
    """
    out_dir:    Path

    # Schedule
    max_epochs:   int   = 70       # frozen
    batch_size:   int   = 2        # frozen
    lr:           float = 1e-4     # frozen
    weight_decay: float = 0.0      # frozen

    # Image size
    image_h: int = 1024            # frozen
    image_w: int = 1224            # frozen

    # Target
    sigma:        float = 1.0      # frozen

    # Model (frozen architecture)
    model_cfg:  ModelConfig   = field(default_factory=ModelConfig)
    loss_cfg:   LossConfig    = field(default_factory=LossConfig)

    # Augmentation (frozen)
    hflip_prob:             float = 0.5
    vflip_prob:             float = 0.5
    brightness_jitter_prob: float = 0.2
    brightness_jitter_range: tuple = (0.9, 1.1)

    # Scheduler
    scheduler:         str   = "plateau"   # "none" or "plateau"
    plateau_factor:    float = 0.5
    plateau_patience:  int   = 5
    plateau_min_lr:    float = 1e-6

    # Runtime
    device:       str = "cuda"
    num_workers:  int = 4
    seed:         int = 42
    use_amp:      bool = True
    grad_clip_norm: float = 0.0
    save_every_epochs: int = 1

    # Resume behaviour: "auto" | "never" | explicit path string
    resume: str = "auto"


@dataclass
class TrainResult:
    """Return value of ``train_eccount``."""
    best_val_loss:  float
    best_epoch:     int
    history_csv:    Path
    best_ckpt:      Path
    last_ckpt:      Path
    epochs_done:    int


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_STOP_REQUESTED: bool = False


def _signal_handler(signum, frame) -> None:
    global _STOP_REQUESTED
    logger.warning("Signal %d received — will save checkpoint and exit cleanly.", signum)
    _STOP_REQUESTED = True


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _atomic_save(obj: Dict[str, Any], path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _append_history(csv_path: Path, row: Dict[str, Any]) -> None:
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _save_checkpoint(
    path: Path,
    *,
    model:           torch.nn.Module,
    optimizer:       torch.optim.Optimizer,
    scaler:          Optional[Any],
    scheduler:       Optional[Any],
    epoch_completed: int,
    global_step:     int,
    best_val_loss:   float,
    best_epoch:      int,
    run_cfg:         Dict[str, Any],
) -> None:
    payload: Dict[str, Any] = {
        "epoch_completed":      int(epoch_completed),
        "global_step":          int(global_step),
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss":        float(best_val_loss),
        "best_epoch":           int(best_epoch),
        "run_config":           run_cfg,
        "saved_at_unix":        time.time(),
    }
    if scaler is not None:
        payload["scaler_state_dict"] = scaler.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    _atomic_save(payload, path)


def _try_resume(
    out_dir: Path,
    device:  torch.device,
    resume:  str,
) -> Optional[Dict[str, Any]]:
    """Load a checkpoint if resuming; return None to start fresh."""
    resume = str(resume).strip()
    last   = out_dir / "last_model.pt"

    if resume.lower() == "never":
        logger.info("Resume disabled.")
        return None

    candidate = last if resume.lower() == "auto" else Path(resume)
    if not candidate.exists():
        logger.info("No checkpoint found at %s. Starting fresh.", candidate)
        return None

    try:
        ckpt = torch.load(candidate, map_location=device)
        logger.info("Loaded checkpoint: epoch %d", ckpt.get("epoch_completed", "?"))
        return ckpt
    except Exception as exc:
        logger.warning("Could not load checkpoint %s: %s. Starting fresh.", candidate, exc)
        return None


# ---------------------------------------------------------------------------
# Training and validation passes
# ---------------------------------------------------------------------------

def _run_epoch(
    model:    torch.nn.Module,
    loader:   DataLoader,
    loss_fn:  torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scaler:   Optional[Any],
    device:   torch.device,
    use_amp:  bool,
    grad_clip: float,
    train:    bool,
) -> float:
    model.train(train)
    total_loss = 0.0
    n_batches  = 0

    ctx = torch.no_grad() if not train else torch.enable_grad()
    with ctx:
        for batch in loader:
            if _STOP_REQUESTED:
                break
            x = batch["image"].to(device, non_blocking=True)
            y = batch["target"].to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                from torch.cuda.amp import autocast
                with autocast():
                    logits = model(x)
                    loss   = loss_fn(logits, y)
            else:
                logits = model(x)
                loss   = loss_fn(logits, y)

            if train and optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()

            total_loss += float(loss.item())
            n_batches  += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train_eccount(
    cfg:           TrainConfig,
    train_dataset: EcCountDataset,
    val_dataset:   EcCountDataset,
) -> TrainResult:
    """Train ecCount with the given datasets.

    Parameters
    ----------
    cfg:
        ``TrainConfig`` (all frozen defaults set).
    train_dataset, val_dataset:
        Pre-built ``EcCountDataset`` objects.

    Returns
    -------
    TrainResult
    """
    global _STOP_REQUESTED
    _STOP_REQUESTED = False

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT,  _signal_handler)

    _set_seed(cfg.seed)

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    history_csv  = out_dir / "train_history.csv"
    last_ckpt    = out_dir / "last_model.pt"
    best_ckpt    = out_dir / "best_model.pt"

    device_str = cfg.device
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available — falling back to CPU.")
        device_str = "cpu"
    device  = torch.device(device_str)
    use_amp = cfg.use_amp and device.type == "cuda"

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    # Build model + optimizer + loss
    model    = build_model(cfg.model_cfg).to(device)
    loss_fn  = build_loss(cfg.loss_cfg).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scaler: Optional[Any] = None
    if use_amp:
        from torch.cuda.amp import GradScaler
        scaler = GradScaler()

    scheduler: Optional[Any] = None
    if cfg.scheduler.lower() == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=cfg.plateau_factor,
            patience=cfg.plateau_patience,
            min_lr=cfg.plateau_min_lr,
        )

    logger.info("Model: %d trainable parameters", _count_params(model))

    # Build run config snapshot for checkpoint compatibility
    run_cfg: Dict[str, Any] = {
        "max_epochs":   cfg.max_epochs,
        "batch_size":   cfg.batch_size,
        "image_h":      cfg.image_h,
        "image_w":      cfg.image_w,
        "model_cfg":    asdict(cfg.model_cfg),
        "loss_cfg":     asdict(cfg.loss_cfg),
        "sigma":        cfg.sigma,
        "seed":         cfg.seed,
    }

    # Try resume
    start_epoch    = 0
    global_step    = 0
    best_val_loss  = float("inf")
    best_epoch     = -1

    ckpt = _try_resume(out_dir, device, cfg.resume)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch   = int(ckpt.get("epoch_completed", 0))
        global_step   = int(ckpt.get("global_step", 0))
        best_val_loss = float(ckpt.get("best_val_loss", float("inf")))
        best_epoch    = int(ckpt.get("best_epoch", -1))
        if scaler is not None and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        logger.info("Resumed from epoch %d (best_val_loss=%.4f)", start_epoch, best_val_loss)

    # Data loaders
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=True,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0), drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=pin,
        persistent_workers=(cfg.num_workers > 0), drop_last=False,
    )

    # Training loop
    for epoch in range(start_epoch, cfg.max_epochs):
        if _STOP_REQUESTED:
            logger.info("Stop requested before epoch %d. Saving checkpoint.", epoch)
            break

        t0         = time.time()
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, scaler,
                                device, use_amp, cfg.grad_clip_norm, train=True)
        val_loss   = _run_epoch(model, val_loader, loss_fn, None, None,
                                device, False, 0.0, train=False)
        elapsed    = time.time() - t0

        global_step += len(train_loader)

        if scheduler is not None:
            scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch + 1

        cur_lr = float(optimizer.param_groups[0]["lr"])
        row = {
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "lr":         cur_lr,
            "best_val_loss": round(best_val_loss, 6),
            "best_epoch": best_epoch,
            "elapsed_s":  round(elapsed, 1),
        }
        _append_history(history_csv, row)
        logger.info(
            "Epoch %3d/%d | train=%.4f | val=%.4f | lr=%.2e | best=%.4f@%d%s",
            epoch + 1, cfg.max_epochs, train_loss, val_loss, cur_lr,
            best_val_loss, best_epoch, " ★" if improved else "",
        )

        ckpt_payload = dict(
            model=model, optimizer=optimizer, scaler=scaler, scheduler=scheduler,
            epoch_completed=epoch + 1, global_step=global_step,
            best_val_loss=best_val_loss, best_epoch=best_epoch, run_cfg=run_cfg,
        )

        if improved:
            _save_checkpoint(best_ckpt, **ckpt_payload)

        if (epoch + 1) % cfg.save_every_epochs == 0 or _STOP_REQUESTED:
            _save_checkpoint(last_ckpt, **ckpt_payload)

    return TrainResult(
        best_val_loss = best_val_loss,
        best_epoch    = best_epoch,
        history_csv   = history_csv,
        best_ckpt     = best_ckpt,
        last_ckpt     = last_ckpt,
        epochs_done   = min(cfg.max_epochs, start_epoch + cfg.max_epochs - start_epoch),
    )