"""
ecdna_bench.eccount.losses
===========================
Loss functions for ecCount training.

Frozen defaults (from §2 of REWRITE_PLAN.md)
--------------------------------------------
* pos_weight  = 20.0
* bce_weight  = 1.0
* dice_weight = 1.0
* smooth      = 1e-6
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LossConfig", "SoftDiceLoss", "WeightedBCEDiceLoss", "build_loss"]


@dataclass
class LossConfig:
    """Configuration for the ecCount combined loss.

    All defaults match the frozen paper values.
    """
    bce_weight:  float = 1.0
    dice_weight: float = 1.0
    pos_weight:  float = 20.0
    smooth:      float = 1e-6


class SoftDiceLoss(nn.Module):
    """Soft Dice loss operating on continuous targets in [0, 1].

    Applies sigmoid to logits internally.
    """

    def __init__(self, smooth: float = 1e-6) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {smooth}")
        self.smooth = float(smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
            )
        targets = targets.float()
        probs   = torch.sigmoid(logits)

        # Flatten spatial + batch for per-batch Dice
        probs_f   = probs.reshape(probs.shape[0], -1)
        targets_f = targets.reshape(targets.shape[0], -1)

        inter = (probs_f * targets_f).sum(dim=1)
        denom = probs_f.sum(dim=1) + targets_f.sum(dim=1)
        dice  = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return (1.0 - dice).mean()


class WeightedBCEDiceLoss(nn.Module):
    """Weighted BCE-with-logits + Soft Dice loss.

    This is the canonical ecCount training loss.
    """

    def __init__(self, cfg: LossConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or LossConfig()
        c = self.cfg
        if c.bce_weight < 0:
            raise ValueError(f"bce_weight must be >= 0, got {c.bce_weight}")
        if c.dice_weight < 0:
            raise ValueError(f"dice_weight must be >= 0, got {c.dice_weight}")
        if c.pos_weight <= 0:
            raise ValueError(f"pos_weight must be > 0, got {c.pos_weight}")
        if c.smooth <= 0:
            raise ValueError(f"smooth must be > 0, got {c.smooth}")
        if c.bce_weight == 0 and c.dice_weight == 0:
            raise ValueError("At least one of bce_weight or dice_weight must be > 0.")

        self.register_buffer(
            "pos_weight_tensor",
            torch.tensor(float(c.pos_weight), dtype=torch.float32),
        )
        self.dice = SoftDiceLoss(smooth=c.smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape != targets.shape:
            raise ValueError(
                f"Shape mismatch: logits {tuple(logits.shape)} vs targets {tuple(targets.shape)}"
            )
        targets = targets.float()
        total   = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)

        if self.cfg.bce_weight > 0:
            bce = F.binary_cross_entropy_with_logits(
                logits, targets,
                pos_weight=self.pos_weight_tensor.to(device=logits.device, dtype=logits.dtype),
            )
            total = total + self.cfg.bce_weight * bce

        if self.cfg.dice_weight > 0:
            total = total + self.cfg.dice_weight * self.dice(logits, targets)

        return total


def build_loss(cfg: LossConfig | None = None) -> nn.Module:
    """Build the canonical ecCount loss module."""
    return WeightedBCEDiceLoss(cfg or LossConfig())
