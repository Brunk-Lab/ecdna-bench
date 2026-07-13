"""
ecdna_bench.eccount.tuning.loss
================================
Grid search over loss hyperparameters.

9-config grid (paper §13b):
    pos_weight ∈ {10, 20, 50}  × bce/dice balance ∈ {BCE-only, Dice-only, equal}

Frozen paper values: pos_weight=20, bce_weight=1.0, dice_weight=1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

__all__ = ["LOSS_GRID", "LossRunResult", "aggregate_loss_results"]

# Each config is (pos_weight, bce_weight, dice_weight)
LOSS_GRID: List[Tuple[float, float, float]] = [
    (10.0,  1.0,  0.0),   # BCE only, pw=10
    (10.0,  0.0,  1.0),   # Dice only, pw=10
    (10.0,  1.0,  1.0),   # Equal, pw=10
    (20.0,  1.0,  0.0),   # BCE only, pw=20
    (20.0,  0.0,  1.0),   # Dice only, pw=20
    (20.0,  1.0,  1.0),   # Equal, pw=20  ← paper value
    (50.0,  1.0,  0.0),   # BCE only, pw=50
    (50.0,  0.0,  1.0),   # Dice only, pw=50
    (50.0,  1.0,  1.0),   # Equal, pw=50
]

# Frozen paper config index in the grid
_PAPER_CONFIG: Tuple[float, float, float] = (20.0, 1.0, 1.0)


@dataclass
class LossRunResult:
    """Result of one loss-config training run."""
    pos_weight:       float
    bce_weight:       float
    dice_weight:      float
    best_val_loss:    float
    best_epoch:       int
    final_train_loss: float
    checkpoint_path:  Optional[str] = None
    extra:            Optional[dict] = None


def aggregate_loss_results(results: List[LossRunResult]) -> pd.DataFrame:
    """Collect loss grid results into a tidy DataFrame sorted by val_loss."""
    rows = [
        {
            "pos_weight":        r.pos_weight,
            "bce_weight":        r.bce_weight,
            "dice_weight":       r.dice_weight,
            "config_id":         f"pw{int(r.pos_weight)}_bce{r.bce_weight:.0f}_dice{r.dice_weight:.0f}",
            "best_val_loss":     r.best_val_loss,
            "best_epoch":        r.best_epoch,
            "final_train_loss":  r.final_train_loss,
            "checkpoint_path":   r.checkpoint_path or "",
            "is_paper_value":    (r.pos_weight, r.bce_weight, r.dice_weight) == _PAPER_CONFIG,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_loss").reset_index(drop=True)
    return df
