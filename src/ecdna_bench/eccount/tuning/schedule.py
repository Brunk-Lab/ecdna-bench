"""
ecdna_bench.eccount.tuning.schedule
=====================================
Grid search over optimizer / learning-rate schedule combinations.

6-config grid (paper §13c):
    optimizer ∈ {Adam, AdamW}  × scheduler ∈ {none, plateau, cosine}

Frozen paper values: Adam, ReduceLROnPlateau(patience=5, factor=0.5).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

__all__ = ["SCHEDULE_GRID", "ScheduleRunResult", "aggregate_schedule_results"]

# Each config: (optimizer_name, scheduler_name, extra_kwargs)
SCHEDULE_GRID: List[Dict] = [
    {"optimizer": "adam",  "scheduler": "none"},
    {"optimizer": "adam",  "scheduler": "plateau",  "plateau_patience": 5, "plateau_factor": 0.5},
    {"optimizer": "adam",  "scheduler": "cosine"},
    {"optimizer": "adamw", "scheduler": "none"},
    {"optimizer": "adamw", "scheduler": "plateau",  "plateau_patience": 5, "plateau_factor": 0.5},
    {"optimizer": "adamw", "scheduler": "cosine"},
]

# Frozen paper value
_PAPER_CONFIG: Tuple[str, str] = ("adam", "plateau")


@dataclass
class ScheduleRunResult:
    """Result of one schedule-config training run."""
    optimizer:        str
    scheduler:        str
    best_val_loss:    float
    best_epoch:       int
    final_train_loss: float
    checkpoint_path:  Optional[str] = None
    extra:            Optional[dict] = None


def aggregate_schedule_results(results: List[ScheduleRunResult]) -> pd.DataFrame:
    """Collect schedule grid results into a tidy DataFrame sorted by val_loss."""
    rows = [
        {
            "optimizer":         r.optimizer,
            "scheduler":         r.scheduler,
            "config_id":         f"{r.optimizer}_{r.scheduler}",
            "best_val_loss":     r.best_val_loss,
            "best_epoch":        r.best_epoch,
            "final_train_loss":  r.final_train_loss,
            "checkpoint_path":   r.checkpoint_path or "",
            "is_paper_value":    (r.optimizer, r.scheduler) == _PAPER_CONFIG,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_loss").reset_index(drop=True)
    return df
