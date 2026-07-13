"""
ecdna_bench.eccount.tuning.sigma
=================================
Ablation over target Gaussian σ values.

Sweeps σ ∈ {0.5, 0.75, 1.0, 1.5, 2.0, 3.0} and returns a tidy DataFrame
with one row per configuration.  The caller trains or evaluates the model
for each σ; this module only defines the grid and result aggregation.

Design rules
------------
* Pure library — no argparse, no hard-coded paths.
* Each σ run produces one ``SigmaRunResult``; ``aggregate_sigma_results``
  collects them into a DataFrame sorted by val_loss.
* The frozen paper value is σ = 1.0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

__all__ = [
    "SIGMA_GRID",
    "SigmaRunResult",
    "aggregate_sigma_results",
]

# Frozen sigma sweep grid (paper §13a)
SIGMA_GRID: List[float] = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]

# Frozen paper value
SIGMA_PAPER: float = 1.0


@dataclass
class SigmaRunResult:
    """Result of one σ training run.

    Attributes
    ----------
    sigma:
        The σ value used.
    best_val_loss:
        Best validation loss achieved.
    best_epoch:
        Epoch at which best_val_loss was reached.
    final_train_loss:
        Training loss at the last epoch.
    checkpoint_path:
        Path to the saved ``best_model.pt``.
    extra:
        Optional dict of additional per-epoch metrics.
    """
    sigma:            float
    best_val_loss:    float
    best_epoch:       int
    final_train_loss: float
    checkpoint_path:  Optional[str] = None
    extra:            Optional[dict] = None


def aggregate_sigma_results(results: List[SigmaRunResult]) -> pd.DataFrame:
    """Collect sigma ablation results into a tidy DataFrame.

    Parameters
    ----------
    results:
        List of ``SigmaRunResult`` objects, one per σ value.

    Returns
    -------
    pd.DataFrame
        Columns: sigma, best_val_loss, best_epoch, final_train_loss,
        checkpoint_path.  Sorted by best_val_loss ascending.
    """
    rows = [
        {
            "sigma":             r.sigma,
            "best_val_loss":     r.best_val_loss,
            "best_epoch":        r.best_epoch,
            "final_train_loss":  r.final_train_loss,
            "checkpoint_path":   r.checkpoint_path or "",
            "is_paper_value":    r.sigma == SIGMA_PAPER,
        }
        for r in results
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("best_val_loss").reset_index(drop=True)
    return df
