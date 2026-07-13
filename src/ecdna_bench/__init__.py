"""
ecdna-bench — a reference FISH imaging resource and benchmarking framework
for ecDNA localization, segmentation and counting.

This package provides:

  - A unified evaluation framework (Hungarian matching with AND/OR policies,
    TP/FP/FN/ignored bookkeeping, and object/pixel/count metric families).
  - The classical three-stage Bayesian-optimized image-analysis pipeline.
  - ecCount: a compact U-Net with Gaussian soft-target supervision and
    peak-extraction post-processing.
  - Adapters that harmonize prediction masks from external methods
    (ecSeg, MIA, Label Engine) into the shared evaluation format.

Public sub-packages:

    ecdna_bench.data          — dataset metadata, IO, QC, split handling
    ecdna_bench.evaluation    — matching, metrics, sensitivity, statistics
    ecdna_bench.classical     — classical pipeline inference
    ecdna_bench.classical_opt — classical pipeline Bayesian optimization
    ecdna_bench.eccount       — ecCount model, training, inference
    ecdna_bench.baselines     — external-baseline mask harmonizers
    ecdna_bench.benchmark     — cross-model benchmark orchestrator
    ecdna_bench.figures       — paper figure generators
    ecdna_bench.cli           — command-line entry points
    ecdna_bench.utils         — shared helpers (logging, seeding, viz)

Top-level re-exports are kept minimal on purpose. Heavy optional dependencies
(torch, bayesian-optimization, matplotlib) are imported only inside the
sub-packages that need them, so a minimal evaluation-only install can import
`ecdna_bench` and `ecdna_bench.evaluation` without pulling in a CUDA stack.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Brunk Laboratory, University of North Carolina at Chapel Hill"
__license__ = "MIT"

# Curated lightweight re-exports. Anything that requires torch, bayesian-
# optimization, or matplotlib belongs in its sub-package and must be imported
# explicitly from there.
from ecdna_bench.config import Config, load_config  # noqa: E402

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "Config",
    "load_config",
]
