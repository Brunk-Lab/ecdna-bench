"""
ecdna_bench.benchmark.registry
================================
Central registry of all models in the benchmark.

Each entry maps a stable short key (used in configs, CLI flags, and DataFrame
columns) to a ``ModelSpec`` that carries the human-readable name, the config
path-key that resolves to the mask directory, and the name of the harmonizer
that must be applied before evaluation (or ``None`` for models whose masks are
already in the canonical binary format).

Keys must be stable across the codebase — the CLI, benchmark runner, figures
module, and CLI all use the same keys.  DO NOT rename them after publication.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = ["ModelSpec", "MODEL_REGISTRY", "MODEL_ORDER", "get_model_spec"]


@dataclass(frozen=True)
class ModelSpec:
    """Specification for one benchmark model.

    Attributes
    ----------
    name:
        Human-readable display name used in paper figures and CSV columns.
    mask_dir_key:
        Key in the paths config (``configs/paths.local.yaml``) whose value
        is the directory containing this model's prediction masks.
        The masks must be in the canonical binary format BEFORE calling
        ``run_benchmark`` — use ``harmonize_all`` first.
    harmonizer:
        Short name of the harmonizer function that must be run to convert
        this model's raw predictions to canonical binary masks.
        ``None`` means the masks are already in canonical format (ecCount
        exports are canonical; Classical pipeline outputs are canonical).
    description:
        Free-form provenance note.
    """
    name:         str
    mask_dir_key: str
    harmonizer:   Optional[str] = None
    description:  str           = ""


# ---------------------------------------------------------------------------
# The canonical six-model registry
# ---------------------------------------------------------------------------
# Keys are stable identifiers used throughout the codebase.
# The display ``name`` must match the labels used in paper Figures 6 exactly.

MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "classical": ModelSpec(
        name         = "Classic (after opt)",
        mask_dir_key = "classical_masks",
        harmonizer   = None,
        description  = "Classical pipeline with per-cell-line Bayesian-optimized parameters.",
    ),
    "classical_before_opt": ModelSpec(
    name         = "Classic (before opt)",
    mask_dir_key = "classical_default_masks",
    harmonizer   = None,
    description  = "Classical pipeline with default parameters, before Bayesian optimization.",
    ),
    "label_engine": ModelSpec(
        name         = "Label Engine",
        mask_dir_key = "label_engine_masks",
        harmonizer   = "label_engine",
        description  = "Label Engine trained on same data/splits; masks from former lab member.",
    ),
    "ecseg": ModelSpec(
        name         = "ecSeg",
        mask_dir_key = "ecseg_masks",
        harmonizer   = "ecseg",
        description  = "ecSeg (Deshpande et al. 2019) with default pre-trained weights.",
    ),
    "mia": ModelSpec(
        name         = "MIA",
        mask_dir_key = "mia_masks",
        harmonizer   = "mia",
        description  = "MIA; masks supplied by MIA authors from their top-performing config.",
    ),
    "eccount_mask": ModelSpec(
        name         = "ecCount (threshold mask)",
        mask_dir_key = "eccount_threshold_masks",
        harmonizer   = None,
        description  = "ecCount binarized probability map (threshold=0.5).",
    ),
    "eccount_peaks": ModelSpec(
        name         = "ecCount (peaks)",
        mask_dir_key = "eccount_peaks_masks",
        harmonizer   = None,
        description  = "ecCount peaks output: disk masks at retained NMS peaks.",
    ),
}

# Canonical display order for all summary tables and figures
MODEL_ORDER: List[str] = [
    "ecCount (peaks)",
    "ecCount (threshold mask)",
    "MIA",
    "Label Engine",
    "Classic (after opt)",
    "Classic (before opt)",   # <-- add this
    "ecSeg",
]

def get_model_spec(key: str) -> ModelSpec:
    """Return the ``ModelSpec`` for *key*, raising a clear error if unknown."""
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model key {key!r}. "
            f"Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key]
