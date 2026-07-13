"""
ecdna_bench.benchmark
======================
Cross-model benchmark orchestrator.

Sub-modules
-----------
registry
    ``MODEL_REGISTRY`` dict and ``ModelSpec`` frozen dataclass.
harmonize
    ``harmonize_all`` — call the appropriate ``baselines/`` harmonizer
    for each model that requires it, validate UID coverage.
run
    ``run_benchmark`` — per-image evaluation across all models × OR/AND.
    Pure library; no argparse; no global state.
aggregate
    ``aggregate_benchmark`` / ``write_frozen_results`` — produce the
    group-level summary CSVs that feed the paper figures.
"""

from importlib import import_module as _imp

__all__ = [
    "MODEL_REGISTRY",
    "ModelSpec",
    "harmonize_all",
    "EvalConfig",
    "run_benchmark",
    "AggregateResult",
    "aggregate_benchmark",
    "write_frozen_results",
]


def __getattr__(name: str):
    _map = {
        "MODEL_REGISTRY":      "registry",
        "ModelSpec":           "registry",
        "harmonize_all":       "harmonize",
        "EvalConfig":          "run",
        "run_benchmark":       "run",
        "AggregateResult":     "aggregate",
        "aggregate_benchmark": "aggregate",
        "write_frozen_results": "aggregate",
    }
    if name in _map:
        mod = _imp(f"ecdna_bench.benchmark.{_map[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'ecdna_bench.benchmark' has no attribute {name!r}")
