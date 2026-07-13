"""
ecdna_bench.classical_opt
==========================
Three-stage Bayesian optimisation pipeline for the classical ecDNA pipeline.

Stage 1 — exhaustive preprocessing-order search
    ``stage1_order.run_stage1(samples, cell_line, cfg)``
    Enumerates all permutations of the 7 preprocessing ops (length 3–6),
    evaluates each on the training split with fixed detection/match params,
    and ranks combos by micro-F1.  Writes per-combo partial CSVs for
    idempotent resume and a per-cell-line ranking CSV.

Stage 2 — BO over preprocessing parameters
    ``stage2_preproc.run_stage2(samples, cell_line, combo, cfg)``
    Fixes the combo from Stage 1 and runs Bayesian optimisation (via
    ``bayesian-optimization``) over the scalar parameters of each op plus
    the two HSV thresholds.  Train F1 is the objective; test F1 is tracked
    each iteration.  Resumes from partial CSV.

Stage 3 — BO over detection + merge parameters
    ``stage3_detmerge.run_stage3(samples, cell_line, combo, preproc_params, cfg)``
    Fixes preprocessing from Stage 2 and runs BO over
    {threshold_factor, morph_close_kernel, min_area, max_area, merge_distance}.
    Writes a per-cell-line JSON that feeds into ``freeze``.

Freeze — commit best parameters
    ``freeze.freeze_best_params(stage3_json_dir, output_path)``
    Reads the per-cell-line Stage-3 JSONs, picks the best combo per cell line,
    and writes ``configs/classical/stage3_frozen_params.json``.
"""

from importlib import import_module as _imp

__all__ = [
    "run_stage1",
    "run_stage2",
    "run_stage3",
    "freeze_best_params",
    "Stage1Config",
    "Stage2Config",
    "Stage3Config",
]


def __getattr__(name: str):
    _map = {
        "run_stage1":    "stage1_order",
        "Stage1Config":  "stage1_order",
        "run_stage2":    "stage2_preproc",
        "Stage2Config":  "stage2_preproc",
        "run_stage3":    "stage3_detmerge",
        "Stage3Config":  "stage3_detmerge",
        "freeze_best_params": "freeze",
    }
    if name in _map:
        mod = _imp(f"ecdna_bench.classical_opt.{_map[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'ecdna_bench.classical_opt' has no attribute {name!r}")
