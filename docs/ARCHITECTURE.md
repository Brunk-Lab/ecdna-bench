# Architecture Guide

This document describes the module structure of `ecdna_bench`, the data flow
through the pipeline, and the key design decisions.

---

## Module map

```
src/ecdna_bench/
│
├── config.py                YAML loader + schema validation
│
├── data/                    ── Data layer ──────────────────────────────────
│   ├── ids.py               UID normalization, split file management
│   ├── metadata.py          Build + validate metadata.csv
│   ├── io.py                load_rgb / load_mask / load_roi / save_mask
│   ├── qc.py                File audit + count-mask consistency check
│   └── samples.py           Sample dataclass + filtering
│
├── evaluation/              ── ★ Key contribution ───────────────────────────
│   ├── objects.py           CC extraction (8-conn, min area 3)
│   ├── matching.py          Hungarian + OR/AND policy + ignored bucket
│   ├── metrics.py           Object / pixel / count metric families
│   ├── sensitivity.py       d_max × IoU_min × policy sweep
│   ├── stats.py             Wilcoxon, Friedman, Benjamini–Hochberg
│   └── density.py           Density-bin stratification
│
├── classical/               ── Classical pipeline (inference) ───────────────
│   ├── preprocess.py        7 atomic operators + registry + chain runner
│   ├── detect.py            Otsu + CC + merge + HSV classifier
│   ├── postprocess.py       Eval-time split heuristic
│   └── infer.py             End-to-end per-image inference
│
├── classical_opt/           ── Classical pipeline (optimization) ────────────
│   ├── stage1_order.py      Exhaustive preprocessing-order search
│   ├── stage2_preproc.py    BO over preprocessing parameters
│   ├── stage3_detmerge.py   BO over detection + merge parameters
│   └── freeze.py            Pick best per cell line → frozen JSON
│
├── eccount/                 ── ecCount (novel contribution) ──────────────────
│   ├── targets.py           Soft Gaussian target generation
│   ├── dataset.py           PyTorch Dataset + synchronized augmentation
│   ├── model.py             Compact U-Net (7,849,601 params)
│   ├── losses.py            wBCE + SoftDice
│   ├── train.py             Resumable training loop
│   ├── postprocess.py       Peak extraction: smooth → ROI → maxima → NMS
│   ├── infer.py             End-to-end inference → 2 output modes
│   └── tuning/              σ / loss / schedule / postprocess ablations
│
├── baselines/               ── External model harmonizers ────────────────────
│   ├── ecseg.py             ecSeg channel-3 extraction → binary mask
│   ├── mia.py               MIA binary normalization
│   └── label_engine.py      Label Engine canonicalization + threshold
│
├── benchmark/               ── Cross-model orchestrator ─────────────────────
│   ├── registry.py          6-model registry (matches paper Fig. 6 order)
│   ├── harmonize.py         Call harmonizers + validate UID coverage
│   ├── run.py               Per-image evaluation × all models × OR/AND
│   └── aggregate.py         Group summaries + frozen-results writer
│
├── figures/                 ── Paper figure generators ──────────────────────
│   ├── common.py            Style, palette, save helpers, data loaders
│   ├── fig1_dataset.py …    Main figures 1–6
│   └── supp/                Supplementary figures sfig4–sfig7
│
├── cli/                     ── Entry points ─────────────────────────────────
│   ├── _common.py           load_config, get_path, require_file
│   ├── build_metadata.py … 10 thin argparse + dispatch modules
│
└── utils/                   ── Shared utilities ──────────────────────────────
    ├── logging.py           Consistent logger setup
    ├── seeding.py           Seed torch/numpy/random
    ├── viz.py               Overlay + crosshair visualization helpers
    └── checksums.py         SHA256 for data release manifest
```

---

## Data flow

```
Raw FISH images
       │
       ▼
  data/metadata.py ──► metadata.csv, counts_master.csv
       │
       ▼
  data/qc.py ──────► consistency.csv  (64-column QC report)
       │
       ├──────────────────────────────────────────────────────────┐
       │                                                          │
       ▼                                                          ▼
CLASSICAL PIPELINE                                        ECCOUNT MODEL
       │                                                          │
classical_opt/stage1 ──► best combo order                        │
classical_opt/stage2 ──► best preproc params                     │
classical_opt/stage3 ──► best detect+merge params       eccount/targets.py
classical_opt/freeze ──► configs/classical/                      │
       │                 stage3_frozen_params.json       eccount/dataset.py
       │                                                          │
classical/infer.py ─────► binary mask PNGs              eccount/train.py
                                                                  │
                                                         eccount/infer.py
                                                                  │
                                                         binary mask PNGs
                                                         peaks mask PNGs
       │                                                          │
       └────────────────────┬─────────────────────────────────────┘
                            │
EXTERNAL BASELINES          │
baselines/{ecseg,mia,le} ──► harmonized binary masks
                            │
                            ▼
                   benchmark/run.py
                   (all models × OR/AND)
                            │
                            ▼
                   benchmark/aggregate.py
                            │
                            ▼
              release/frozen_results/
                  per_image_metrics_{policy}_mxdist_20_IOU_0.1.csv
                            │
                            ▼
                   figures/fig*.py
                            │
                            ▼
              release/figures/fig{1..6}/*.{png,svg}
```

---

## Key design decisions

### 1. Everything reads from one place
Every script reads parameters from `configs/` and results from
`release/frozen_results/`.  No script hard-codes a path or re-computes
something already in a CSV.

### 2. OR/AND policy is first-class
The matching policy (`OR` or `AND`) is a required argument to `match_objects`.
There is no implicit default.  All downstream metrics inherit the policy.

### 3. "Ignored" is a first-class bucket
A prediction that had a valid candidate in the cost matrix but was outcompeted
in the Hungarian assignment is labelled "ignored", not FP.  This prevents
penalising a model for predictions that were plausible but redundant.

### 4. ecCount pipeline order is strict
Post-processing follows the exact order: Gaussian smooth → re-apply ROI →
local maxima → absolute threshold → greedy NMS.
Swapping smooth and ROI would cause boundary bleed; the order is tested in
`test_postprocess.py::test_pipeline_order_smooth_then_roi`.

### 5. Figures never re-run inference
All `make_figN` functions accept DataFrames, not file paths.  The CLI loads
the CSVs; the figure functions are pure data → matplotlib.

### 6. No argparse in library code
Business logic lives in `src/ecdna_bench/`.  Argument parsing lives exclusively
in `src/ecdna_bench/cli/`.  This makes every algorithm unit-testable without
subprocess or filesystem setup.

### 7. ecCount model parameter count is guarded
`build_model(ModelConfig())` asserts exactly **7,849,601** trainable parameters.
An architectural change will fail loudly rather than silently producing wrong
published results.

### 8. Classical params are data, not code
Per-cell-line optimised parameters live in
`configs/classical/stage3_frozen_params.json`, which is version-controlled.
The inference code reads this JSON; it never contains hard-coded numbers.

### 9. Stage-1 training d_max ≠ evaluation d_max
Stage-1 order search uses d_max = **10 px** for internal matching (smaller,
faster).  The main evaluation uses d_max = **20 px**.
These are documented separately in `REWRITE_PLAN.md §2` and enforced in tests.

### 10. Baselines ship prediction masks only
`src/ecdna_bench/baselines/` contains harmonization adapters that normalize
pre-computed prediction files.  The baseline models themselves are not included;
see `docs/EXTERNAL_BASELINES.md` for provenance and download instructions.
