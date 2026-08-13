# Changelog

All notable changes to ecdna-bench are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.0.0] — {{RELEASE_DATE}}

Initial public release accompanying the *Nature Computational Science*
submission.

### Added — Dataset
- Full reference resource: **2,986** paired RGB + DAPI FISH images across five
  cell lines (NCI-H2170, SUM159PT, SNU16, COLO320DM, NCI-H716).
- Benchmark subset: **1,145** images across four cell lines (NCI-H2170, SNU16,
  NCI-H716, COLO320DM) with manually annotated GT masks, ROI masks, and
  per-image ecDNA counts. SUM159PT is excluded from the benchmark (no ROI masks).
- Fixed train/val/test splits (800 / 170 / 175) committed in
  `release/split_files/`.
- SHA256 manifest for data-integrity verification.

### Added — Evaluation framework (`src/ecdna_bench/evaluation/`)
- Unified binary-mask → connected-component → Hungarian-matching pipeline.
- OR and AND validity policies; `d_max`, `IoU_min`, `α` all configurable.
- **Ignored** prediction bucket (predictions that had a valid candidate but
  lost the assignment — not counted as FP).
- Three metric families: object (P / R / F1), pixel (Dice / IoU), count
  (MAE / RMSE / bias / Pearson r / Spearman ρ / MdAPE).
- Pooled (micro-averaged) F1 adopted as the headline aggregation metric.
- Sensitivity sweep over `d_max × IoU_min × policy`.

### Added — Classical pipeline (`src/ecdna_bench/classical/`, `classical_opt/`)
- Preprocessing registry (top-hat, CLAHE, gamma, sigmoid, bilateral,
  sharpening, background subtraction).
- Otsu threshold + morphological close + component detection + centroid merge.
- HSV-based ecDNA / chromosome classifier.
- 3-stage Bayesian optimisation (Stage 1: operator-order search;
  Stage 2: preprocessing params; Stage 3: detection + merge params).
- Per-cell-line frozen parameters committed to `configs/classical/`.

### Added — ecCount (`src/ecdna_bench/eccount/`)
- Soft Gaussian target generation (σ = 1.0, element-wise max, normalised).
- Compact U-Net (base_channels = 32, GroupNorm, bilinear upsample, 7,849,601
  trainable parameters).
- wBCE (pos_weight = 20) + SoftDice loss.
- Resumable training loop with ReduceLROnPlateau scheduler;
  best checkpoint at epoch 49 (val loss 0.6421).
- Post-processing: Gaussian smooth → ROI re-apply → local-maxima peaks → NMS.
- Two output modes: threshold mask and peaks mask.
- Ablation helpers (`eccount/tuning/`): σ, loss, schedule, post-process grids.

### Added — External baselines (`src/ecdna_bench/baselines/`)
- Harmonisation adapters for ecSeg, MIA, and Label Engine.
- Label Engine UID canonicalisation logic (authoritative copy).

### Added — Benchmark orchestrator (`src/ecdna_bench/benchmark/`)
- Model registry covering the six benchmarked models — Classic (after opt),
  Label Engine, ecSeg, MIA, ecCount (threshold mask), ecCount (peaks) — plus
  Classic (before opt), which is used only for the before/after
  Bayesian-optimisation comparison and is not reported as a benchmark entry.
- Cross-model evaluation across OR/AND policies.
- Group aggregation: overall, by split, by cell line, by density bin.
- Frozen-result writer producing the canonical per-image and summary CSVs.

### Added — CLI (`src/ecdna_bench/cli/`)
- Nine entry-point commands: `build_metadata`, `run_qc`,
  `optimize_classical`, `run_classical`, `train_eccount`, `run_eccount`,
  `run_baseline`, `benchmark`, `sensitivity`.
- Consistent `--config` + `--log-level` interface; all runnable as
  `python -m ecdna_bench.cli.<command>`.

### Added — Figures (`notebooks/`)
- Notebooks `01`–`05` regenerate all main and supplementary figures, reading
  only from the frozen CSVs in `release/frozen_results/` and writing to
  `release/figures/notebookNN/`. No inference happens at figure time.

### Added — Tests (`tests/`)
- 126 pytest tests covering matching invariants, metrics, targets,
  post-processing, and I/O helpers. Synthetic fixtures only, apart from two
  I/O tests that are skipped unless the released sample images are present.

### Added — Documentation
- `README.md`, `DATASET.md`, `CITATION.cff`, `Makefile`.
- `docs/REPRODUCTION.md` (figure → notebook → output mapping),
  `docs/ARCHITECTURE.md` (module map), `docs/EXTERNAL_BASELINES.md`
  (baseline provenance), `LONGLEAF_INSTALL.md` (HPC setup).

### Benchmark headline (held-out test split, n = 175, OR policy, pooled F1)
- ecCount (peaks): Obj F1 0.939, count MAE 13.3.
- ecCount (threshold mask): Obj F1 0.916, count MAE 17.6.
- Label Engine: Obj F1 0.825, count MAE 34.4.
- MIA: Obj F1 0.813, count MAE 40.4.
- Classic (after opt): Obj F1 0.775, count MAE 40.9.
- ecSeg: Obj F1 0.508, count MAE 110.3.
