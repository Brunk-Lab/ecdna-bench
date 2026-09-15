# Changelog

All notable changes to ecdna-bench are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

- ROI model (`ecdna_bench.roi`) and its command-line entry point.
- Zenodo DOI for the v1.0.0 release in `CITATION.cff` and the README.
- Article DOI after acceptance.

---

## [1.0.0] — 2026-09-17

First public release, accompanying the manuscript *An open imaging and AI
resource enabling unbiased quantification of extrachromosomal DNA at scale*.

### Resource

- Imaging resource deposited at the BioImage Archive, accession S-BIAD4097
  (<https://doi.org/10.6019/S-BIAD4097>): 2,986 paired probe-channel RGB and
  DAPI image sets from five cell lines with gold-standard ecDNA annotation
  (points, rendered masks, sparse masks), manual ROI masks for the 1,145-image
  benchmark, predicted ROI masks for all image sets, prediction masks of seven
  method outputs, and the partition files.
- Fixed benchmark partitions (800 / 170 / 175) in `release/split_files/`.
- Frozen result tables for every published number in
  `release/frozen_results/` (`or_matching/` canonical, `and_matching/` for the
  sensitivity analysis), and figure source data in `release/figures/`.

### Evaluation framework (`src/ecdna_bench/evaluation/`)

- Binary mask → 8-connected components (≥ 3 px) → one-to-one Hungarian matching
  with OR and AND eligibility policies (`d_max`, `IoU_min`, `α` configurable).
- Ignored-prediction outcome for eligible predictions that lose the assignment.
- Object (precision, recall, F1), pixel (Dice, IoU) and count (MAE, RMSE, bias,
  Pearson, Spearman, MdAPE) metrics; object and pixel metrics pooled over
  images.
- Sensitivity sweep over `d_max × IoU_min × policy`.

### Methods

- Classical pipeline (`classical/`) with three-stage Bayesian optimization
  (`classical_opt/`) and per-cell-line frozen parameters in `configs/classical/`.
- ecCount (`eccount/`): Gaussian targets (σ = 1 px, element-wise maximum),
  U-Net with 7,849,601 parameters, weighted BCE + soft Dice, peak and threshold
  outputs; released checkpoint from epoch 49 (validation loss 0.6421); tuning
  helpers in `eccount/tuning/`.
- Adapters converting ecSeg, MIA and Label Engine outputs to the common mask
  format (`baselines/`), and the benchmark orchestrator (`benchmark/`) with the
  model registry (`classical`, `classical_before_opt`, `label_engine`, `ecseg`,
  `mia`, `eccount_mask`, `eccount_peaks`).

### Tools

- Command-line modules for every stage: `build_metadata`, `run_qc`,
  `optimize_classical`, `run_classical`, `train_eccount`, `run_eccount`,
  `run_baseline`, `benchmark`, `sensitivity`.
- `scripts/fetch_bia_subset.py`: download all or part of S-BIAD4097.
- `scripts/prepare_local_run.py`: self-contained run folders for archive data
  or new images, and object counting for mask folders.
- `scripts/verify_headline_numbers.py`: compare a results folder with the
  published values.
- `scripts/train_eccount_loco.py` and `slurm/submit_eccount_loco.sh`:
  leave-one-cell-line-out training, inference and scoring with split guards.
- `scripts/setup_new_user.sh` (version 2): Longleaf account setup that keeps
  all project settings in a session started with `ecdna`; `--migrate` removes
  the settings written by version 1.
- `scripts/rename_gold_standard.py`: the terminology migration used for this
  release (below).

### Documentation

- `docs/TUTORIAL_EXTERNAL.md` (any computer) and `docs/TUTORIAL_LONGLEAF.md`
  (Brunk Lab members).
- Tutorial notebooks in `notebooks/tutorials/`: images and the gold standard,
  running ecCount, scoring, retraining.
- `env/environment-cpu.yml` and `env/requirements-cpu.txt` for computers
  without an NVIDIA GPU.
- `README.md`, `DATASET.md`, `CITATION.cff`, `docs/REPRODUCTION.md`,
  `docs/ARCHITECTURE.md`, `docs/EXTERNAL_BASELINES.md`.

### Changed

- **Terminology:** "ground truth" is now "gold standard" (GS) in documentation,
  comments, notebook text and messages. Identifiers, file and folder names,
  column names and the names used in the deposited archive record keep `gt`, so
  existing code, frozen tables and downloads remain compatible.
- The threshold-mask output's model key is `eccount_mask` everywhere in the
  documentation.

### Fixed

- `env/environment.yml`: OpenCV is pinned to an existing release
  (`opencv-python-headless==4.10.0` does not resolve on PyPI).
- Leave-one-cell-line-out documentation no longer places a per-run
  configuration in `configs/`, where `configs/paths.local.yaml` would override
  it.
- `slurm/submit_eccount_loco.sh` takes the project root from
  `ECDNA_PROJECT_ROOT` (or the submission folder) and the interpreter from
  `ECDNA_PYTHON`, like the other job scripts.

### Published values (OR matching, pooled)

| Method | All 1,145: F1 / MAE / bias | Test 175: F1 / MAE |
|---|---|---|
| ecCount (peaks) | 0.942 / 13.1 / +0.4 | 0.939 / 13.3 |
| ecCount (threshold mask) | 0.917 / 18.1 / −11.4 | 0.916 / 17.6 |
| Label Engine | 0.825 / 36.0 / −28.8 | 0.825 / 34.4 |
| MIA | 0.800 / 47.0 / −40.6 | 0.813 / 40.4 |
| Classic (after opt) | 0.777 / 44.9 / −24.3 | 0.775 / 40.9 |
| ecSeg | 0.464 / 121.5 / −118.5 | 0.508 / 110.3 |
