# Reproduction Guide

This document maps every paper figure and table to the **notebook** that
generates it and lists the output files produced. There is no `make_figures`
CLI; all figures are produced by the five notebooks under `notebooks/`.

**Prerequisites:** follow the [Quick start](../README.md#quick-start) section to
install the package and configure paths. All commands assume the working
directory is the repo root and `configs/paths.local.yaml` is populated.

---

## How figures are generated

The five notebooks read **only** from the frozen CSVs in
`release/frozen_results/` (and the figure source-data CSVs alongside each
figure) and write their outputs to `release/figures/notebook0N/`. **No model
inference happens at figure time** — the figures are a pure function of the
frozen results. Data-plot panels are written as `.svg`; microscopy composite
panels (RGB / GT / ROI overlays) are written as `.png` rasters because they are
not vectorisable.

To regenerate a figure, run its notebook (interactively in JupyterLab, or
headlessly):

```bash
jupyter nbconvert --to notebook --execute \
    notebooks/03_models_comparison.ipynb \
    --output 03_models_comparison_reproduction.ipynb
```

---

## Notebook → outputs (authoritative map)

### `notebooks/01_dataset_exploration.ipynb` — Figure 1 (dataset composition)
**Input:** `release/manifests/metadata.csv` + the `notebook01` figure
source-data CSVs (`source_benchmark_count_summary.csv`,
`source_full_resource_summary.csv`, `source_density_bin_summary.csv`,
`fig1_*_composition_*.csv`, `fig1_count_distribution.csv`,
`fig1_split_heatmap.csv`).
**Outputs** (`release/figures/notebook01/`):
- Data plots (`.svg`): `fig1_count_distribution`,
  `fig1_count_by_cell_line_boxstrip`, `fig1_density_composition_heatmap`,
  `fig1_resource_composition_horizontal`, `fig1_split_heatmap`.
- Microscopy composites (`.png`): `fig1_anchor_composite`,
  `fig1_anchor_rgb_gt_with_zoom`, `fig1_anchor_inset_clean`,
  `fig1_anchor_roi_masking_workflow`.

### `notebooks/02_classical_pipeline.ipynb` — Figure 3 (classical pipeline + BO)
**Input:** `release/frozen_results/or_matching/per_image_metrics.csv`, the BO
trajectory/probe source CSVs (`source_bo_trajectories.csv`,
`source_bo_selected_probes.csv`, `source_frozen_params_table.csv`), and the
before/after source CSVs.
**Outputs** (`release/figures/notebook02/`, `.svg`):
`fig3_pipeline_schematic`, `fig3_preproc_steps`, `fig3_preproc_steps_composite`,
`fig3_detection_steps`, `fig3_bo_trajectories_simplified`,
`fig3_bo_before_after_summary`, `fig3_before_after_bo_overlay`,
`fig3_before_after_f1_by_cell_line`, `fig3_frozen_params_table`.

### `notebooks/03_models_comparison.ipynb` — Figures 4, 5, 6
**Input:** `release/frozen_results/or_matching/` CSVs +
`release/model_checkpoints/train_history.csv`.
**Outputs** (`release/figures/notebook03/`, `.svg`):
- Figure 4: `fig4_qualitative_six_models`.
- Figure 5 (ecCount): `fig5_training_curves`, `fig5_probability_map`,
  `fig5_postproc_steps`, `fig5_threshold_vs_peaks`,
  `fig5_threshold_vs_peaks_zoomed`.
- Figure 6 (benchmark): `fig6_f1_overall`, `fig6_f1_by_cell_line`,
  `fig6_f1_by_density`, `fig6_test_count_mae`, `fig6_test_only`,
  `fig6_count_agreement`.

### `notebooks/04_matching_and_sensitivity.ipynb` — Figure 2 + supplementary
**Input:** `source_fig2_pair_table.csv`,
`source_fig2_matching_example_summary.csv`, `sensitivity_sweep.csv`,
`source_figS_sensitivity_*.csv`, `source_figS_or_vs_and_headline.csv`.
**Outputs** (`release/figures/notebook04/`, `.svg`):
`fig2_matching_example`, `figS_or_vs_and_headline`,
`figS_sensitivity_heatmaps`, `figS_sensitivity_range`,
`figS_sensitivity_range_and`.

### `notebooks/05_finalisation.ipynb` — Figures 3/4/6 additions + supplementary
**Input:** `release/frozen_results/or_matching/` CSVs,
`pixel_dice_locked_numbers.csv`, `source_fig3_*wilcoxon*.csv`,
`source_figS_gt_count_disclosure*.csv`, `source_figS_qualitative_gallery_*.csv`,
`source_fig6_global_performance_bars.csv`.
**Outputs** (`release/figures/notebook05/`, `.svg`):
`fig3_anchor_classical_overlay`, `fig3_classical_wilcoxon_delta`,
`fig4_segmentation_test_bars`, `fig6_global_performance_bars`,
`fig6_f1_by_cell_line_heatmap`, `fig6_count_agreement_top3`,
`figS_f1_by_cell_line_heatmap_7row`, `figS_gt_count_disclosure`,
`figS_pixel_dice_locked_numbers`, `figS_qualitative_gallery`.

---

## Figure / Table → notebook quick reference

Some display items are assembled from panels produced by **more than one**
notebook.

| Display item | Notebook(s) |
|---|---|
| Figure 1 — dataset composition | `01` |
| Figure 2 — evaluation / matching framework | `04` |
| Figure 3 — classical pipeline + BO | `02` (main panels) + `05` (anchor overlay, Wilcoxon Δ) |
| Figure 4 — qualitative gallery + segmentation baselines | `03` (qualitative) + `05` (segmentation test bars) |
| Figure 5 — ecCount | `03` |
| Figure 6 — six-model benchmark | `03` (F1/MAE/agreement) + `05` (global bars, per-cell-line heatmap, top-3 agreement) |
| Supplementary figures (`figS_*`) | `04` (sensitivity, OR-vs-AND) + `05` (per-cell-line heatmap, GT-count disclosure, pixel-Dice, qualitative gallery) |
| Table 1 — overall benchmark results | `benchmark` CLI (see below) |

Supplementary-figure **numbers** are assigned in
`release/manuscript/Supplementary_Figures_updated.docx`; the `figS_*` filenames
above are the generation-level identifiers.

---

## Source data

Every `.svg` panel is written together with the source-data CSV it was drawn
from, in the same `release/figures/notebook0N/` directory. Those CSVs are the
canonical figure source data bundled for submission.

---

## Table 1 — Overall benchmark results

Source data: `release/frozen_results/or_matching/summary_overall.csv`.

To regenerate the underlying numbers from scratch:

```bash
python -m ecdna_bench.cli.benchmark --config configs/default.yaml
```

The Table 1 values in the paper are taken directly from the `obj_f1`,
`count_mae`, and `pix_dice` columns of `summary_overall.csv`.

---

## Running the full pipeline from scratch

To reproduce from raw data (no pre-computed masks). All stages use a single
config, `configs/default.yaml`:

```bash
# 1. Build and QC metadata (~10 min)
python -m ecdna_bench.cli.build_metadata --config configs/default.yaml
python -m ecdna_bench.cli.run_qc         --config configs/default.yaml

# 2. Optimise classical pipeline (~6 h on 16 CPUs)
python -m ecdna_bench.cli.optimize_classical --stage all --config configs/default.yaml

# 3. Run classical inference (~30 min on 16 CPUs)
python -m ecdna_bench.cli.run_classical --config configs/default.yaml

# 4. Train ecCount (~12 h on 1 GPU, 70 epochs)
python -m ecdna_bench.cli.train_eccount --config configs/default.yaml

# 5. Run ecCount inference (~20 min on 1 GPU)
python -m ecdna_bench.cli.run_eccount --config configs/default.yaml

# 6. Harmonise baselines (~2 min each)
python -m ecdna_bench.cli.run_baseline --model ecseg        --config configs/default.yaml
python -m ecdna_bench.cli.run_baseline --model mia          --config configs/default.yaml
python -m ecdna_bench.cli.run_baseline --model label_engine --config configs/default.yaml

# 7. Run benchmark (~10 min on 8 CPUs)
python -m ecdna_bench.cli.benchmark   --config configs/default.yaml

# 8. Sensitivity sweep (~30 min on 4 CPUs)
python -m ecdna_bench.cli.sensitivity --config configs/default.yaml

# 9. Generate all figures: run notebooks/01..05
#    (interactively in JupyterLab, or headlessly with jupyter nbconvert --execute)
```

---

## Hardware requirements

| Step | Minimum | Used in paper |
|------|---------|---------------|
| Classical optimisation | 8 CPU cores | UNC Longleaf HPC, 16 cores |
| ecCount training | 1 GPU (8 GB VRAM) | NVIDIA A100 40 GB |
| ecCount inference | 1 GPU (4 GB VRAM) | NVIDIA A100 40 GB |
| Benchmark evaluation | 4 CPU cores | UNC Longleaf HPC, 8 cores |
| Figures (notebooks 01–05) | CPU only | MacBook Pro M2 |

---

## Expected runtimes

Approximate wall-clock times on the hardware used in the paper:

| Stage | Runtime |
|-------|---------|
| build_metadata + run_qc | ~10 min |
| optimize_classical (all cell lines) | ~6 h |
| run_classical | ~30 min |
| train_eccount (70 epochs) | ~12 h |
| run_eccount | ~20 min |
| run_baseline (3 models) | ~5 min total |
| benchmark | ~10 min |
| sensitivity | ~30 min |
| figures (notebooks 01–05) | a few minutes each, CPU only |