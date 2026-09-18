# Reproduction guide

This guide maps the published numbers and figure panels to the files and code
that produce them, and lists the commands to recompute them. For installation
and data download, see [`TUTORIAL_EXTERNAL.md`](TUTORIAL_EXTERNAL.md) (any
computer) or [`TUTORIAL_LONGLEAF.md`](TUTORIAL_LONGLEAF.md) (Brunk Lab).

All commands run from the repository root.

---

## 1. Published numbers

Every number in the paper's comparison comes from the frozen tables in
`release/frozen_results/`:

| File (in `or_matching/`) | Content |
|---|---|
| `summary_overall.csv` | pooled metrics per method, all 1,145 benchmark image sets |
| `summary_by_split.csv` | the same per partition (train, val, test) |
| `summary_by_cell_line.csv` | the same per cell line |
| `summary_by_density_bin.csv` | the same per gold-standard count bin |
| `per_image_metrics.csv` | one row per image set and method |
| `ranked_by_obj_f1.csv` | methods ranked by pooled object-level F1 |

`and_matching/` holds the same files for the AND policy (sensitivity analysis
only); the file names are identical, so always check the folder.

Check any results folder against the paper:

```bash
python scripts/verify_headline_numbers.py                       # the frozen tables
python scripts/verify_headline_numbers.py --results runs/benchmark
```

Recompute the tables from the prediction masks (into a new folder, never over
the frozen ones):

```bash
python -m ecdna_bench.cli.benchmark --config configs/default.yaml \
    --output-dir runs/benchmark --n-workers 8
python scripts/verify_headline_numbers.py --results runs/benchmark
```

Outside the lab's cluster, prepare the configuration first with
`scripts/prepare_local_run.py bia` (see the external tutorial, section 3).

---

## 2. Figure panels

The figure notebooks read the frozen tables and write each panel with a
`source_*.csv` of the plotted values into `release/figures/notebookNN/`. No
model is run at figure time. Data plots are written as `.svg`; microscopy
composites as `.png`.

Before re-running a notebook, set its output folder in the first cell to a
folder of your own; `release/figures/` is the published version.

```bash
jupyter nbconvert --to notebook --execute notebooks/03_models_comparison.ipynb \
    --output-dir runs/figures_check
```

| Notebook | Panels (file prefixes) | Main inputs |
|---|---|---|
| `01_dataset_exploration.ipynb` | `fig1_*`: resource composition, count distributions, partitions, the example image set | `release/manifests/metadata.csv`, `notebook01/source_*.csv` |
| `02_classical_pipeline.ipynb` | `fig3_*`: classical pipeline, optimization trajectories, before and after optimization, frozen parameters | `or_matching/per_image_metrics.csv`, `source_bo_*.csv`, `source_frozen_params_table.csv` |
| `03_models_comparison.ipynb` | `fig4_qualitative_six_models`; `fig5_*` (ecCount training and post-processing); `fig6_*` (comparison) | `or_matching/*.csv`, `train_history.csv` |
| `04_matching_and_sensitivity.ipynb` | `fig2_matching_example`; `figS_*` sensitivity and OR versus AND | `source_fig2_*.csv`, `sensitivity_sweep.csv` |
| `05_finalisation.ipynb` | `fig3_anchor_classical_overlay`, `fig3_classical_wilcoxon_delta`, `fig4_segmentation_test_bars`, `fig6_global_performance_bars`, `fig6_f1_by_cell_line_heatmap`, `fig6_count_agreement_top3`, `figS_*` heatmap, count disclosure, pixel-level Dice and gallery | `or_matching/*.csv`, `source_fig3_*wilcoxon*.csv`, `source_figS_*.csv` |
| `06_test_split_figures.ipynb` | test-split figures (file list in `release/figures/notebook06/`) | `or_matching/per_image_metrics.csv`, `sensitivity_sweep.csv`, the tables of notebook 07 |
| `07_biology_demonstration.ipynb` | `fig7*`: drug-treatment demonstration; `fig8*`: predicted-ROI validation | `or_matching/per_image_metrics.csv`, drug-treatment material and predicted ROIs on the Brunk Lab cluster |

Notebooks 06 and 07 are lab records: they read folders that exist only on the
Brunk Lab cluster, and 07 must run before 06. Their outputs are released
unchanged in `release/figures/notebook06/` and `release/figures/notebook07/`.

The file prefixes are generation-level names. The final figure and Extended
Data panel numbers are those of the manuscript, where panels were assembled
from these files.

Panels produced by scripts rather than notebooks:

| Script | Panels |
|---|---|
| `scripts/collect_loco_results.py`, `scripts/rescore_in_distribution_by_scope.py`, `scripts/build_source_extended_data_fig8.py`, `scripts/loco_figures.py` | leave-one-cell-line-out results (main Fig. 6d–g, Extended Data Fig. 8) |
| ROI analysis scripts (`roi_accuracy_metrics.py`, `roi_accuracy_plots.py`, `roi_example_figures.py`) | ROI agreement (Extended Data Fig. 2) |

---

## 3. The statistics

The paired Wilcoxon signed-rank tests of the classical pipeline before and
after optimization (Extended Data Fig. 4e) are computed from
`release/figures/notebook05/source_fig3_classical_wilcoxon_delta.csv`; the
results are in `source_fig3_wilcoxon.csv` in the same folder. They were
recomputed under SciPy 1.13.1 and 1.15.3 with identical results.

---

## 4. The full pipeline

From the raw images (after downloading the full archive and writing
`configs/paths.local.yaml`):

```bash
# metadata and quality control
python -m ecdna_bench.cli.build_metadata     --config configs/default.yaml
python -m ecdna_bench.cli.run_qc             --config configs/default.yaml

# classical pipeline: optimization (per cell line), then inference
python -m ecdna_bench.cli.optimize_classical --config configs/default.yaml --stage all
python -m ecdna_bench.cli.run_classical      --config configs/default.yaml

# ecCount: training, then inference
python -m ecdna_bench.cli.train_eccount      --config configs/default.yaml
python -m ecdna_bench.cli.run_eccount        --config configs/default.yaml

# comparators: convert their native outputs to the common mask format
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model ecseg
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model mia
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model label_engine

# scoring and the matching sensitivity analysis
python -m ecdna_bench.cli.benchmark          --config configs/default.yaml --output-dir runs/benchmark
python -m ecdna_bench.cli.sensitivity        --config configs/default.yaml

# leave-one-cell-line-out (four runs plus the size-matched control)
python scripts/train_eccount_loco.py --hold-out all --dry-run
sbatch --array=0-4 slurm/submit_eccount_loco.sh        # or train_eccount_loco.py per cell line
```

On a SLURM cluster, `slurm/` has a job file for each stage.

The native outputs of the comparators are not produced by this repository:
ecSeg was run with its released `metaseg.h5` weights, MIA predictions are
archived outputs of the original study, and Label Engine is trained and run
with <https://github.com/Brunk-Lab/Label-Engine>. The deposited prediction masks
(S-BIAD4097, `predictions/`) are the converted outputs used for every result.
See [`EXTERNAL_BASELINES.md`](EXTERNAL_BASELINES.md).

---

## 5. Compute

| Stage | Resources used | Notes |
|---|---|---|
| Classical optimization | CPU nodes, one job per cell line | about 6 h per job |
| ecCount training | one GPU | 70 epochs at about 80 s each (`train_history.csv`) |
| ecCount inference | one GPU (a CPU works) | several seconds per image on a CPU |
| Scoring and sensitivity | CPU nodes | minutes to hours depending on workers |
| Leave-one-cell-line-out | one GPU per run | about 1 h 40 min per run; about 35 min when NCI-H2170 is held out |
| ROI model training | one NVIDIA L40 GPU | 23.7 h for 200 epochs |
| Figures | CPU | minutes per notebook |
