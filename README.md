# ecdna-bench

**An open imaging resource, benchmark and models for counting extrachromosomal DNA (ecDNA) in metaphase FISH images, including ecCount, a probabilistic-localization model.**

This repository accompanies the manuscript *An open imaging and AI resource
enabling unbiased quantification of extrachromosomal DNA at scale*
(Behnamie et al., 2026). It contains the evaluation framework, the six
benchmarked methods (or their adapters), the trained models, the frozen result
tables behind every figure, and step-by-step tutorials.

| | |
|---|---|
| Images and annotations | BioImage Archive **S-BIAD4097**, <https://doi.org/10.6019/S-BIAD4097> (CC BY 4.0) |
| Code | this repository (MIT License) |
| Model weights | assets of the GitHub release (see [Release assets](#release-assets)) |
| Label Engine | <https://github.com/Brunk-Lab/Label-Engine> |
| Tutorials | [outside UNC](docs/TUTORIAL_EXTERNAL.md) · [Brunk Lab on Longleaf](docs/TUTORIAL_LONGLEAF.md) · [notebooks](notebooks/tutorials/) |

> **Terminology.** The manual ecDNA annotation is the **gold standard (GS)**.
> The archive record calls the same annotation "ground truth", and file,
> folder and column names keep the short form `gt` (`images/gt_image/`,
> `gt_count`, ...) so that they match the deposited data and the frozen tables.

---

## Contents

- [Quick start](#quick-start)
- [The resource](#the-resource)
- [Evaluation framework](#evaluation-framework)
- [Results](#results)
- [ecCount](#eccount)
- [ROI model](#roi-model)
- [Reproducing the paper](#reproducing-the-paper)
- [Repository layout](#repository-layout)
- [Release assets](#release-assets)
- [Citation](#citation)
- [License and contact](#license-and-contact)

---

## Quick start

```bash
git clone https://github.com/Brunk-Lab/ecdna-bench.git
cd ecdna-bench

# no GPU (laptop, workstation, Apple silicon)
conda env create -f env/environment-cpu.yml
conda activate ecdna-bench-cpu
# or, with an NVIDIA GPU on Linux:
#   conda env create -f env/environment.yml && conda activate ecdna-bench
# or, without conda:
#   python3.10 -m venv .venv && source .venv/bin/activate
#   python -m pip install -r env/requirements-cpu.txt

python -m pip install -e . --no-deps

python -m pytest -q                        # installation check, no data needed
python scripts/verify_headline_numbers.py  # frozen tables vs the paper: 12 x MATCH
```

Then pick a route in [`docs/TUTORIAL_EXTERNAL.md`](docs/TUTORIAL_EXTERNAL.md):
re-score the deposited predictions, run ecCount on the benchmark or on your own
images, or retrain it.

Download only what you need from the archive:

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
    --sections gt,roi,predictions
python scripts/prepare_local_run.py bia --bia-root ~/ecdna_data \
    --out-dir runs/bia_test --split test
# then the commands in runs/bia_test/NEXT_STEPS.txt
```

The archive record is private until 1 September 2027; editors and reviewers
have access through the link in the reviewer instructions.

---

## The resource

**2,986 image sets** (paired probe-channel RGB and DAPI images, 2,448 × 2,048
px, ×60) from five ecDNA-positive cancer cell lines, each with a manual
gold-standard annotation of every ecDNA signal. **1,145 image sets** also carry a
manual region-of-interest (ROI) mask around the metaphase spread; they form the
benchmark, with fixed partitions of 800 training, 170 validation and 175
held-out test image sets.

| Cell line | Full resource | Benchmark | Train / val / test | Median GS count (benchmark) | Range |
|---|---|---|---|---|---|
| NCI-H2170 | 1,891 | 888 | 621 / 133 / 134 | 153 | 9–1,687 |
| SNU16 | 273 | 123 | 86 / 18 / 19 | 241 | 57–1,231 |
| NCI-H716 | 139 | 70 | 49 / 10 / 11 | 174 | 71–468 |
| COLO320DM | 225 | 64 | 44 / 9 / 11 | 43 | 1–144 |
| SUM159PT | 458 | 0 | — | — | — |
| **Total** | **2,986** | **1,145** | **800 / 170 / 175** | **159** | **1–1,687** |

SUM159PT has no manual ROI masks and is therefore outside the benchmark; the
archive provides model-predicted ROI masks for all 2,986 image sets.

Sources: `release/figures/notebook01/source_full_resource_summary.csv`,
`release/figures/notebook01/source_benchmark_count_summary.csv`.
Full data card: [`DATASET.md`](DATASET.md).

---

## Evaluation framework

Every method's output is converted to a binary mask at the native resolution
and scored the same way:

1. **Objects** are 8-connected components of at least 3 px. Gold-standard
   points are rendered as 5 × 5-pixel diamonds (13 px) before this step, and so
   are ecCount peaks.
2. **Eligibility.** A predicted and a gold-standard object may be paired if the
   centroid distance is at most `d_max = 20 px` **or** the mask IoU is at least
   `IoU_min = 0.1` (OR policy; the AND policy requires both).
3. **Assignment.** Eligible pairs are matched one-to-one (Hungarian algorithm) at
   minimum total cost `C = α(1 − IoU) + (1 − α)·d/d_max`, `α = 0.5`.
4. **Outcomes.** Matched pairs are true positives, unmatched gold-standard
   objects false negatives, and predictions with no eligible partner false
   positives. A prediction that had an eligible partner but lost the assignment
   is *ignored* (neither TP nor FP).
5. **Metrics.** Object-level precision, recall and F1, and pixel-level Dice,
   pool TP, FP and FN over images before the ratio is taken. Count metrics (MAE,
   RMSE, mean signed bias, Pearson and Spearman correlation, median absolute
   percentage error) are computed per image and averaged.

All settings are in the `evaluation:` section of
[`configs/default.yaml`](configs/default.yaml). The paper uses OR matching
throughout; AND results are written alongside, to `and_matching/`.

---

## Results

Pooled over images, OR matching.

| Method | All 1,145: object F1 | Count MAE | Mean signed bias | Test 175: object F1 | Count MAE | Pixel-level Dice |
|---|---|---|---|---|---|---|
| ecCount (peaks) | **0.942** | **13.1** | **+0.4** | **0.939** | **13.3** | 0.310 |
| ecCount (threshold mask) | 0.917 | 18.1 | −11.4 | 0.916 | 17.6 | 0.302 |
| Label Engine | 0.825 | 36.0 | −28.8 | 0.825 | 34.4 | 0.247 |
| MIA | 0.800 | 47.0 | −40.6 | 0.813 | 40.4 | 0.217 |
| Classic (after opt) | 0.777 | 44.9 | −24.3 | 0.775 | 40.9 | 0.117 |
| ecSeg | 0.464 | 121.5 | −118.5 | 0.508 | 110.3 | 0.165 |

Pixel-level Dice is for the test split. It is low for every method because the
gold-standard masks are diamonds placed at annotated points, not object
outlines. Label Engine was trained on the benchmark training partition; MIA
values come from the archived predictions of the original study's model; ecSeg
was applied with its released weights, without retraining.

Sources: `release/frozen_results/or_matching/summary_overall.csv`,
`release/frozen_results/or_matching/summary_by_split.csv`,
`release/figures/notebook05/source_fig6_global_performance_bars.csv`.
`python scripts/verify_headline_numbers.py` checks the first five columns.

---

## ecCount

A U-Net that predicts a probability map with one Gaussian peak per ecDNA; counts
are read from the local maxima.

| Property | Value |
|---|---|
| Input | ROI-masked RGB image downsampled by two to 1,024 × 1,224 px (area interpolation), intensities divided by 255 |
| Training target | Gaussian per gold-standard object, σ = 1 px at model resolution, combined by element-wise maximum |
| Architecture | U-Net, 32 base channels, GroupNorm (8 groups), bilinear upsampling, 7,849,601 trainable parameters |
| Loss | weighted binary cross-entropy (positive weight 20) + soft Dice |
| Training | Adam, learning rate 1 × 10⁻⁴, batch size 2, 70 epochs, ReduceLROnPlateau (patience 5, factor 0.5), fp32, seed 42 |
| Selected checkpoint | epoch 49, validation loss 0.6421 |
| Peaks output | smoothing (σ = 0.5 px) → ROI re-applied → local maxima above 0.35 → non-maximum suppression (2 px) → peaks drawn as 5 × 5-pixel diamonds at native resolution |
| Threshold output | probability map thresholded at 0.5 |

Sources: `configs/default.yaml` (`eccount:`), the release asset
`train_history.csv`.

```bash
python -m ecdna_bench.cli.run_eccount --config <run_config.yaml> --split test
```

[`docs/TUTORIAL_EXTERNAL.md`](docs/TUTORIAL_EXTERNAL.md), sections 4 to 6, shows
how to run it on the benchmark, on your own images, and how to retrain it
(including leave-one-cell-line-out runs with `scripts/train_eccount_loco.py`).

---

## ROI model

A separate model predicts the metaphase ROI, so that images without a manual
mask can be analyzed. Its masks for all 2,986 image sets are in the archive
(`predicted_roi/`).

| Property | Value |
|---|---|
| Architecture | residual U-Net, four input channels (RGB + DAPI), 35,923,337 parameters |
| Training | the 800 training and 170 validation image sets of the benchmark, at 1,024 × 1,224 px; best epoch 97 |
| Inference | probability map upsampled to the native frame, Gaussian smoothing (σ = 10 px), threshold 0.4, holes filled, component containing the image center kept |
| Checkpoint | release asset `roi_model_best_checkpoint.pth` |

Agreement with the manual masks over the 1,145 benchmark images: median IoU
0.821, median Dice 0.902, 98.5 % of gold-standard objects inside the predicted
region. With predicted instead of manual ROIs, ecCount (peaks) reaches an
object-level F1 of 0.913 on the 1,145 benchmark images and 0.896 on the 345
validation and test images (manuscript, Supplementary Methods §11). Predicted
ROIs were not used for any benchmark comparison.

Predict ROI masks for your own image sets (`<unique_id>.tif` or `.png` in both
folders; one 0/255 PNG per image set is written):

```bash
python -m ecdna_bench.cli.roi predict --rgb <rgb folder> --dapi <dapi folder> \
    --checkpoint roi_model_best_checkpoint.pth --output <output folder>
```

The code is in `src/ecdna_bench/roi/`. `python -m ecdna_bench.cli.roi train --help`
lists the training options; training needs `albumentations==2.0.8`, the version
used for the released model.

---

## Reproducing the paper

**Frozen tables.** Every published number is in `release/frozen_results/`
(`or_matching/` is canonical). `scripts/verify_headline_numbers.py` compares any
results folder with the paper.

**Figures.** The notebooks in `notebooks/` read the frozen tables and write each
panel together with a `source_*.csv` of the plotted values to
`release/figures/notebookNN/`:

| Notebook | Content |
|---|---|
| `01_dataset_exploration.ipynb` | the resource and the benchmark |
| `02_classical_pipeline.ipynb` | the classical pipeline and its optimization |
| `03_models_comparison.ipynb` | ecCount and the six-method comparison |
| `04_matching_and_sensitivity.ipynb` | the matching framework and its sensitivity |
| `05_finalisation.ipynb` | statistics, heatmaps and the qualitative gallery |

Set the output folder in the first cell before re-running a notebook, so that the
released files are not overwritten. See [`docs/REPRODUCTION.md`](docs/REPRODUCTION.md)
for the figure-to-file mapping. Notebooks 01–05 read the archive through `ECDNA_DATA_ROOT`
(the folder that contains `images/` and `predictions/`). Before the first run,
rebuild the harmonized baseline masks with `python scripts/build_harmonized_masks.py`
and place the released `eccount_best.pt` in `release/model_checkpoints/`. The live
ecCount step in notebook 03 runs only on a GPU and is skipped otherwise; on another
GPU model its illustrative counts can differ slightly. Notebooks 06 and 07 read
folders that exist only on the Brunk Lab cluster and are kept as a record of how
those figures were made. For a first run outside UNC, start with the tutorial
notebooks in [`notebooks/tutorials/`](notebooks/tutorials/), which need only the
archive files.

**Full pipeline.** Each stage is a command-line module; on a SLURM cluster the
job files in `slurm/` run the same commands.

```bash
python -m ecdna_bench.cli.build_metadata     --config configs/default.yaml
python -m ecdna_bench.cli.run_qc             --config configs/default.yaml
python -m ecdna_bench.cli.optimize_classical --config configs/default.yaml --stage all
python -m ecdna_bench.cli.run_classical      --config configs/default.yaml
python -m ecdna_bench.cli.train_eccount      --config configs/default.yaml
python -m ecdna_bench.cli.run_eccount        --config configs/default.yaml
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model ecseg
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model mia
python -m ecdna_bench.cli.run_baseline       --config configs/default.yaml --model label_engine
python -m ecdna_bench.cli.benchmark          --config configs/default.yaml --output-dir runs/benchmark
python -m ecdna_bench.cli.sensitivity        --config configs/default.yaml
```

Machine-specific paths go in `configs/paths.local.yaml` (template:
`configs/paths.example.yaml`), which is merged over `configs/default.yaml`.
`configs/default.yaml` holds the published settings and is not meant to be
edited. Model keys for `--models`: `classical`, `classical_before_opt`,
`label_engine`, `ecseg`, `mia`, `eccount_mask` (threshold mask) and
`eccount_peaks`.

---

## Repository layout

```
ecdna-bench/
├── configs/            default.yaml (published settings), classical parameters, path template
├── env/                environment.yml (GPU), environment-cpu.yml, requirements-cpu.txt
├── src/ecdna_bench/
│   ├── cli/            one command per pipeline stage
│   ├── data/           metadata, QC, image I/O
│   ├── evaluation/     objects, matching, metrics, sensitivity, statistics
│   ├── classical/      rule-based pipeline
│   ├── classical_opt/  three-stage Bayesian optimization
│   ├── eccount/        ecCount model, training, inference, tuning
│   ├── baselines/      adapters for ecSeg, MIA and Label Engine outputs
│   ├── benchmark/      model registry, harmonization, scoring, aggregation
│   └── utils/
├── scripts/            downloads, run preparation, checks, leave-one-cell-line-out training, lab setup
├── slurm/              job files for a SLURM cluster
├── notebooks/          figure notebooks 01–05; tutorials/
├── release/            frozen results, figure source data, split files, manifests
├── docs/               tutorials, reproduction guide, architecture, baseline provenance
└── tests/              pytest suite (synthetic data only)
```

---

## Release assets

Large files are attached to the GitHub release rather than committed:

| Asset | Content |
|---|---|
| `eccount_best.pt` | released ecCount weights; place at `release/model_checkpoints/eccount_best.pt` |
| `train_history.csv` | per-epoch losses of the released ecCount run |
| `roi_model_best_checkpoint.pth` | ROI-model weights |
| `eccount_loco_<cell line>.pt` | leave-one-cell-line-out checkpoints (four runs and the NCI-H2170 size-matched control) |
| `SHA256SUMS` | checksums of every asset |

```bash
sha256sum -c SHA256SUMS      # macOS: shasum -a 256 -c SHA256SUMS
```

---

## Citation

If you use the code, the models or the resource, please cite the article and
the dataset. A machine-readable citation is in [`CITATION.cff`](CITATION.cff).

> Behnamie P, Summers R, Chen J, Mehta A, Igbinigie A, Liu Q, Murray M,
> Guilbaud D, Niethammer M, Brunk E. An open imaging and AI resource enabling
> unbiased quantification of extrachromosomal DNA at scale. 2026 (manuscript
> submitted).

> BioImage Archive S-BIAD4097. https://doi.org/10.6019/S-BIAD4097

---

## License and contact

Code: MIT License ([`LICENSE`](LICENSE)). Data: CC BY 4.0 (BioImage Archive
S-BIAD4097).

Corresponding author: Elizabeth Brunk (elizabeth_brunk@med.unc.edu), Brunk Lab,
University of North Carolina at Chapel Hill. Questions and problems: please open
an issue.
