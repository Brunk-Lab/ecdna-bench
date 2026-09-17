# ecdna-bench on your own computer

A tutorial for anyone outside UNC: no Longleaf account, no SLURM, no shared
folders. Everything runs on a laptop or a workstation; only retraining needs a
GPU.

If you are a Brunk Lab member working on Longleaf, use
[`TUTORIAL_LONGLEAF.md`](TUTORIAL_LONGLEAF.md) instead.

---

## Contents

0. [What is where](#0-what-is-where)
1. [Install](#1-install)
2. [Download the model weights](#2-download-the-model-weights)
3. [Route B: re-score the published predictions](#3-route-b-re-score-the-published-predictions)
4. [Route C: run ecCount yourself and score it](#4-route-c-run-eccount-yourself-and-score-it)
5. [Route D: count ecDNA in your own images](#5-route-d-count-ecdna-in-your-own-images)
6. [Route E: retrain ecCount](#6-route-e-retrain-eccount)
7. [The tutorial notebooks](#7-the-tutorial-notebooks)
8. [Troubleshooting](#8-troubleshooting)
9. [Reference](#9-reference)

Pick a route:

| Route | What you get | Data to download | Hardware |
|---|---|---|---|
| **A** | Installation checked; the published tables verified | none | any computer |
| **B** | The paper's comparison recomputed from the deposited prediction masks | gold-standard masks, ROI masks, prediction masks | any computer |
| **C** | ecCount run by you on the benchmark images, then scored | as B, plus RGB images and the weights | CPU is enough; a GPU is faster |
| **D** | ecDNA counts for your own images | your images and the weights | CPU is enough |
| **E** | A retrained ecCount, for example with one cell line held out | all benchmark images | NVIDIA GPU |

Route A is sections 1 and 2. Every later route starts from it.

---

## 0. What is where

| Item | Location |
|---|---|
| Code, configuration, frozen result tables, tutorials | GitHub: <https://github.com/Brunk-Lab/ecdna-bench> |
| Images, annotations, ROI masks, prediction masks | BioImage Archive, accession **S-BIAD4097** (<https://doi.org/10.6019/S-BIAD4097>) |
| Trained ecCount weights (`eccount_best.pt`) and the ROI-model checkpoint | Assets of the GitHub release (the **Releases** page of the repository) |
| Label Engine | <https://github.com/Brunk-Lab/Label-Engine> |

**Access to the images.** The archive record is private until 1 September 2027.
Until then it is available to the journal's editors and reviewers through the
private link in the reviewer instructions; everyone else can follow Route A and
Route D (with their own images) straight away, and Routes B, C and E once the
record is public.

**Terminology.** The manual ecDNA annotation is called the **gold standard** in
the paper and in this repository. The archive record, which was deposited
earlier, calls the same annotation "ground truth", and file and folder names keep
the short form `gt` (for example `images/gt_image/`). Column names in the frozen
tables (`gt_count`, `gt_path`, ...) are unchanged for the same reason. `gt` and
"gold standard" always refer to the same annotation.

**Command style.** Commands are written for a Unix shell (macOS or Linux). On
Windows, run them in WSL (Windows Subsystem for Linux). Run every command from
the repository folder unless stated otherwise: relative paths in the
configuration are resolved from the folder you run in.

---

## 1. Install

### 1.1 Get the code

```bash
git clone https://github.com/Brunk-Lab/ecdna-bench.git
cd ecdna-bench
```

No git? Download the source archive of the latest release from the
**Releases** page and unpack it.

### 1.2 Create the environment

Pick one of the three options. They install the same library versions as the
environment that produced the paper; only the PyTorch build differs.

**Option 1: conda, no GPU (laptops, macOS on Apple silicon, most workstations)**

```bash
conda env create -f env/environment-cpu.yml
conda activate ecdna-bench-cpu
```

**Option 2: pip, no conda (Python 3.10 must already be installed)**

```bash
python3.10 -m venv ~/venvs/ecdna-bench
source ~/venvs/ecdna-bench/bin/activate
python -m pip install --upgrade pip
python -m pip install -r env/requirements-cpu.txt
```

**Option 3: conda with an NVIDIA GPU (Linux)**

```bash
conda env create -f env/environment.yml
conda activate ecdna-bench
```

> PyTorch 2.8 has no build for Intel-based Macs. On such a machine, use Linux
> or Windows, or a cloud notebook.

### 1.3 Install the package

With the environment active:

```bash
python -m pip install -e . --no-deps
```

`-e` links the installation to this folder, so edits to the code take effect
without reinstalling. `--no-deps` stops pip from changing the versions you just
installed.

### 1.4 Check the installation (Route A)

Three checks, about two minutes in total.

**The test suite** needs no data:

```bash
python -m pytest -q
```

The last line should report tests `passed` and none `failed`. Some tests are
skipped on purpose when data or a GPU is not available.

**The network builds with the published architecture:**

```bash
python -c "
from ecdna_bench.eccount.model import build_model, ModelConfig
m = build_model(ModelConfig())
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
"
```

Expected: `7849601`.

**The frozen result tables match the paper:**

```bash
python scripts/verify_headline_numbers.py
```

Expected: a table of the six methods for all 1,145 benchmark images and for the
175 held-out test images, every row marked `MATCH`, and

```
VERDICT: all 12 model rows MATCH the published values
```

The script compares object-level F1 (three decimals), count mean absolute
error (MAE) and mean signed count bias (one decimal) with the published values.
It reads `release/frozen_results/or_matching/`; the paper uses OR matching
throughout.

---

## 2. Download the model weights

Needed for Routes C, D and E (as a starting point) and for tutorial notebook 2.

1. Open the repository's **Releases** page and download `eccount_best.pt` and
   `SHA256SUMS` from the release assets.
2. Put the checkpoint where the configuration expects it and check it:

```bash
mkdir -p release/model_checkpoints
mv ~/Downloads/eccount_best.pt release/model_checkpoints/
( cd release/model_checkpoints && grep eccount_best.pt ~/Downloads/SHA256SUMS | sha256sum -c - )
```

Expected: `eccount_best.pt: OK`. On macOS use `shasum -a 256 -c -` instead of
`sha256sum -c -`.

The released checkpoint is the model reported in the paper: best validation
loss 0.6421 at epoch 49 of 70.

---

## 3. Route B: re-score the published predictions

This recomputes the comparison of the six methods on the 175 held-out test
images from the deposited prediction masks. No model is run.

### 3.1 Download

`scripts/fetch_bia_subset.py` reads the archive's file lists and downloads only
what you ask for, keeping the archive's folder layout. Check first what it would
fetch:

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
    --sections gt,roi,predictions --dry-run
```

then download:

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
    --sections gt,roi,predictions
```

The last line should read `VERDICT: all selected files are present`. If the
download stops, run the same command again: files already complete are skipped.

**Reviewers** (while the record is private): add
`--base-url "<share link from the reviewer instructions>"`. The share link has
the form
`https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4097?key=<key>`; the
script asks the BioStudies API for the current location of the private files.
The address of the private `Files` folder
(`https://ftp.ebi.ac.uk/pub/databases/biostudies/.private/NN/<key>/S-BIAD/097/S-BIAD4097/Files`)
also works, but it can change while the record is private; the share link does
not. The key is a password: do not paste it into issues, notebooks or shared
scripts. The script never prints it. To keep it out of your shell history, read
it into an environment variable once per terminal (paste the link at the
prompt; nothing is shown):

```bash
printf "share link: "; stty -echo; read -r ECDNA_BIA_BASE_URL; stty echo; echo
export ECDNA_BIA_BASE_URL
```

What the options mean:

| Option | Values | Meaning |
|---|---|---|
| `--split` | `train`, `val`, `test`, `any` | Benchmark partition (default `test`) |
| `--subset` | `benchmark`, `extension`, `any` | `benchmark` = the 1,145 images with manual ROI masks (default); `extension` = the other 1,841 |
| `--sections` | `images,gt,roi,predictions,predicted_roi` | Which parts of the archive |
| `--image-types` | `rgb,dapi` | Which images, when `images` is selected |
| `--gt-types` | `mask,points,sparse` | Which gold-standard representations (default `mask`) |
| `--methods` | e.g. `ecSeg,MIA` | Restrict the prediction masks to some methods |
| `--cell-line` | e.g. `SNU16` | Restrict to a cell line (repeat for several) |
| `--max-images` | a number | At most this many image sets per cell line, for a quick trial |
| `--all` | | Everything in the archive (27,080 files, 54.75 GB) |

### 3.2 Prepare a run folder

The benchmark table shipped with the code
(`release/manifests/dl_master_metadata_stage1_step3_consistency.csv`) records
file locations on the computer that produced the paper. This step writes a copy
with your local locations, converts the prediction masks to the common binary
format, and writes a complete configuration file:

```bash
python scripts/prepare_local_run.py bia --bia-root ~/ecdna_data \
    --out-dir runs/bia_test --split test
```

It prints what it found and the next commands, which are also saved in
`runs/bia_test/NEXT_STEPS.txt`. Every line under `prediction masks:` should
show `0 missing` and `0 shape mismatch`.

### 3.3 Score

Copy the scoring command from `NEXT_STEPS.txt`; it has this form:

```bash
python -m ecdna_bench.cli.benchmark --config runs/bia_test/run_config.yaml \
    --models eccount_peaks eccount_mask label_engine mia classical ecseg classical_before_opt \
    --skip-harmonize --output-dir runs/bia_test/results --n-workers 1
```

`--skip-harmonize` is deliberate: the masks were already converted in step 3.2,
Label Engine's raw output with the benchmark's own rule.

Scoring needs about 10 GB of memory per worker: on the 175 test images one
worker peaked at 10 GB and took about 80 minutes, four workers peaked at 35 GB.
Raise `--n-workers` only if the computer has that much memory;
`prepare_local_run.py bia --n-workers N` writes the matching command and
configuration.

### 3.4 Compare with the paper

```bash
python scripts/verify_headline_numbers.py --results runs/bia_test/results
```

The script recognizes the 175-image scope and compares each method with the
published test-split values:

| Method | Object-level F1 | Count MAE | Mean signed bias |
|---|---|---|---|
| ecCount (peaks) | 0.939 | 13.3 | +1.9 |
| ecCount (threshold mask) | 0.916 | 17.6 | −9.1 |
| Label Engine | 0.825 | 34.4 | −26.9 |
| MIA | 0.813 | 40.4 | −35.7 |
| Classic (after opt) | 0.775 | 40.9 | −17.4 |
| ecSeg | 0.508 | 110.3 | −107.4 |

`Classic (before opt)` is printed as `not in the paper table`; it is the
classical pipeline before parameter optimization.

The results folder holds `or_matching/` and `and_matching/`. The files in the two
have the same names; the paper uses `or_matching/`. Inside, `summary_overall.csv`
is the pooled summary, `per_image_metrics.csv` has one row per image and method,
and `summary_by_cell_line.csv` and `summary_by_density_bin.csv` stratify the
results (`density_bin` in file names means the gold-standard count bins of the
paper).

**All 1,145 images.** Download with `--split any` instead of `--split test`,
prepare with `--split all`, and the check compares with the all-image table
(ecCount peaks: 0.942, 13.1, +0.4).

---

## 4. Route C: run ecCount yourself and score it

### 4.1 Download

The RGB images are needed in addition to the files of Route B:

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
    --sections images,gt,roi --image-types rgb
```

(If you already did Route B into the same folder, only the images are added.)

### 4.2 Prepare, run, score

```bash
python scripts/prepare_local_run.py bia --bia-root ~/ecdna_data \
    --out-dir runs/own_test --split test --own-eccount \
    --checkpoint release/model_checkpoints/eccount_best.pt --device cpu

python -m ecdna_bench.cli.run_eccount --config runs/own_test/run_config.yaml \
    --split test --device cpu

python -m ecdna_bench.cli.benchmark --config runs/own_test/run_config.yaml \
    --models eccount_peaks eccount_mask --skip-harmonize \
    --output-dir runs/own_test/results_own_eccount --n-workers 1

python scripts/verify_headline_numbers.py --results runs/own_test/results_own_eccount
```

With a GPU, use `--device cuda` in the first two commands.

`run_eccount` writes two masks per image, into
`runs/own_test/own_eccount/eccount_peaks/` and
`runs/own_test/own_eccount/eccount_mask/`; its last line reports `ok=175` and
`err=0`. If `err` is not zero, run it again with `--log-level DEBUG` to see why. A second run skips finished images; add
`--force` to recompute them.

Inference takes several seconds per image on a CPU (about 6 s on two cores in
our tests on a two-core virtual machine), so the 175 test images take roughly
20 minutes.

**What to expect.** ecCount (peaks) and ecCount (threshold mask) should
reproduce the test-split values in the table of section 3.4. Floating-point
arithmetic differs slightly between CPUs and GPUs, so a handful of peaks can
change; if the check reports `DIFFERS`, compare the printed values with the
published ones: a difference in the last digit comes from the hardware, a larger
one points to a different checkpoint or configuration.

**What the model does to an image**, in order (the same steps are shown one by
one in tutorial notebook 2):

1. the RGB image is downsampled by two, to 1,024 × 1,224 px, with area
   interpolation;
2. pixels outside the ROI mask are set to zero;
3. 8-bit intensities are divided by 255;
4. the network outputs a probability map;
5. the map is smoothed (Gaussian, σ = 0.5 px), restricted to the ROI again, and
   local maxima above 0.35 are kept (minimum distance 2 px);
6. peak positions are scaled back to the native 2,448 × 2,048 px frame and each
   is drawn as a 5 × 5-pixel diamond, the footprint of a gold-standard point:
   this is the **peaks** output;
7. the probability map thresholded at 0.5 and resized to the native frame is the
   **threshold mask** output.

All values come from the `eccount:` section of `configs/default.yaml`.

---

## 5. Route D: count ecDNA in your own images

### 5.1 What your images should look like

- RGB composites of the FISH probe channel, like the files in `images/rgb/` of
  the archive. The model was trained on metaphase spreads imaged at ×60 on an
  Echo Revolution microscope, 2,448 × 2,048 px.
- Images of another size are resized to the network input, which changes the
  apparent size of ecDNA signals. The preparation step flags such images.
- One ROI mask per image is strongly recommended: a binary image of the same
  size, white inside the metaphase spread, named like the image
  (`spread_01.tif` and `spread_01.png`, say). Without it the whole field is
  analyzed, and neighboring nuclei and debris produce extra detections.

Check a few results by eye before relying on the counts, especially for images
that differ from the resource (other probes, microscopes or magnifications).

### 5.2 Run

```bash
python scripts/prepare_local_run.py images --images /path/to/my_images \
    --roi-dir /path/to/my_rois --out-dir runs/my_images \
    --checkpoint release/model_checkpoints/eccount_best.pt --device cpu

python -m ecdna_bench.cli.run_eccount --config runs/my_images/run_config.yaml \
    --split all --device cpu

python scripts/prepare_local_run.py count \
    --masks runs/my_images/eccount/eccount_peaks \
    --out runs/my_images/ecdna_counts.csv
```

Options of the first command: `--require-roi` skips images without an ROI mask;
`--recursive` looks into sub-folders; `--exclude REGEX` ignores matching file
names (for example `--exclude dapi` if DAPI images sit in the same folder).

`ecdna_counts.csv` has one row per image. The count is the number of
8-connected components of at least 3 px in the peaks mask, which is how every
count in the paper is defined.

### 5.3 Look at the result

Tutorial notebook 2, section 4, runs the same steps on a folder of images and
shows the probability map and the detected peaks over the image
(`MY_IMAGES` and `MY_ROIS` point it at your folders).

### 5.4 Images without an ROI mask

The archive contains ROI masks predicted by a separate model for all 2,986
image sets (`predicted_roi/`), and `prepare_local_run.py bia --roi predicted`
uses them in place of the manual masks. To predict ROI masks for new images,
see the section on the ROI model in the README.

---

## 6. Route E: retrain ecCount

The published model was trained for 70 epochs on one NVIDIA GPU, at about 80 s
per epoch (`train_history.csv` of the release). Training on a CPU is not
practical.

### 6.1 Download all benchmark images

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split any \
    --sections images,gt,roi --image-types rgb
```

### 6.2 Prepare and train

```bash
python scripts/prepare_local_run.py bia --bia-root ~/ecdna_data \
    --out-dir runs/retrain --split all --own-eccount

python -m ecdna_bench.cli.train_eccount --config runs/retrain/run_config.yaml
```

Training uses the rows of `runs/retrain/manifest_local.csv` whose `split` is
`train`, selects the checkpoint by the loss on the `val` rows, and never reads
the `test` rows. It writes `best_model.pt` and `train_history.csv` (one row per
epoch; you can watch it while training runs) to `runs/retrain/eccount_training/`.
Every hyperparameter comes from `configs/default.yaml`, copied into
`run_config.yaml`; the published run used fp32 (`amp: false`), seed 42, batch
size 2 and Adam at a learning rate of 1 × 10⁻⁴.

**Another split** is a change to the `split` column of `manifest_local.csv`;
nothing else needs editing. Tutorial notebook 4 builds such a split and checks
it.

### 6.3 Leave one cell line out

`scripts/train_eccount_loco.py` trains on three cell lines and evaluates on the
fourth, with guards that stop if the held-out line leaks into training. Check the
splits first (two seconds, no GPU):

```bash
python scripts/train_eccount_loco.py --hold-out all --dry-run \
    --config runs/retrain/run_config.yaml
```

The dry run also checks that your download is complete: it expects 1,145 image
sets carrying 228,039 gold-standard objects, and stops otherwise. Then, one run
per held-out line:

```bash
python scripts/train_eccount_loco.py --hold-out SNU16 \
    --config runs/retrain/run_config.yaml --out-root runs/loco
```

It prints the inference and scoring commands for that run when it finishes.
`--size-matched-control` trains the control used in the paper for NCI-H2170,
which makes up 888 of the 1,145 benchmark images.

### 6.4 Use your checkpoint

Point Route C or D at it with `--checkpoint runs/retrain/eccount_training/best_model.pt`.

---

## 7. The tutorial notebooks

`notebooks/tutorials/` contains four notebooks that explain the resource and
the methods step by step:

| Notebook | Content |
|---|---|
| `01_images_and_gold_standard.ipynb` | One image set: RGB, DAPI, ROI and gold standard; how points become objects; your own point annotations |
| `02_run_eccount.ipynb` | ecCount on one image, step by step; then a folder of your own images |
| `03_score_against_gold_standard.ipynb` | Object extraction, matching, TP/FP/FN/ignored; OR versus AND; pooled versus per-image F1; scoring a new method |
| `04_retrain_eccount.ipynb` | The Gaussian training target; a leave-one-cell-line-out split; a short training run |

Start Jupyter from the repository folder with the environment active:

```bash
python -m ipykernel install --user --name ecdna-bench-cpu --display-name "ecdna-bench (CPU)"
jupyter lab
```

and choose that kernel in each notebook. Without data, every notebook runs on a
small synthetic image so that you can check the installation. For real data, set
the data folder before starting Jupyter:

```bash
export ECDNA_DATA_ROOT=~/ecdna_data
jupyter lab
```

Other settings the notebooks read: `ECCOUNT_WEIGHTS` (checkpoint path, if not in
`release/model_checkpoints/`), `ECDNA_RESULTS` (a results folder for notebook 3),
`MY_IMAGES` and `MY_ROIS` (your folders, notebook 2).

The figure notebooks in `notebooks/` (01 to 05) regenerate the paper's figures
and their source tables. They read the archive through `ECDNA_DATA_ROOT` (set
above) and need the full archive, not only the test split. Before the first run:

```bash
python scripts/build_harmonized_masks.py   # baseline masks from predictions/
```

and place the released checkpoint at `release/model_checkpoints/eccount_best.pt`.
The notebooks write into `release/figures/notebookNN/`; in a clone of the
repository, `git diff release/figures` shows whether anything changed and
`git checkout -- release/figures` restores the released files. Alternatively, set
the output folder in the first cell to a folder of your own. The live ecCount
step in notebook 03 needs a GPU and is skipped without one.

---

## 8. Troubleshooting

**`ModuleNotFoundError: No module named 'ecdna_bench'`.** The environment is not
active, or the package is not installed in it: `conda activate ecdna-bench-cpu`
(or `source ~/venvs/ecdna-bench/bin/activate`), then
`python -m pip install -e . --no-deps`.

**`Config file not found: configs/benchmark.yaml`.** Always pass `--config`
explicitly, as in the commands above.

**`Config key 'paths.consistency_csv' is not set`.** The command was given
`configs/default.yaml`, which holds no machine paths. Use the `run_config.yaml`
written by `prepare_local_run.py`.

**A setting in `run_config.yaml` seems to be ignored.** The tools merge a file
named `paths.local.yaml` that sits next to the configuration file, and its paths
win. `prepare_local_run.py` refuses to write into a folder that has one; do not
add one to a run folder.

**`Access denied (403)` or `401` from the downloader.** The record is still
private; see the reviewer note in section 3.1.

**`filelist_images.tsv was not found`.** `--base-url` must be the study's
`Files` folder: the URL ends in `/Files`, not in a web page address with
`?key=`.

**Download interrupted.** Run the same command again. `download_log.tsv` in the
output folder lists every file with its status.

**`N row(s) skipped because files are not in the download`.** The preparation
step found benchmark rows whose files you did not download. Harmless for a
partial download; for a complete run, download the missing sections.

**`run_eccount complete` reports `err` above zero.** Rerun with
`--log-level DEBUG`. The usual cause is an image path that cannot be read.

**`Unknown model key 'eccount_threshold'`.** The benchmark keys are
`eccount_peaks` and `eccount_mask` (the threshold mask), `label_engine`, `mia`,
`classical`, `classical_before_opt` and `ecseg`.

**The numbers do not match the paper.** In order of likelihood:
1. an `and_matching/` table was read instead of `or_matching/`;
2. a different scope: the paper quotes all 1,145 images and the 175 test images
   separately;
3. per-image F1 was averaged instead of pooling true positives, false positives
   and false negatives over images (the paper pools: 0.942 pooled against 0.931
   averaged for ecCount peaks);
4. a different checkpoint, or a changed configuration value.

**The kernel is not listed in Jupyter.** Register it (section 7) with the
environment active, then reload the Jupyter page.

**Out of memory while scoring.** Lower `--n-workers`; each worker needs about
10 GB.

---

## 9. Reference

### 9.1 Published values

Pooled over images, OR matching (`d_max = 20 px`, `IoU_min = 0.1`,
`alpha = 0.5`), objects are 8-connected components of at least 3 px.

| Method | All 1,145 images: F1 | MAE | Bias | Test 175 images: F1 | MAE |
|---|---|---|---|---|---|
| ecCount (peaks) | 0.942 | 13.1 | +0.4 | 0.939 | 13.3 |
| ecCount (threshold mask) | 0.917 | 18.1 | −11.4 | 0.916 | 17.6 |
| Label Engine | 0.825 | 36.0 | −28.8 | 0.825 | 34.4 |
| MIA | 0.800 | 47.0 | −40.6 | 0.813 | 40.4 |
| Classic (after opt) | 0.777 | 44.9 | −24.3 | 0.775 | 40.9 |
| ecSeg | 0.464 | 121.5 | −118.5 | 0.508 | 110.3 |

Source: `release/frozen_results/or_matching/summary_overall.csv` and
`summary_by_split.csv`.

### 9.2 The resource

| | |
|---|---|
| Full resource | 2,986 image sets, five cell lines |
| Benchmark | 1,145 image sets with manual ROI masks, four cell lines |
| Partitions | 800 training / 170 validation / 175 test |
| Image size | 2,448 × 2,048 px |
| ecCount | 7,849,601 parameters; input 1,024 × 1,224 px; best epoch 49, validation loss 0.6421 |

### 9.3 Archive layout (S-BIAD4097, `Files/`)

```
filelist_images.tsv  filelist_gt.tsv  filelist_roi.tsv
filelist_predictions.tsv  filelist_predicted_roi.tsv
images/
    rgb/          probe-channel RGB images (TIFF)
    dapi/         DAPI images (8-bit PNG)
    gt_image/     gold-standard masks, one 5 x 5-pixel diamond per point (0/255)
    gt_coords/    gold-standard point coordinates (NumPy)
    gt_npz/       gold-standard masks, sparse (NumPy .npz)
    roi_mask/     manual ROI masks (benchmark images only)
predictions/
    eccount_peaks/  eccount_threshold/  label_engine/  mia/
    classical_optimised/  classical_default/  ecseg/
predicted_roi/    model-predicted ROI masks, all 2,986 image sets
splits/           train_ids.csv  val_ids.csv  test_ids.csv
```

`classical_optimised` is `Classic (after opt)` and `classical_default` is
`Classic (before opt)`. Every file is named by the image set's unique
identifier. See `DATASET.md` for the full description.

### 9.4 Scripts used in this tutorial

| Script | Purpose |
|---|---|
| `scripts/fetch_bia_subset.py` | Download all or part of S-BIAD4097 (standard library only) |
| `scripts/prepare_local_run.py` | `bia`: run folder for archive data; `images`: run folder for your images; `count`: count objects in masks |
| `scripts/verify_headline_numbers.py` | Compare a results folder with the published values (standard library only) |
| `scripts/train_eccount_loco.py` | Leave-one-cell-line-out training with split guards |

### 9.5 Getting help

Open an issue on the GitHub repository with the command you ran, the last
twenty lines of its output, your operating system and
`python -c "import torch, numpy; print(torch.__version__, numpy.__version__)"`.
Never include the private download location.
