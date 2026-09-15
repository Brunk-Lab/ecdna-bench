# Dataset card: the ecdna-bench imaging resource

## Summary

| Property | Value |
|---|---|
| Content | Metaphase FISH images of ecDNA with manual annotation of every ecDNA signal |
| Image sets | 2,986 (full resource); 1,145 with manual ROI masks (benchmark) |
| Cell lines | NCI-H2170, SUM159PT, SNU16, COLO320DM, NCI-H716 (benchmark: all except SUM159PT) |
| Modalities per image set | probe-channel RGB image and DAPI image of the same field, 2,448 × 2,048 px, ×60 |
| Annotation | gold standard: one point per ecDNA signal, released as points, rendered masks and sparse masks |
| Gold-standard objects | 493,663 in the full resource; 228,039 in the benchmark |
| Benchmark partitions | 800 training / 170 validation / 175 test, fixed at the image-set level |
| Also deposited | manual ROI masks (1,145), predicted ROI masks (2,986), prediction masks of seven method outputs (1,145 each) |
| Archive | BioImage Archive **S-BIAD4097**, <https://doi.org/10.6019/S-BIAD4097>; 27,080 files, 54.75 GB |
| License | CC BY 4.0 |

**Terminology.** "Gold standard" (GS) is the manual annotation. The archive
record uses the earlier term "ground truth" for the same annotation, and file and
folder names use `gt`.

---

## Access

The record is private until **1 September 2027**. Until then, editors and
reviewers download it through the private link in the reviewer instructions.
After release, the files are at
`https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/097/S-BIAD4097/Files/`
and can be downloaded over HTTPS, FTP, Aspera or Globus, following the BioImage
Archive's download instructions.

To download part of the record with the folder layout intact:

```bash
python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test --sections gt,roi,predictions
```

(`--help` lists all options; `--dry-run` shows what would be downloaded.) The
browser file table of the archive selects at most 1,000 files at a time; the
script and the FTP, Aspera and Globus routes have no such limit.

---

## Composition

| Cell line | Full resource | Benchmark | Train / val / test |
|---|---|---|---|
| NCI-H2170 | 1,891 | 888 | 621 / 133 / 134 |
| SNU16 | 273 | 123 | 86 / 18 / 19 |
| NCI-H716 | 139 | 70 | 49 / 10 / 11 |
| COLO320DM | 225 | 64 | 44 / 9 / 11 |
| SUM159PT | 458 | 0 | — |
| **Total** | **2,986** | **1,145** | **800 / 170 / 175** |

Source: `release/figures/notebook01/source_full_resource_summary.csv`.

SUM159PT has no manual ROI masks and is therefore not part of the benchmark,
nor of any model training, validation or evaluation. It is released for reuse,
with predicted ROI masks.

### Gold-standard counts in the benchmark

| Cell line | Mean | Median | Min | Max |
|---|---|---|---|---|
| NCI-H2170 | 199.2 | 153 | 9 | 1,687 |
| SNU16 | 283.5 | 241 | 57 | 1,231 |
| NCI-H716 | 190.6 | 174 | 71 | 468 |
| COLO320DM | 46.0 | 43 | 1 | 144 |
| **All** | **199.2** | **159** | **1** | **1,687** |

| Count bin (objects per image) | Image sets | Share of the benchmark |
|---|---|---|
| 0–9 | 7 | 0.6 % |
| 10–49 | 79 | 6.9 % |
| 50–149 | 450 | 39.3 % |
| 150–299 | 403 | 35.2 % |
| ≥ 300 | 206 | 18.0 % |

Sources: `release/figures/notebook01/source_benchmark_count_summary.csv`,
`release/figures/notebook01/source_density_bin_summary.csv` (file and column
names call the count bins `density_bin`).

---

## Archive layout

Paths are relative to the record's `Files/` folder. Every file is named by the
image set's **unique identifier**, which is shared by all modalities and
annotations of one metaphase spread. Identifiers have the form
`<cell line>_<condition tokens>_<image index>[_<qualifier>]`, for example
`ncih2170_facs_fish_0723_low_her2_52`.

```
Files/
├── filelist_images.tsv          images, index table and split files
├── filelist_gt.tsv              gold-standard annotations
├── filelist_roi.tsv             manual ROI masks
├── filelist_predictions.tsv     method prediction masks
├── filelist_predicted_roi.tsv   predicted ROI masks
├── images/
│   ├── rgb/          <uid>.tif   probe-channel RGB image, 8-bit, 2,048 × 2,448 × 3
│   ├── dapi/         <uid>.png   DAPI image, 8-bit
│   ├── gt_image/     <uid>.png   gold-standard mask, 8-bit, values 0 and 255
│   ├── gt_coords/    <uid>.npy   gold-standard point coordinates
│   ├── gt_npz/       <uid>.npz   gold-standard mask, sparse (compressed NumPy)
│   ├── roi_mask/     <uid>.*     manual ROI mask, binary (benchmark image sets only)
│   └── metadata_internal.csv     image index: cell line, condition, file names
├── predictions/
│   ├── eccount_peaks/            ecCount (peaks)
│   ├── eccount_threshold/        ecCount (threshold mask)
│   ├── label_engine/             Label Engine
│   ├── mia/                      MIA
│   ├── classical_optimised/      Classic (after opt)
│   ├── classical_default/        Classic (before opt), the classical pipeline before optimization
│   └── ecseg/                    ecSeg
├── predicted_roi/                predicted ROI masks, all 2,986 image sets
└── splits/
    ├── train_ids.csv             800 identifiers
    ├── val_ids.csv               170 identifiers
    └── test_ids.csv              175 identifiers
```

Each file list is tab-separated. `Files` is the path of the file; the other
columns give the data or annotation type, the cell line, the experimental
condition, the subset (`benchmark` or `extension`), the partition, the
gold-standard count and, for annotations and predictions, the `source_image`
they belong to and the `Method`.

| File list | Rows |
|---|---|
| `filelist_images.tsv` | 5,976 |
| `filelist_gt.tsv` | 8,958 |
| `filelist_roi.tsv` | 1,145 |
| `filelist_predictions.tsv` | 8,015 |
| `filelist_predicted_roi.tsv` | 2,986 |

---

## The gold standard

**Annotation.** ecDNA signals were annotated in Fiji/ImageJ with the point tool,
on the probe-channel RGB image with the DAPI image for chromosomal context, by
multiple trained operators. One point was placed at the approximate center of
each signal judged to be extrachromosomal. No minimum size was imposed, no
consensus procedure was applied, and no confidence scores were recorded.

**Representations.** The same annotation is released three ways:

- `gt_coords/`: the point coordinates. These files come from several annotation
  sessions and differ in layout (delimiters, column names, column order); read
  them with a check of the coordinate order, as in tutorial notebook 1.
- `gt_image/`: a binary mask in which every point is drawn as a 5 × 5-pixel
  diamond (13 px).
- `gt_npz/`: the same mask in sparse form.

**Objects and counts.** A gold-standard object is an 8-connected component of the
rendered mask with an area of at least 3 px. Diamonds of points closer than a few
pixels touch and form one object, so an image's object count is at most its
point count. All counts in the paper and in the file lists are object counts.
Both numbers are in the benchmark table (`ecDNA_gt`: objects;
`coord_count_npy`: points).

**What the diamonds are.** They encode position, not the extent of a signal.
Pixel-level metrics against them measure agreement with this rendering, which is
why pixel-level Dice is low for every method.

---

## ROI masks

**Manual.** For each benchmark image set, one freehand region drawn with a lasso
tool encloses the metaphase spread to be quantified and all its annotated ecDNA,
and excludes unburst nuclei, neighboring cells and debris. Pixels outside the
region were set to zero for every method, and all benchmark evaluation is
restricted to it.

**Predicted.** A residual U-Net with RGB and DAPI input (35,923,337 parameters),
trained on the 800 training and 170 validation image sets of the benchmark,
predicted a region for all 2,986 image sets. Over the benchmark, the predicted
regions have a median IoU of 0.821 and a median Dice coefficient of 0.902 with the
manual ones and contain 98.5 % of the gold-standard objects. Predicted regions
were not used for any benchmark result. They are computational annotations, not
annotations of record.

---

## Prediction masks

Seven outputs for each of the 1,145 benchmark image sets: the six compared
methods and the classical pipeline before parameter optimization. Each is a
binary mask at the native 2,448 × 2,048 px resolution, restricted to the manual
ROI, in the common format used for scoring. They are predictions, not
annotations; object-level F1 over the benchmark ranges from 0.942
(ecCount (peaks)) to 0.464 (ecSeg).

How each was produced: the classical pipeline, Label Engine and ecCount were
fitted to the benchmark training partition; MIA masks are archived outputs of the
original study's model; ecSeg was applied with its released weights, without
retraining. Method details are in the manuscript and in
[`docs/EXTERNAL_BASELINES.md`](docs/EXTERNAL_BASELINES.md).

---

## Partitions

The benchmark is split into fixed lists of identifiers (`splits/` in the archive,
`release/split_files/` in this repository): 800 training, 170 validation and
175 test image sets. Each partition mirrors the cell-line composition of the
benchmark (about 70 : 15 : 15 within every cell line). All modalities and
annotations of an image set are in the same partition, and no identifier is in
more than one. No stratification by gold-standard count was applied.

Every model fitted in this study (the classical pipeline, Label Engine and
ecCount) used these partitions: training images for fitting, validation images
for selection, and test images for reporting only. Image sets from the same slide
or condition can be in different partitions, so the test partition measures
generalization within the imaged conditions; transfer to an unseen cell line was
assessed separately, by leave-one-cell-line-out training.

---

## Quality control

Automated checks confirmed unique identifiers, complete modality coverage,
readable files and consistent dimensions for all 2,986 image sets. No image set
met the pre-defined exclusion criteria (missing modality, unreadable file,
dimension mismatch, missing or unreadable annotation), and none was excluded for
image quality, signal-to-noise ratio, ecDNA burden or annotation difficulty.

---

## The benchmark table in this repository

`release/manifests/dl_master_metadata_stage1_step3_consistency.csv` lists the
1,145 benchmark image sets (one row each) with their cell line, partition,
gold-standard object count (`ecDNA_gt`), point count (`coord_count_npy`), file
checks and file locations. Notes:

- The `*_fullpath` columns are locations on the computer that produced the
  paper. `scripts/prepare_local_run.py bia` writes a copy with the locations of
  your download.
- The `*_read_error` columns are empty when a file was read without error.
- `count_mask_consistent` is true for all 1,145 rows; the tools use only rows
  where it is true.

---

## License and citation

The data are released under the Creative Commons Attribution 4.0 International
license (CC BY 4.0). The code in this repository is under the MIT License.

Please cite the dataset (BioImage Archive S-BIAD4097,
<https://doi.org/10.6019/S-BIAD4097>) and the accompanying article (see
[`CITATION.cff`](CITATION.cff)).
