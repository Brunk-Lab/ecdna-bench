# Dataset Card — ecdna-bench

## Summary

| Property | Value |
|----------|-------|
| Task | ecDNA localisation / segmentation / counting in FISH images |
| Total images | 2,986 (full resource) |
| Benchmark subset | 1,145 (with manually drawn ROI masks) |
| Cell lines | 5 (NCI-H2170, SUM159PT, SNU16, COLO320DM, NCI-H716) |
| Image modality | RGB fluorescence in situ hybridisation (FISH) + DAPI |
| Annotation type | Binary segmentation masks + per-image ecDNA counts |
| Splits | 800 train / 170 val / 175 test (fixed, UID-level) |
| License | CC BY 4.0 |
| Repository | BioImage Archive — accession `{{ACCESSION}}` |
| Version | 1.0.0 |

---

## Cell line breakdown

### Benchmark subset (1,145 images with manual ROI masks)

| Cell line | N (benchmark) | Mean count | Median | Min | Max | Split (train/val/test) |
|-----------|---------------|------------|--------|-----|-----|------------------------|
| NCI-H2170 | 888 | 199 | 153 | 9 | 1,687 | 621 / 133 / 134 |
| SNU16 | 123 | 284 | 241 | 57 | 1,231 | 86 / 18 / 19 |
| NCI-H716 | 70 | 191 | 174 | 71 | 468 | 49 / 10 / 11 |
| COLO320DM | 64 | 46 | 43 | 1 | 144 | 44 / 9 / 11 |
| **Total** | **1,145** | **199** | **159** | **1** | **1,687** | **800 / 170 / 175** |

Source: `release/figures/notebook01/source_benchmark_count_summary.csv`.

### Full resource (2,986 images)

| Cell line | N (total) | Benchmark? |
|-----------|-----------|------------|
| NCI-H2170 | 1,891 | Included (888 with ROI masks) |
| SUM159PT  | 458   | **Not in benchmark subset** (no manual ROI masks) |
| SNU16     | 273   | Included (123 with ROI masks) |
| COLO320DM | 225   | Included (64 with ROI masks) |
| NCI-H716  | 139   | Included (70 with ROI masks) |
| **Total** | **2,986** | 1,145 in benchmark subset |

Source: `release/figures/notebook01/source_full_resource_summary.csv`.

> **SUM159PT exclusion:** SUM159PT images were collected but manual ROI masks
> were not drawn for this cell line prior to submission. These images are
> included in the data release for future use but are excluded from all
> benchmark evaluations.

---

## ecDNA count distribution (benchmark subset, n = 1,145)

| Density bin | Count range | N images | % of benchmark |
|-------------|-------------|----------|----------------|
| Very low  | 0-9     | 7   | 0.6 % |
| Low       | 10-49   | 79  | 6.9 % |
| Medium    | 50-149  | 450 | 39.3 % |
| High      | 150-299 | 403 | 35.2 % |
| Very high | 300+    | 206 | 18.0 % |

Bin edges are inclusive definitions; the lowest per-image count observed in
the benchmark subset is 1.

Source: `release/figures/notebook01/source_density_bin_summary.csv`.

---

## File layout

Each image in the benchmark subset is identified by a **unique_id** (UID).

```
data_root/
├── rgb/
│   └── {cell_line}/{uid}.png           # 3-channel RGB FISH image
├── dapi/
│   └── {cell_line}/{uid}.png           # 1-channel DAPI image
├── gt_masks/
│   └── {cell_line}/{uid}.png           # Binary GT mask: {0, 255}
├── roi_masks/
│   └── {cell_line}/{uid}.png           # Binary ROI mask: {0, 255}
└── mia_masks/                          # Pre-computed MIA baseline masks
    └── {cell_line}/{uid}.png
```

All images are **uint8 PNG** files. GT and ROI masks use pixel values {0, 255}.
RGB images are the standard 8-bit fluorescence composite.

---

## Annotation protocol

- **GT masks:** Manual annotation of individual ecDNA foci in the RGB FISH
  channel. Each foreground component corresponds to one or more ecDNA objects.
- **ROI masks:** Manual delineation of the metaphase-spread region to exclude
  background and out-of-focus areas. Evaluation is performed only within the ROI.
- **Counts:** Derived from GT masks by connected-component analysis
  (8-connectivity, minimum area 3 px). Count = number of components in the GT.
- **Inter-annotator agreement:** see the Online Methods of the accompanying
  paper.

---

## Splits

Fixed train/val/test splits at the UID level are committed in:

```
release/split_files/
├── train_ids.csv    # 800 UIDs
├── val_ids.csv      # 170 UIDs
└── test_ids.csv     # 175 UIDs
```

**Splitting criteria:** stratified by cell line and ecDNA density bin.
The test split is held out from all hyperparameter tuning and model selection.
The validation split is used for model selection (ecCount best checkpoint,
Bayesian-optimisation stopping).

---

## Downloading the data

The imaging data and the ecCount checkpoint are deposited at the BioImage
Archive (accession `{{ACCESSION}}`). Follow the download instructions on the
accession page, then verify integrity:

```bash
python -m ecdna_bench.cli.run_qc --config configs/default.yaml
```

The QC step verifies file existence, readability, shape consistency, and
count-mask consistency for all 1,145 benchmark images.

---

## SHA256 manifest

A per-file SHA256 manifest is provided at:

```
release/manifests/manifest_v1.0.csv
```

It covers every released image, mask, and prediction directory, and is the
authoritative integrity record for the deposition.

Regenerate it with:

```bash
python scripts/generate_manifest.py \
    --data-root /path/to/data_root \
    --out release/manifests/manifest_v1.0.csv
```

---

## License

The dataset is released under **Creative Commons Attribution 4.0 International
(CC BY 4.0)**. You are free to share and adapt the material for any purpose,
provided appropriate credit is given.

See [`LICENSE`](LICENSE) for the code license (MIT).
