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
| Deposition size | 24,095 files, 50.9 GiB |
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

Each image is identified by a **unique_id** (UID) — a lowercase string
encoding the cell line and the experimental condition, for example
`ncih2170_facs_fish_0723_low_her2_52` or `colo320dm_qpcr_jq1_24h_ctrl_100`.

Directories are flat: the UID alone identifies a file within each directory,
and the same UID is used across every directory that contains that image.

```
data_root/
├── bioimage_archive/              # full resource — 2,986 images
│   ├── rgb/{uid}.tif              # 3-channel RGB FISH image, 2448 × 2048 px
│   ├── dapi/{uid}.png             # 1-channel DAPI image
│   ├── gt_image/{uid}.png         # binary GT mask, {0, 255}
│   ├── gt_coords/{uid}.npy        # GT ecDNA centroid coordinates
│   ├── gt_npz/{uid}.npz           # packed GT arrays
│   ├── roi_mask/{uid}.png         # binary ROI mask, {0, 255} — 1,145 images
│   └── metadata_internal.csv
└── benchmark/                     # fixed 1,145-image benchmark
    ├── predictions/
    │   ├── classical_default/{uid}.png
    │   ├── classical_optimised/{uid}.png
    │   ├── eccount_peaks/{uid}.png
    │   ├── eccount_threshold/{uid}.png
    │   ├── ecseg/{uid}.png
    │   ├── label_engine/{uid}.png
    │   └── mia/{uid}.tif
    ├── splits/{train,val,test}_ids.csv
    └── metadata_internal.csv
```

`rgb/`, `dapi/`, `gt_image/`, `gt_coords/` and `gt_npz/` each contain 2,986
files, one per image in the full resource. `roi_mask/` contains 1,145 files:
the remaining 1,841 images have no manual ROI mask and are therefore not part
of the benchmark. Each of the seven prediction directories contains exactly
1,145 masks, one per benchmark image.

All masks are uint8 with pixel values {0, 255}. RGB images are the standard
8-bit fluorescence composite.

---

## Image formats

Two formats are used. Files that are primary acquisitions or the output of
external tools are archived exactly as they were produced; files derived by
this project's own tooling are stored as lossless PNG.

| Path | Files | Format | Size |
|------|-------|--------|------|
| `bioimage_archive/rgb/` | 2,986 | TIFF | 41.84 GiB |
| `benchmark/predictions/mia/` | 1,145 | TIFF | 5.35 GiB |
| `bioimage_archive/dapi/` | 2,986 | PNG | 3.18 GiB |
| `bioimage_archive/gt_npz/` | 2,986 | NPZ | 0.44 GiB |
| `benchmark/predictions/` (six others) | 6,870 | PNG | 0.10 GiB |
| `bioimage_archive/gt_image/` | 2,986 | PNG | 0.02 GiB |
| `bioimage_archive/roi_mask/` | 1,145 | PNG | 0.01 GiB |
| `bioimage_archive/gt_coords/` | 2,986 | NPY | < 0.01 GiB |
| metadata and split CSVs | 5 | CSV | < 0.01 GiB |

4,131 files are TIFF and 13,987 are PNG. The deposition totals 24,095 files
and 50.9 GiB.

### RGB images

The RGB FISH images are the primary record of the resource. Every benchmark
result reported in the accompanying paper was computed by reading these files
as deposited, so the SHA256 digests in `manifest_v1.0.csv` identify the exact
bytes those results derive from.

Conversion to PNG was measured rather than assumed: a 20-image random sample
gave a mean compression ratio of 0.164, projecting 6.86 GiB against the
41.84 GiB retained. Retaining TIFF therefore costs approximately 35 GiB,
accepted so that the archived object remains the acquisition itself rather
than a derivative whose losslessness a user cannot independently check.

TIFF is a standard bioimaging container accepted by the BioImage Archive, so
retention costs storage and transfer time, not accessibility.

### MIA prediction masks

The MIA masks are the output of an external baseline and are archived in the
format that tool produced. They are uint8 binary masks at 2448 × 2048 px and
are therefore highly compressible; retaining them as TIFF costs approximately
5.3 GiB relative to PNG. The six prediction sets generated by this project's
own code are written as PNG and total 0.10 GiB.

### Converted directories

`dapi/` — 2,984 of 2,986 files already contained PNG bytes under a `.tif`
extension. Conversion corrected the container-extension mismatch; for those
files no image data was re-encoded.

`gt_image/` — uncompressed binary TIFF at 13.9 GB, reduced losslessly to
24.0 MB, a factor of approximately 580. These masks are derived from the
annotation pipeline and remain reconstructible from `gt_coords/`.

Both conversions were verified pixel-for-pixel **and** by inspecting the
output file's magic bytes before any source file was removed, so every
converted file is a genuine PNG rather than a re-containered TIFF.

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

Fixed train/val/test splits at the UID level:

```
benchmark/splits/            # in the data deposition
├── train_ids.csv            # 800 UIDs
├── val_ids.csv              # 170 UIDs
└── test_ids.csv             # 175 UIDs
```

The same three files are committed in the code repository at
`release/split_files/`.

**Splitting criteria:** stratified by cell line and ecDNA density bin.
The test split is held out from all hyperparameter tuning and model selection.
The validation split is used for model selection (ecCount best checkpoint,
Bayesian-optimisation stopping).

---

## Downloading the data

The imaging data and the ecCount checkpoint are deposited at the BioImage
Archive (accession `{{ACCESSION}}`). The deposition has two top-level
directories, `bioimage_archive/` and `benchmark/`; both are required to
reproduce the benchmark. Follow the download instructions on the accession
page, then verify integrity:

```python
from ecdna_bench.utils.checksums import verify_manifest

failures = verify_manifest(
    "release/manifests/manifest_v1.0.csv",
    root="/path/to/data_root",
)
print("OK" if not failures else failures[:10])
```

An empty result means the download matches the manifest exactly. A further
consistency check on the benchmark subset is available via:

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

Columns: `relative_path`, `sha256`, `size_bytes`, `file_type`. Paths are
relative to `data_root`. The manifest covers 24,095 files totalling 50.9 GiB
and is the authoritative integrity record for the deposition.

**Scope.** Every file beneath `bioimage_archive/` and `benchmark/` is hashed.
The following patterns are never hashed, being backups, editor artefacts, or
the residue of interrupted conversions:

```
*.bak   *.bak_pre_png   *.tmp   *.png.tmp   *~
.DS_Store   Thumbs.db   *.swp   __pycache__/*   *.pyc
```

Inspect the scope without hashing, or regenerate the manifest:

```bash
# report scope and sizes only
python scripts/generate_manifest.py --data-root /path/to/data_root --list

# regenerate (approximately 100 minutes for the full deposition)
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