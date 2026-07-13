# External Baselines — Provenance and Reproduction Notes

This document describes the three external baseline models compared against
ecCount in Figure 6 of the paper.  It records provenance, settings, and
instructions for anyone wishing to reproduce the prediction masks.

---

## Overview

| Model | Role in paper | Prediction masks shipped? |
|---|---|---|
| Label Engine | Segmentation baseline trained on same data | ✅ |
| ecSeg | Public FISH segmentation model | ✅ |
| MIA | Counting baseline from MIA authors | ✅ |

All prediction masks are released as binary uint8 PNGs (values {0, 255})
in the harmonized format used by the evaluation framework.  The raw
(pre-harmonization) masks are included in the data release for provenance.

---

## Label Engine

### What it is
Label Engine is a supervised segmentation model trained by a former lab
member on the same labeled dataset and the same train/val/test splits used
for ecCount.  It was trained prior to the ecCount project and its
architecture details are described in the companion publication.

### Who ran it
Former lab member (name in author list).  Prediction masks were generated
on the test split using the best checkpoint from their final training run.

### Settings
- Architecture: Label Engine v1 (see companion paper)
- Training split: same 800-image train split as ecCount
- Inference: no test-time augmentation
- Output: single-channel probability map, thresholded at 0.5

### Harmonization
```python
from ecdna_bench.baselines.label_engine import harmonize_label_engine
results = harmonize_label_engine(
    pred_dir   = Path("release/raw_masks/label_engine"),
    output_dir = Path("release/harmonized_masks/label_engine"),
    uid_list   = test_uids,
    threshold  = 0.5,
    min_area   = 3,
)
```

### Filename convention
Label Engine files are named ``{uid}_LE.png``.  The harmonizer applies
the canonicalization logic from ``ecdna_bench.baselines.label_engine``
(``control`` → ``ctrl``, trailing suffixes ``_a/_b/_f/_s`` stripped, etc.).

### SHA256 of prediction directory
```
TODO: insert SHA256 of release/raw_masks/label_engine/ before submission
```

---

## ecSeg

### What it is
ecSeg (Deshpande et al. 2019, *Nature Genetics*) is a public multi-class
segmentation model trained on FISH microscopy images from a different lab.
It classifies pixels into four classes: background, nucleus, chromosome,
and ecDNA.

- **Paper:** Deshpande I. et al. (2019). *Nat Genet* 51, 1309–1318.
- **Code:** https://github.com/ucrajesh/ecSeg

### Who ran it
Corresponding author ran the public ecSeg model on all 1,145 benchmark
images using the default pre-trained weights.

### Settings
- Model: ecSeg v1, default pre-trained weights
- Input: original-resolution RGB FISH images
- Output: multi-channel PNG, channel 3 = ecDNA class
- No retraining; no fine-tuning

### Harmonization
```python
from ecdna_bench.baselines.ecseg import harmonize_ecseg
results = harmonize_ecseg(
    pred_dir   = Path("release/raw_masks/ecseg"),
    output_dir = Path("release/harmonized_masks/ecseg"),
    uid_list   = all_uids,
    channel    = 3,    # ecDNA class index
    min_area   = 3,
)
```

### Verification
The ecSeg harmonizer includes ``audit_ecseg_coverage`` which counts
foreground pixels per image after channel extraction.  This count was
verified to match ecSeg's own internal quantification (object-count MAE = 0
against ecSeg's CSV report).

### SHA256 of prediction directory
```
TODO: insert SHA256 of release/raw_masks/ecseg/ before submission
```

---

## MIA

### What it is
MIA (Multiple Instance Analysis) is a weakly-supervised ecDNA counting
model developed by Kim et al.

- **Paper:** Kim et al. (TODO: citation when available)
- The MIA authors provided prediction masks directly from their
  top-performing configuration.

### Who ran it
MIA authors.  No code was run by the ecCount team; the authors supplied the
prediction files.

### Settings
- Configuration: top-performing MIA config (details in MIA paper)
- Input: same benchmark image set
- Output: binary or near-binary prediction masks
- No retraining on our dataset

### Harmonization
```python
from ecdna_bench.baselines.mia import harmonize_mia
results = harmonize_mia(
    pred_dir   = Path("release/raw_masks/mia"),
    output_dir = Path("release/harmonized_masks/mia"),
    uid_list   = all_uids,
    threshold  = 0.0,   # MIA masks are already binary
    min_area   = 3,
)
```

### SHA256 of prediction directory
```
TODO: insert SHA256 of release/raw_masks/mia/ before submission
```

---

## Harmonized mask format

All harmonized masks (and all ecCount output masks) conform to:

| Property | Value |
|---|---|
| Format | PNG |
| Channels | 1 (grayscale) |
| Dtype | uint8 |
| Foreground | 255 |
| Background | 0 |
| Resolution | Original full-resolution (same as GT mask) |
| Min CC area | 3 px (matching evaluation framework) |

---

## Reproducing the harmonized masks from scratch

```bash
# 1. Install the package
pip install -e .

# 2. Download raw prediction masks from BioImage Archive (TODO: DOI)
# Place under release/raw_masks/{label_engine,ecseg,mia}/

# 3. Run harmonizers
python -m ecdna_bench.cli.run_baseline --model label_engine
python -m ecdna_bench.cli.run_baseline --model ecseg
python -m ecdna_bench.cli.run_baseline --model mia

# Output: release/harmonized_masks/{label_engine,ecseg,mia}/
```

---

*Last updated: 2026-04-23*
