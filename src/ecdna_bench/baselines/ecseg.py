"""
ecdna_bench.baselines.ecseg
============================
Harmonization adapter for ecSeg prediction masks.

ecSeg background
----------------
ecSeg (Deshpande et al.) is a multi-class segmentation model trained on FISH
images.  When run on ecDNA FISH images it produces a multi-channel PNG where
channel index 3 corresponds to the ecDNA class.  The raw output PNG may have
3 or 4 channels; in either case the ecDNA signal lives at index 3 (0-based)
when the PNG is treated as a stacked label image or read as RGBA.

Harmonization pipeline per image
---------------------------------
1. Load the prediction PNG with ``cv2.IMREAD_UNCHANGED`` (preserves all channels).
2. Extract channel 3 (ecDNA class).
3. Binarize: foreground = channel_3 > 0.
4. Apply min-area-3 connected-component filter (matching §2 of REWRITE_PLAN.md).
5. Save as uint8 PNG with values {0, 255}.

Shared helpers in this module
------------------------------
``load_and_binarize``, ``apply_min_area_filter``, ``write_binary_mask`` are
re-exported from this module and used by ``mia.py`` and ``label_engine.py``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from skimage.measure import label, regionprops

__all__ = [
    # Public harmonizer
    "harmonize_ecseg",
    # Shared helpers (used by mia.py and label_engine.py)
    "load_and_binarize",
    "apply_min_area_filter",
    "write_binary_mask",
    # Audit helper
    "audit_ecseg_coverage",
]

logger = logging.getLogger(__name__)

# ecSeg channel index for the ecDNA class (0-based)
ECSEG_ECDNA_CHANNEL: int = 3


# ---------------------------------------------------------------------------
# Shared helpers (reused across all three harmonizers)
# ---------------------------------------------------------------------------

def load_and_binarize(
    path: Path,
    threshold: float = 0.0,
    channel: Optional[int] = None,
) -> np.ndarray:
    """Load an image file and return a binary (0/1) uint8 mask.

    Parameters
    ----------
    path:
        Path to a PNG (or any cv2-readable image file).
    threshold:
        Pixels strictly above this value are foreground.
    channel:
        If not None, extract this channel index before binarizing.
        If None and the image is multi-channel, convert to grayscale first.

    Returns
    -------
    np.ndarray of shape (H, W), dtype uint8, values {0, 1}.
    """
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    if channel is not None:
        if img.ndim == 2:
            raise ValueError(
                f"channel={channel} requested but image at {path} is single-channel"
            )
        if channel >= img.shape[2]:
            raise ValueError(
                f"channel={channel} out of range for image with {img.shape[2]} channels: {path}"
            )
        img = img[:, :, channel]
    elif img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]

    return (img.astype(np.float32) > threshold).astype(np.uint8)


def apply_min_area_filter(
    binary: np.ndarray,
    min_area: int = 3,
) -> np.ndarray:
    """Remove connected components smaller than *min_area* pixels.

    This matches the minimum CC area used in the main evaluation framework
    (§2 of REWRITE_PLAN.md: min CC area = 3 px).

    Parameters
    ----------
    binary:
        2-D array with values {0, 1}.
    min_area:
        Minimum pixel area to retain.

    Returns
    -------
    np.ndarray of shape (H, W), dtype uint8, values {0, 1}.
    """
    if min_area <= 0:
        return binary.astype(np.uint8)

    labeled = label(binary, connectivity=2)
    out     = np.zeros_like(binary, dtype=np.uint8)
    for region in regionprops(labeled):
        if region.area >= min_area:
            out[labeled == region.label] = 1
    return out


def write_binary_mask(
    binary: np.ndarray,
    output_path: Path,
) -> None:
    """Write a {0, 1} binary mask as a uint8 PNG with pixel values {0, 255}.

    Parameters
    ----------
    binary:
        2-D uint8 array with values {0, 1}.
    output_path:
        Destination PNG path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mask_255 = (binary * 255).astype(np.uint8)
    tmp = str(output_path) + ".tmp.png"
    cv2.imwrite(tmp, mask_255)
    os.replace(tmp, str(output_path))


# ---------------------------------------------------------------------------
# ecSeg harmonizer
# ---------------------------------------------------------------------------

def harmonize_ecseg(
    pred_dir:   Path,
    output_dir: Path,
    uid_list:   List[str],
    min_area:   int = 3,
    channel:    int = ECSEG_ECDNA_CHANNEL,
) -> Dict[str, str]:
    """Harmonize ecSeg predictions to canonical binary masks.

    For each UID, the function:
    1. Looks for a prediction PNG named ``{uid}.png`` or ``{uid}_ecseg_ecdna_mask.png``
       in *pred_dir*.
    2. Extracts ``channel`` (default 3 = ecDNA class) from the multi-channel PNG.
    3. Binarizes and applies min-area-3 filter.
    4. Writes ``{output_dir}/{uid}.png`` with pixel values {0, 255}.

    Parameters
    ----------
    pred_dir:
        Directory containing ecSeg output PNGs.
    output_dir:
        Where to write the harmonized binary masks.
    uid_list:
        UIDs to process.
    min_area:
        Minimum connected-component area (default 3, matching evaluation framework).
    channel:
        Channel index for ecDNA class in ecSeg output (default 3).

    Returns
    -------
    dict ``{uid: "ok" | "missing" | "error: <msg>"}``
    """
    pred_dir   = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build filename index (case-insensitive stem → path)
    file_index: Dict[str, Path] = {}
    if pred_dir.exists():
        for p in pred_dir.glob("*"):
            if p.is_file():
                file_index[p.stem.lower()] = p

    results: Dict[str, str] = {}

    for uid in uid_list:
        out_path = output_dir / f"{uid}.png"
        pred_path = _locate_file(uid, file_index,
                                 suffixes=["", "_ecseg_ecdna_mask", "_ecseg"])
        if pred_path is None:
            results[uid] = "missing"
            logger.debug("ecSeg | missing prediction for uid=%s", uid)
            continue

        try:
            binary   = load_and_binarize(pred_path, threshold=0.0, channel=channel)
            filtered = apply_min_area_filter(binary, min_area=min_area)
            write_binary_mask(filtered, out_path)
            results[uid] = "ok"
        except Exception as exc:
            results[uid] = f"error: {exc}"
            logger.warning("ecSeg | uid=%s error: %s", uid, exc)

    ok      = sum(1 for v in results.values() if v == "ok")
    missing = sum(1 for v in results.values() if v == "missing")
    errors  = sum(1 for v in results.values() if v.startswith("error"))
    logger.info("ecSeg harmonize | ok=%d missing=%d errors=%d", ok, missing, errors)
    return results


def audit_ecseg_coverage(
    pred_dir: Path,
    uid_list: List[str],
    channel:  int = ECSEG_ECDNA_CHANNEL,
) -> Dict[str, int]:
    """Count foreground pixels per UID without writing output.

    Used to verify that the ecSeg channel-3 extraction gives the expected
    object count (MAE = 0 against ecSeg's own quantification).

    Returns
    -------
    dict ``{uid: n_foreground_pixels_after_min_area_3_filter}``
    """
    pred_dir   = Path(pred_dir)
    file_index = _build_file_index(pred_dir)
    counts: Dict[str, int] = {}

    for uid in uid_list:
        pred_path = _locate_file(uid, file_index,
                                 suffixes=["", "_ecseg_ecdna_mask", "_ecseg"])
        if pred_path is None:
            counts[uid] = -1
            continue
        try:
            binary   = load_and_binarize(pred_path, threshold=0.0, channel=channel)
            filtered = apply_min_area_filter(binary, min_area=3)
            counts[uid] = int(filtered.sum())
        except Exception:
            counts[uid] = -1
    return counts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_file_index(directory: Path) -> Dict[str, Path]:
    idx: Dict[str, Path] = {}
    if not directory.exists():
        return idx
    for p in directory.glob("*"):
        if p.is_file():
            idx[p.stem.lower()] = p
    return idx


def _locate_file(
    uid:        str,
    file_index: Dict[str, Path],
    suffixes:   List[str],
) -> Optional[Path]:
    """Try multiple stem variants to locate a prediction file.

    Tries ``{uid}{suffix}`` for each suffix in *suffixes* (empty string
    first = exact UID match).  Returns ``None`` if no variant is found.
    No prefix-match fallback: a partial match would silently return the
    wrong file when UIDs share a common prefix.
    """
    uid_l = uid.lower()
    for sfx in suffixes:
        stem = (uid_l + sfx.lower()) if sfx else uid_l
        if stem in file_index:
            return file_index[stem]
    return None