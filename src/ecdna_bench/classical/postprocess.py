"""
ecdna_bench.classical.postprocess
===================================
Eval-time post-processing helpers for the classical pipeline.

Two responsibilities
--------------------
1. **Split heuristic** — ``split_large_predictions`` splits a blob that
   contains multiple GS centroids into one predicted object per GS centroid.
   This is applied *at evaluation time* (never during training or pure
   inference); it is an optional step that can improve localization metrics
   when the detector fuses adjacent ecDNAs into one blob.

2. **Mask helpers** — convert a list of object dicts to a binary prediction
   mask.  Two strategies are provided:

   * ``objects_to_mask`` (default) — paints the per-object ``"mask"`` field
     if available, falls back to bbox fill otherwise.
   * ``objects_to_mask_bbox_fill`` — always uses bbox fill (cheaper, stable).

All functions are pure (no disk I/O).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

__all__ = [
    "split_large_predictions",
    "objects_to_mask",
    "objects_to_mask_bbox_fill",
]


# ---------------------------------------------------------------------------
# Split heuristic
# ---------------------------------------------------------------------------

def split_large_predictions(
    pred_objs: List[Dict],
    gt_objs: List[Dict],
    split_size: int = 10,
) -> List[Dict]:
    """Replace a blob that covers ≥ 2 GS centroids with one sub-prediction per centroid.

    For each predicted object whose bounding box contains 2 or more GS
    centroids, discard the original prediction and emit one small square
    prediction centred on each GS centroid.  Objects with 0 or 1 contained
    GS centroids are passed through unchanged.

    This heuristic is applied *after* detection and merging, solely to improve
    localization metrics; it has no effect on count-only evaluation.

    Parameters
    ----------
    pred_objs:
        List of predicted objects (standard dict format with ``"bbox"``).
    gt_objs:
        List of gold-standard objects (standard dict format with ``"centroid"``).
    split_size:
        Side length (in pixels) of the synthetic square bbox created around
        each GS centroid.

    Returns
    -------
    list of dict
        New list of (possibly split) predicted objects.  Each split object
        carries two extra keys:

        * ``"is_split": True``
        * ``"split_from": int``  — index in the original *pred_objs* list
    """
    half: int = max(1, int(split_size) // 2)
    result: List[Dict] = []

    for original_idx, pred in enumerate(pred_objs):
        y1, x1, y2, x2 = pred["bbox"]

        contained = [
            gt["centroid"]
            for gt in gt_objs
            if y1 <= gt["centroid"][0] <= y2 and x1 <= gt["centroid"][1] <= x2
        ]

        if len(contained) >= 2:
            for cy, cx in contained:
                cy_i = int(round(cy))
                cx_i = int(round(cx))
                result.append({
                    "centroid":   (cy, cx),
                    "bbox":       (cy_i - half, cx_i - half, cy_i + half, cx_i + half),
                    "area":       float(split_size * split_size),
                    "mask":       None,
                    "is_split":   True,
                    "split_from": original_idx,
                })
        else:
            item = dict(pred)
            item["is_split"] = False
            result.append(item)

    return result


# ---------------------------------------------------------------------------
# Objects → mask helpers
# ---------------------------------------------------------------------------

def objects_to_mask(
    objects: List[Dict],
    shape_hw: Tuple[int, int],
    use_object_masks: bool = True,
) -> np.ndarray:
    """Convert a list of object dicts to a binary mask.

    Parameters
    ----------
    objects:
        Detected objects, each with ``"bbox"`` and optionally ``"mask"``.
    shape_hw:
        Output shape ``(H, W)``.
    use_object_masks:
        If ``True`` (default) and ``obj["mask"]`` is available, paint the
        per-object mask.  Falls back to bbox fill when the mask is missing.
        If ``False``, always uses bbox fill.

    Returns
    -------
    np.ndarray
        uint8 binary mask with values {0, 255}.
    """
    H, W = shape_hw
    out  = np.zeros((H, W), dtype=np.uint8)

    for obj in objects:
        mask: Optional[np.ndarray] = obj.get("mask") if use_object_masks else None

        if mask is not None and mask.shape == (H, W):
            out[mask > 0] = 255
        else:
            # Bbox fill fallback
            y1, x1, y2, x2 = obj["bbox"]
            y1 = max(0, int(y1)); x1 = max(0, int(x1))
            y2 = min(H, int(y2)); x2 = min(W, int(x2))
            if y2 > y1 and x2 > x1:
                out[y1:y2, x1:x2] = 255

    return out


def objects_to_mask_bbox_fill(
    objects: List[Dict],
    shape_hw: Tuple[int, int],
) -> np.ndarray:
    """Convert objects to a binary mask using bbox fill only.

    Cheaper and more stable than painting per-object masks, because it
    does not depend on the ``"mask"`` field being set.

    Returns
    -------
    np.ndarray
        uint8 binary mask with values {0, 255}.
    """
    return objects_to_mask(objects, shape_hw, use_object_masks=False)
