"""
ecdna_bench.evaluation.objects — connected-component extraction and the
standard object schema used throughout the evaluation framework.

Standard object dict
--------------------
Every downstream module consumes lists of dicts in the following shape::

    {
        "centroid": (cy, cx),          # float (row, col), sub-pixel OK
        "bbox": (y1, x1, y2, x2),      # int; min_row, min_col, max_row, max_col
                                       # (max_row, max_col) are exclusive,
                                       # matching scikit-image's regionprops.
        "area": int,                   # pixel count, >= min_area
        "mask": np.ndarray | None,     # uint8 binary mask, same shape as the
                                       # full image. May be None when the
                                       # source did not carry a mask (e.g.
                                       # centroid-only gold standard).
    }

Any extra keys ("gt_pos_count", "score", "is_split", …) are allowed and are
passed through unchanged. The matcher in
:mod:`ecdna_bench.evaluation.matching` reads only the four keys above; any
additional metadata is caller-defined.

Why a dict, not a dataclass?
----------------------------
Historically these object lists are produced by many different code paths
(classical pipeline, GS decoder, external-baseline adapters, ecCount
post-processor) and consumed by one matcher. A plain ``dict`` is the most
portable container for that many-producer / one-consumer pattern, and it
imposes no import ordering between the producers. The fields are documented
here and enforced only by convention; the matcher validates that the four
required keys exist before use.

Connectivity
------------
All CC operations in this module use 8-connectivity (``connectivity=2`` in
scikit-image terms). This matches the paper's Supplementary §13 specification
and the ``evaluation.matching.connectivity = 8`` config field.

Minimum area
------------
The default ``min_area = 3`` is the paper's documented cutoff for both the
prediction side and the gold-standard side (Supplementary §13). Components
smaller than 3 pixels are almost always salt-noise or single-pixel artifacts
rather than real ecDNA, and keeping them would inflate both the false-positive
count (on the prediction side) and the false-negative count (on the GS side).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
from skimage.measure import label as sk_label
from skimage.measure import regionprops


__all__ = [
    "objects_from_mask",
    "bbox_iou",
    "centroid_distance",
]


# ==============================================================================
# Binarization helper — used by every mask-to-objects entry point.
# ==============================================================================


def _binarize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Return a boolean foreground mask from any of the formats we encounter:

    - 2D uint8 with values in {0, 1} (already-binary GS)
    - 2D uint8 with values in {0, 255} (PNG-encoded binary)
    - 2D float in [0, 1] (probability map)
    - 2D integer instance-labeled mask (any positive label is foreground)
    - 3D RGB (converted to grayscale first)

    We deliberately do not threshold probability maps at 0.5 here: converting
    a probability map to objects is a separate concern handled by the ecCount
    post-processor (see ``eccount.postprocess``), not by the evaluation-side
    object extractor. A probability map arriving here should already have been
    thresholded by its caller; we treat any strictly positive value as
    foreground, which is the right behavior for already-binarized inputs and
    degenerates to the ``(prob > 0)`` test for probability maps — which is
    almost never what you want but at least fails loudly rather than silently
    halving the recall.
    """
    if mask.ndim == 3:
        # Use the standard BGR→gray conversion; the result is 0 iff all three
        # channels were 0, which is the only regime we care about here.
        mask = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    if mask.dtype == bool:
        return mask

    # Treat any strictly positive integer or float value as foreground.
    return mask > 0


# ==============================================================================
# Main entry point — mask → list of object dicts.
# ==============================================================================


def objects_from_mask(
    mask: np.ndarray,
    *,
    min_area: int = 3,
    connectivity: int = 8,
    attach_mask: bool = True,
    extra_keys: bool = False,
) -> list[dict[str, Any]]:
    """
    Extract connected components from a binary (or binarizable) mask and
    return them as a list of standard object dicts.

    Parameters
    ----------
    mask : np.ndarray
        Input mask. See :func:`_binarize_mask` for accepted formats.
    min_area : int, default 3
        Minimum area, in pixels, for a component to be kept. Components
        smaller than this are discarded silently.
    connectivity : {4, 8}, default 8
        Pixel connectivity used to label components. 8 is the paper
        convention.
    attach_mask : bool, default True
        If True, each returned dict carries a per-component binary ``mask``
        of the same shape as the input. If False, the ``mask`` key is
        ``None``. Turning this off is a useful memory optimization when a
        caller only needs centroids/bboxes/areas (e.g., the centroid-only
        GS code path), at the cost of losing IoU-at-mask capability in the
        matcher.
    extra_keys : bool, default False
        If True, additionally attach ``gt_pos_count`` (the per-component
        positive-pixel count). This is the original caching key from
        ``detection.extract_gt_objects_from_mask`` and is preserved for
        backward compatibility with code that may read it, but it is not
        required by any current matcher.

    Returns
    -------
    objects : list of dict
        One dict per surviving component, in scikit-image's natural label
        order (which is deterministic for a given input). The order carries
        no biological meaning but is reproducible, which matters for
        reproducibility of the Hungarian assignment when costs tie.
    """
    if mask.size == 0:
        return []

    if connectivity not in (4, 8):
        raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")

    # scikit-image uses connectivity=1 for 4-conn and connectivity=2 for 8-conn.
    sk_conn = 2 if connectivity == 8 else 1

    binary = _binarize_mask(mask).astype(np.uint8)
    labeled = sk_label(binary, connectivity=sk_conn)

    objects: list[dict[str, Any]] = []
    for prop in regionprops(labeled):
        if prop.area < min_area:
            continue

        y1, x1, y2, x2 = prop.bbox  # (min_row, min_col, max_row, max_col)
        obj: dict[str, Any] = {
            "centroid": (float(prop.centroid[0]), float(prop.centroid[1])),
            "bbox": (int(y1), int(x1), int(y2), int(x2)),
            "area": int(prop.area),
            "mask": None,
        }

        if attach_mask:
            # Build the per-component mask as a view into the labeled image.
            # This is slightly more memory than strictly necessary (full-frame
            # mask per component) but matches what every downstream consumer
            # expects and lets the matcher compute true-IoU directly.
            obj["mask"] = (labeled == prop.label).astype(np.uint8)

        if extra_keys:
            # gt_pos_count is tautologically equal to area for binary inputs,
            # which is the only case we currently handle, but we expose it
            # under its historical name so legacy consumers do not break.
            obj["gt_pos_count"] = int(prop.area)

        objects.append(obj)

    return objects


# ==============================================================================
# Pairwise geometry helpers.
# ==============================================================================


def bbox_iou(
    bb1: tuple[int, int, int, int],
    bb2: tuple[int, int, int, int],
) -> float:
    """
    Intersection-over-union between two axis-aligned bounding boxes in
    ``(y1, x1, y2, x2)`` format with ``(y2, x2)`` exclusive.

    Returns 0.0 when the boxes do not overlap or have degenerate area.
    """
    y1a, x1a, y2a, x2a = bb1
    y1b, x1b, y2b, x2b = bb2

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h

    area1 = max(0, x2a - x1a) * max(0, y2a - y1a)
    area2 = max(0, x2b - x1b) * max(0, y2b - y1b)
    union_area = area1 + area2 - inter_area

    return inter_area / union_area if union_area > 0 else 0.0


def centroid_distance(
    c1: tuple[float, float],
    c2: tuple[float, float],
) -> float:
    """
    Euclidean distance between two (row, col) centroids.

    Uses ``math.hypot`` instead of ``np.hypot`` because the matcher calls
    this function in a tight double-``for`` loop over predictions × ground
    truths; the numpy overhead is non-negligible at that scale.
    """
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])
