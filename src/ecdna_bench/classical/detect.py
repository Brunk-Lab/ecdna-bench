"""
ecdna_bench.classical.detect
=============================
Object detection (Otsu → CC → area filter), centroid-based merging, and
HSV-based ecDNA/chromosome classifier.

The HSV classifier was previously mixed into ``visualization.py``; it lives
here because it is a detection-stage concern (labelling detected blobs), not
a drawing concern.

All functions are pure: they operate on NumPy arrays and return plain Python
objects.  No disk I/O.
"""

from __future__ import annotations

import math
from typing import Dict, List, Literal, Tuple

import cv2
import numpy as np
from skimage.measure import label, regionprops

__all__ = [
    "detect_objects",
    "merge_close_objects",
    "classify_hsv",
    "label_objects_hsv",
    "oddize",  # re-exported for classical_opt convenience
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def oddize(v: int) -> int:
    """Return *v* rounded to the nearest odd integer (≥ 1)."""
    v = max(1, int(round(v)))
    return v if v % 2 == 1 else v + 1


# ---------------------------------------------------------------------------
# Core detection
# ---------------------------------------------------------------------------

def detect_objects(
    gray: np.ndarray,
    threshold_factor: float = 1.0,
    morph_close_kernel: int = 5,
    min_area: int = 5,
    max_area: int = 900,
) -> List[Dict]:
    """Detect objects from a preprocessed grayscale image.

    Pipeline
    --------
    1. Otsu threshold × *threshold_factor*.
    2. Morphological closing with an elliptical kernel of size *morph_close_kernel*.
    3. 8-connected component labelling (skimage).
    4. Area filter [*min_area*, *max_area*].

    Parameters
    ----------
    gray:
        Preprocessed 2-D grayscale image (uint8 recommended).
    threshold_factor:
        Multiplier on the Otsu threshold.  Values > 1 make thresholding stricter.
    morph_close_kernel:
        Diameter in pixels of the elliptical closing kernel.
    min_area, max_area:
        Inclusive area bounds in pixels.

    Returns
    -------
    list of dict, each with keys:

    * ``"centroid"``  — ``(cy, cx)``  float row/col
    * ``"bbox"``      — ``(y1, x1, y2, x2)`` ints
    * ``"area"``      — int, pixel area
    * ``"mask"``      — H×W uint8 binary mask for this object
    """
    if gray.ndim != 2:
        raise ValueError("detect_objects expects a 2-D grayscale image")

    gray_u8 = gray.astype(np.uint8)

    # Otsu threshold
    otsu_val, _ = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = int(min(255, max(0, otsu_val * float(threshold_factor))))
    _, binary = cv2.threshold(gray_u8, thresh, 255, cv2.THRESH_BINARY)

    # Morphological close
    ksize  = oddize(morph_close_kernel)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Connected components (8-connected)
    labeled  = label(closed, connectivity=2)
    objects: List[Dict] = []
    for p in regionprops(labeled):
        if min_area <= p.area <= max_area:
            y1, x1, y2, x2 = p.bbox
            objects.append({
                "centroid": p.centroid,                           # (cy, cx)
                "bbox":     (int(y1), int(x1), int(y2), int(x2)),
                "area":     int(p.area),
                "mask":     (labeled == p.label).astype(np.uint8),
            })

    return objects


# ---------------------------------------------------------------------------
# Merging close objects
# ---------------------------------------------------------------------------

def merge_close_objects(
    objects: List[Dict],
    merge_distance: float,
) -> List[Dict]:
    """Merge objects whose centroids are closer than *merge_distance* pixels.

    Objects are merged greedily in index order.  When object *i* absorbs
    object *j*:

    * bbox is expanded to the union bounding box,
    * centroid is the unweighted mean of absorbed centroids,
    * area is the sum,
    * mask is the logical OR (or ``None`` if any side was missing its mask).

    Parameters
    ----------
    objects:
        Detected objects (standard dict format).
    merge_distance:
        Maximum centroid-to-centroid distance (px) to trigger a merge.

    Returns
    -------
    list of dict
        Merged objects in the same standard format.
    """
    if not objects:
        return []

    merge_distance = float(merge_distance)
    taken          = [False] * len(objects)
    merged: List[Dict] = []

    for i in range(len(objects)):
        if taken[i]:
            continue

        cur       = objects[i].copy()
        cy, cx    = cur["centroid"]
        y1, x1, y2, x2 = cur["bbox"]
        mmask = (cur["mask"].astype(np.uint8).copy()
                 if cur.get("mask") is not None else None)

        for j in range(i + 1, len(objects)):
            if taken[j]:
                continue
            cy2, cx2 = objects[j]["centroid"]
            if math.dist((cy, cx), (cy2, cx2)) < merge_distance:
                y1b, x1b, y2b, x2b = objects[j]["bbox"]

                # Expand bbox
                y1  = min(y1, y1b); x1 = min(x1, x1b)
                y2  = max(y2, y2b); x2 = max(x2, x2b)
                cur["bbox"] = (y1, x1, y2, x2)

                # Average centroid (simple; not area-weighted)
                cy = (cy + cy2) / 2.0
                cx = (cx + cx2) / 2.0
                cur["centroid"] = (cy, cx)

                # Accumulate area
                cur["area"] = float(cur["area"]) + float(objects[j]["area"])

                # Merge masks
                m2 = objects[j].get("mask")
                if mmask is not None and m2 is not None:
                    mmask = np.logical_or(mmask > 0, m2 > 0).astype(np.uint8)
                else:
                    mmask = None

                taken[j] = True

        cur["mask"] = mmask
        merged.append(cur)
        taken[i] = True

    return merged


# ---------------------------------------------------------------------------
# HSV-based ecDNA / chromosome classifier
# ---------------------------------------------------------------------------

def classify_hsv(
    roi_bgr: np.ndarray,
    white_value_threshold: float = 170.0,
    white_saturation_threshold: float = 50.0,
) -> Literal["chromosome", "ecDNA"]:
    """Classify one object ROI as 'chromosome' or 'ecDNA' using HSV mean.

    Rule: high *V* and low *S* → chromosome-like (whitish); otherwise → ecDNA.

    Parameters
    ----------
    roi_bgr:
        Cropped BGR image of the object's bounding box.
    white_value_threshold:
        Minimum mean V value to consider as "white / chromosome".
    white_saturation_threshold:
        Maximum mean S value to consider as "white / chromosome".

    Returns
    -------
    ``"chromosome"`` or ``"ecDNA"``
    """
    if roi_bgr.size == 0:
        return "ecDNA"

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    _, s_mean, v_mean, _ = cv2.mean(hsv)

    if v_mean > white_value_threshold and s_mean < white_saturation_threshold:
        return "chromosome"
    return "ecDNA"


def label_objects_hsv(
    objects: List[Dict],
    rgb_image: np.ndarray,
    white_value_threshold: float = 170.0,
    white_saturation_threshold: float = 50.0,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Label every object as 'ecDNA' or 'chromosome' using the HSV classifier.

    Each object dict in the returned list gets a new ``"label"`` key
    (``"ecDNA"`` or ``"chromosome"``).

    Parameters
    ----------
    objects:
        Standard object dicts with ``"bbox"`` key.
    rgb_image:
        Full-frame BGR or RGB image from which ROIs are extracted.
    white_value_threshold, white_saturation_threshold:
        HSV classification thresholds.

    Returns
    -------
    labelled_objects : list of dict
        Same objects with an additional ``"label"`` key.
    counts : dict
        ``{"ecDNA": int, "chromosome": int}``
    """
    # Work in BGR (OpenCV convention) — accept both BGR and RGB
    if rgb_image.ndim != 3:
        raise ValueError("rgb_image must be a 3-channel image")

    bgr = rgb_image  # caller is responsible for channel order; cv2 HSV works on BGR

    labelled: List[Dict] = []
    counts: Dict[str, int] = {"ecDNA": 0, "chromosome": 0}

    for obj in objects:
        y1, x1, y2, x2 = obj["bbox"]
        roi = bgr[int(y1):int(y2), int(x1):int(x2)]

        lbl = classify_hsv(
            roi,
            white_value_threshold=white_value_threshold,
            white_saturation_threshold=white_saturation_threshold,
        )
        new_obj = dict(obj)
        new_obj["label"] = lbl
        labelled.append(new_obj)
        counts[lbl] += 1

    return labelled, counts
