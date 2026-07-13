"""
ecdna_bench.utils.viz — pure drawing primitives for ecDNA visualizations.

This module draws rectangles, circles, color-coded overlays, and composes
images for comparison. It has no knowledge of matching, metrics, or the
classical pipeline. Pipeline-specific overlays live in the module that
produces the data being drawn (e.g., classification overlays live in
`ecdna_bench.classical.detect`).

Object format convention
------------------------
Every function that accepts an object list expects dicts with:
    - "bbox"     : (y1, x1, y2, x2)   integer pixel coordinates
    - "centroid" : (cy, cx)           float row/column coordinates

This is the same convention used everywhere else in the package.
"""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

# Colors are (B, G, R) because OpenCV works in BGR. The defaults match the
# figure palette used in the paper: green for TP, red/orange for FP,
# blue/magenta for FN.
COLOR_TP: tuple[int, int, int] = (0, 255, 0)
COLOR_FP: tuple[int, int, int] = (0, 0, 255)
COLOR_FN: tuple[int, int, int] = (255, 0, 0)
COLOR_IGNORED: tuple[int, int, int] = (128, 128, 128)
COLOR_BBOX_DEFAULT: tuple[int, int, int] = (0, 255, 0)
COLOR_CENTROID_DEFAULT: tuple[int, int, int] = (0, 0, 255)


def to_bgr(image: np.ndarray) -> np.ndarray:
    """
    Ensure an image is 3-channel BGR suitable for OpenCV drawing.

    - (H, W)    grayscale -> (H, W, 3)
    - (H, W, 3) passed through (copied) so callers can draw without mutating
      the caller's array.
    - Anything else raises ValueError.
    """
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image.copy()
    raise ValueError(f"Unsupported image shape for visualization: {image.shape}")


def draw_bboxes(
    image: np.ndarray,
    objects: Sequence[dict],
    color: tuple[int, int, int] = COLOR_BBOX_DEFAULT,
    thickness: int = 1,
    draw_centroids: bool = True,
    centroid_color: tuple[int, int, int] = COLOR_CENTROID_DEFAULT,
    centroid_radius: int = 1,
) -> np.ndarray:
    """
    Draw bounding boxes (and optionally centroids) for a list of objects.

    Returns a new BGR image; the input is not mutated.
    """
    vis = to_bgr(image)
    for obj in objects:
        y1, x1, y2, x2 = obj["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
        if draw_centroids:
            cy, cx = obj["centroid"]
            cv2.circle(vis, (int(cx), int(cy)), centroid_radius, centroid_color, -1)
    return vis


def draw_points(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int] = COLOR_CENTROID_DEFAULT,
    radius: int = 2,
    marker: str = "circle",
) -> np.ndarray:
    """
    Draw points at (x, y) coordinates. Accepts an [N, 2] array.

    Supported markers:
      - "circle"    : filled disk
      - "crosshair" : plus sign (used for ground-truth point annotations)
    """
    vis = to_bgr(image)
    if len(points) == 0:
        return vis
    pts = np.asarray(points).astype(np.int32)
    for px, py in pts:
        if marker == "circle":
            cv2.circle(vis, (int(px), int(py)), radius, color, -1)
        elif marker == "crosshair":
            cv2.line(vis, (int(px) - radius, int(py)), (int(px) + radius, int(py)), color, 1)
            cv2.line(vis, (int(px), int(py) - radius), (int(px), int(py) + radius), color, 1)
        else:
            raise ValueError(f"Unknown marker type: {marker!r}")
    return vis


def draw_tp_fp_fn(
    image: np.ndarray,
    pred_objs: Sequence[dict],
    gt_objs: Sequence[dict],
    matched_pairs: Sequence[tuple[int, int]],
    unmatched_pred: Sequence[int],
    unmatched_gt: Sequence[int],
    ignored_pred: Sequence[int] | None = None,
    *,
    color_tp: tuple[int, int, int] = COLOR_TP,
    color_fp: tuple[int, int, int] = COLOR_FP,
    color_fn: tuple[int, int, int] = COLOR_FN,
    color_ignored: tuple[int, int, int] = COLOR_IGNORED,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw matched / unmatched-pred / unmatched-gt / ignored-pred boxes.

    Color coding matches the paper's Figure 2e:
      TP      = green predicted boxes
      FP      = red predicted boxes
      FN      = blue GT boxes
      ignored = grey predicted boxes (lost the one-to-one assignment tiebreaker)
    """
    vis = to_bgr(image)

    for p_idx, _ in matched_pairs:
        y1, x1, y2, x2 = pred_objs[p_idx]["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_tp, thickness)

    for p_idx in unmatched_pred:
        y1, x1, y2, x2 = pred_objs[p_idx]["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_fp, thickness)

    for g_idx in unmatched_gt:
        y1, x1, y2, x2 = gt_objs[g_idx]["bbox"]
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_fn, thickness)

    if ignored_pred is not None:
        for p_idx in ignored_pred:
            y1, x1, y2, x2 = pred_objs[p_idx]["bbox"]
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color_ignored, thickness)

    return vis


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    alpha: float = 0.45,
) -> np.ndarray:
    """
    Semi-transparently paint `color` onto `image` wherever `mask` is nonzero.

    Parameters
    ----------
    image
        Source image (grayscale or BGR). Returned result is always BGR.
    mask
        Binary or boolean mask with the same H, W as `image`.
    color
        BGR tuple for the overlay paint.
    alpha
        Blending weight in [0, 1] — 0 = no overlay, 1 = fully replace.
    """
    vis = to_bgr(image).astype(np.float32)
    if mask.shape[:2] != vis.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {vis.shape[:2]}"
        )
    m = mask.astype(bool)
    paint = np.zeros_like(vis)
    paint[..., 0] = color[0]
    paint[..., 1] = color[1]
    paint[..., 2] = color[2]
    vis[m] = (1 - alpha) * vis[m] + alpha * paint[m]
    return np.clip(vis, 0, 255).astype(np.uint8)


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    vertical: bool = False,
) -> np.ndarray:
    """
    Concatenate two images side-by-side (or top-bottom).

    The second image is resized to match the shared dimension of the first
    (height for horizontal, width for vertical) while preserving aspect ratio.
    """
    img1 = to_bgr(left)
    img2 = to_bgr(right)

    if not vertical:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        if h1 != h2:
            scale = h1 / h2
            img2 = cv2.resize(img2, (int(w2 * scale), h1), interpolation=cv2.INTER_AREA)
        return np.concatenate([img1, img2], axis=1)
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    if w1 != w2:
        scale = w1 / w2
        img2 = cv2.resize(img2, (w1, int(h2 * scale)), interpolation=cv2.INTER_AREA)
    return np.concatenate([img1, img2], axis=0)


def grid(
    images: Sequence[np.ndarray],
    ncols: int,
    pad: int = 2,
    pad_value: int = 0,
) -> np.ndarray:
    """
    Arrange images into a grid with `ncols` columns and uniform padding.

    Useful for building qualitative panels in supplementary figures without
    reaching for matplotlib.
    """
    if not images:
        raise ValueError("grid() requires at least one image")
    bgr_images = [to_bgr(im) for im in images]
    h = max(im.shape[0] for im in bgr_images)
    w = max(im.shape[1] for im in bgr_images)
    padded = []
    for im in bgr_images:
        dh = h - im.shape[0]
        dw = w - im.shape[1]
        padded.append(
            cv2.copyMakeBorder(im, 0, dh, 0, dw, cv2.BORDER_CONSTANT, value=(pad_value,) * 3)
        )
    n = len(padded)
    nrows = (n + ncols - 1) // ncols
    canvas = np.full(
        ((h + pad) * nrows - pad, (w + pad) * ncols - pad, 3),
        pad_value,
        dtype=np.uint8,
    )
    for i, im in enumerate(padded):
        r, c = divmod(i, ncols)
        y0 = r * (h + pad)
        x0 = c * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = im
    return canvas
