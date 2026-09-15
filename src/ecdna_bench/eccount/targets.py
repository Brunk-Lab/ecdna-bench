"""
ecdna_bench.eccount.targets
============================
Soft Gaussian target map generation for ecCount.

Key design constraints (from §2 of REWRITE_PLAN.md)
----------------------------------------------------
* σ = 1.0 px is the frozen paper value.  It is defined in the FINAL
  training pixel space (1024 × 1224), so targets must be generated
  AFTER resizing the GS mask.
* Merge mode is element-wise **max** (NOT sum).  Overlapping Gaussians
  do not accumulate — the brightest wins.
* The map is normalized to [0, 1] after all centroids are drawn.
* An all-zero GS mask → all-zero target (``keep_empty_as_zeros=True``).
* Centroids are extracted with 8-connectivity (cv2.CV_32S).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import cv2
import numpy as np

__all__ = [
    "SoftTargetConfig",
    "make_centroid_gaussian_target",
    "make_target_and_points",
    "extract_component_centroids",
    "gaussian_at_points",
    "resize_target",
]

MergeMode = Literal["max", "sum"]


@dataclass
class SoftTargetConfig:
    """Configuration for centroid-Gaussian target generation.

    Attributes
    ----------
    sigma:
        Gaussian standard deviation in final training pixel space.
        **Frozen value: 1.0 px** (paper §2).
    connectivity:
        Connected-component connectivity for centroid extraction (8 or 4).
    merge_mode:
        How overlapping Gaussians are combined.  ``"max"`` is the paper
        value; ``"sum"`` is available for ablations only.
    normalize:
        If True, normalize the final map to [0, 1].
    keep_empty_as_zeros:
        If True, an all-zero GS mask returns an all-zero target instead
        of raising an error.
    """
    sigma:               float     = 1.0   # frozen paper value
    connectivity:        int       = 8
    merge_mode:          MergeMode = "max"
    normalize:           bool      = True
    keep_empty_as_zeros: bool      = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_2d(arr: np.ndarray, name: str = "mask") -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr


def _normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------

def extract_component_centroids(
    gt_mask: np.ndarray,
    connectivity: int = 8,
) -> np.ndarray:
    """Extract one centroid (x, y) per connected component.

    Parameters
    ----------
    gt_mask:
        2-D binary or grayscale mask.  Any nonzero pixel is foreground.
    connectivity:
        cv2 connectivity (8 or 4).

    Returns
    -------
    np.ndarray of shape [N, 2] in (x, y) order, dtype float32.
    Returns shape [0, 2] for an empty mask.
    """
    _ensure_2d(gt_mask, "gt_mask")
    binary = (gt_mask > 0).astype(np.uint8)

    num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=connectivity, ltype=cv2.CV_32S
    )

    if num_labels <= 1:   # 1 = background only
        return np.zeros((0, 2), dtype=np.float32)

    return centroids[1:].astype(np.float32)   # drop background


# ---------------------------------------------------------------------------
# Windowed Gaussian drawing
# ---------------------------------------------------------------------------

def _draw_gaussian(
    canvas: np.ndarray,
    cx: float,
    cy: float,
    sigma: float,
    merge_mode: MergeMode,
) -> None:
    """Draw one Gaussian centred at (cx, cy) into *canvas* in-place."""
    H, W = canvas.shape
    radius = int(3 * sigma) + 1

    x0 = max(0, int(cx) - radius)
    x1 = min(W, int(cx) + radius + 1)
    y0 = max(0, int(cy) - radius)
    y1 = min(H, int(cy) + radius + 1)

    if x0 >= x1 or y0 >= y1:
        return

    xs = np.arange(x0, x1, dtype=np.float32)
    ys = np.arange(y0, y1, dtype=np.float32)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    denom = 2.0 * sigma * sigma
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / denom)

    if merge_mode == "max":
        canvas[y0:y1, x0:x1] = np.maximum(canvas[y0:y1, x0:x1], g)
    else:
        canvas[y0:y1, x0:x1] += g


def gaussian_at_points(
    shape: Tuple[int, int],
    points_xy: np.ndarray,
    sigma: float,
    merge_mode: MergeMode = "max",
    normalize: bool = True,
) -> np.ndarray:
    """Render Gaussian blobs at the given (x, y) points.

    Parameters
    ----------
    shape:
        (H, W) of the output array.
    points_xy:
        [N, 2] float array of (x, y) coordinates.
    sigma:
        Gaussian standard deviation in pixels.
    merge_mode:
        ``"max"`` (paper value) or ``"sum"``.
    normalize:
        Normalize output to [0, 1].

    Returns
    -------
    np.ndarray of shape (H, W), dtype float32.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be > 0, got {sigma}")

    H, W = shape
    canvas = np.zeros((H, W), dtype=np.float32)

    for cx, cy in points_xy:
        _draw_gaussian(canvas, float(cx), float(cy), sigma, merge_mode)

    if normalize:
        canvas = _normalize01(canvas)

    return canvas


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def make_centroid_gaussian_target(
    gt_mask: np.ndarray,
    config: Optional[SoftTargetConfig] = None,
) -> np.ndarray:
    """Generate a soft Gaussian target map from a GS mask.

    This is the main entry point used by ``EcCountDataset``.

    Parameters
    ----------
    gt_mask:
        2-D gold-standard mask (any nonzero pixel = foreground).
    config:
        ``SoftTargetConfig``.  Uses paper defaults (σ=1.0) if None.

    Returns
    -------
    np.ndarray of shape (H, W) in [0, 1], dtype float32.
    An all-zero mask returns an all-zero map (``keep_empty_as_zeros=True``).
    """
    cfg = config or SoftTargetConfig()
    _ensure_2d(gt_mask, "gt_mask")
    binary = (gt_mask > 0).astype(np.uint8)

    if binary.max() == 0:
        if cfg.keep_empty_as_zeros:
            return np.zeros(binary.shape, dtype=np.float32)
        raise ValueError("GS mask is empty and keep_empty_as_zeros=False")

    pts = extract_component_centroids(binary, connectivity=cfg.connectivity)
    return gaussian_at_points(
        shape      = binary.shape,
        points_xy  = pts,
        sigma      = cfg.sigma,
        merge_mode = cfg.merge_mode,
        normalize  = cfg.normalize,
    )


def make_target_and_points(
    gt_mask: np.ndarray,
    config: Optional[SoftTargetConfig] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (target_map, centroids_xy).

    Convenience wrapper that returns both the soft target and the extracted
    centroids, useful for visualisation and validation scripts.
    """
    cfg = config or SoftTargetConfig()
    _ensure_2d(gt_mask, "gt_mask")
    pts = extract_component_centroids(gt_mask, connectivity=cfg.connectivity)
    target = gaussian_at_points(
        shape      = gt_mask.shape,
        points_xy  = pts,
        sigma      = cfg.sigma,
        merge_mode = cfg.merge_mode,
        normalize  = cfg.normalize,
    )
    return target, pts


def resize_target(
    target: np.ndarray,
    new_shape: Tuple[int, int],
) -> np.ndarray:
    """Resize a target map using nearest-neighbour interpolation.

    Parameters
    ----------
    target:
        2-D float32 target map.
    new_shape:
        (H, W) of the output.

    Returns
    -------
    np.ndarray of shape *new_shape*, dtype float32.
    """
    _ensure_2d(target, "target")
    H, W = new_shape
    return cv2.resize(target, (W, H), interpolation=cv2.INTER_NEAREST).astype(np.float32)
