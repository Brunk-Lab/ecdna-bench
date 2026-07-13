"""
ecdna_bench.eccount.postprocess
=================================
Post-processing: predicted probability map → ecDNA peak points.

Exact pipeline order (from §2 of REWRITE_PLAN.md)
--------------------------------------------------
1. Gaussian smoothing  (smooth_sigma = 0.5)
2. Re-apply ROI mask   (after smoothing, to avoid boundary bleed)
3. Local maxima detection
4. Absolute threshold  (threshold_abs = 0.35)
5. Greedy Euclidean NMS (nms_min_distance = 2)

Two output helpers
------------------
* ``peaks_to_mask`` — diamonds of radius *disk_radius* at each retained peak
* ``prob_to_threshold_mask`` — simple sigmoid-probability binarization

All functions are pure (no disk I/O, no PyTorch).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

__all__ = [
    "PostprocessConfig",
    "detect_points_from_map",
    "peaks_to_mask",
    "prob_to_threshold_mask",
    "find_local_maxima",
    "greedy_distance_suppression",
]


@dataclass
class PostprocessConfig:
    """Configuration for ecCount post-processing.

    All defaults are the frozen paper values (YAML: eccount.postprocess.*).
    Field names intentionally match the YAML keys so they can be
    unpacked directly from the config dict.
    """
    smooth_sigma:       float           = 0.5   # frozen
    reapply_roi:        bool            = True  # frozen – apply ROI AFTER smoothing
    threshold_abs:      float           = 0.35  # frozen
    peak_min_distance:  int             = 2     # frozen
    nms_min_distance:   int             = 2     # frozen
    exclude_border:     int             = 0     # frozen
    point_disk_radius:  int             = 2     # frozen – diamond radius; r=2 → 5×5 diamond
    max_points:         Optional[int]   = None  # None = no cap; useful for ablations


def _validate_config(cfg: PostprocessConfig) -> None:
    if cfg.smooth_sigma < 0:
        raise ValueError(f"smooth_sigma must be >= 0, got {cfg.smooth_sigma}")
    if cfg.threshold_abs < 0:
        raise ValueError(f"threshold_abs must be >= 0, got {cfg.threshold_abs}")
    if cfg.peak_min_distance < 1:
        raise ValueError(f"peak_min_distance must be >= 1, got {cfg.peak_min_distance}")
    if cfg.nms_min_distance < 0:
        raise ValueError(f"nms_min_distance must be >= 0, got {cfg.nms_min_distance}")
    if cfg.exclude_border < 0:
        raise ValueError(f"exclude_border must be >= 0, got {cfg.exclude_border}")
    if cfg.point_disk_radius < 0:
        raise ValueError(f"point_disk_radius must be >= 0, got {cfg.point_disk_radius}")


def _ensure_2d(arr: np.ndarray, name: str) -> np.ndarray:
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D, got shape {arr.shape}")
    return arr


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------

def find_local_maxima(
    pred_map:          np.ndarray,
    threshold_abs:     float,
    peak_min_distance: int,
    exclude_border:    int = 0,
) -> List[Tuple[int, int, float]]:
    """Detect candidate local maxima above *threshold_abs*.

    Uses ``scipy.ndimage.maximum_filter`` to avoid the skimage dependency.

    Returns a list of ``(x, y, score)`` sorted by descending score,
    where x = column, y = row.
    """
    pred = _ensure_2d(pred_map, "pred_map").astype(np.float32)
    size     = 2 * int(peak_min_distance) + 1
    max_map  = maximum_filter(pred, size=size, mode="constant")
    local_max = (pred == max_map) & (pred >= float(threshold_abs))

    if exclude_border > 0:
        b = int(exclude_border)
        local_max[:b, :]  = False
        local_max[-b:, :] = False
        local_max[:, :b]  = False
        local_max[:, -b:] = False

    ys, xs = np.where(local_max)
    scores  = pred[ys, xs]
    peaks   = [(int(x), int(y), float(s)) for x, y, s in zip(xs, ys, scores)]
    peaks.sort(key=lambda t: t[2], reverse=True)
    return peaks


def greedy_distance_suppression(
    peaks:        List[Tuple[int, int, float]],
    min_distance: int,
    max_points:   Optional[int] = None,
) -> List[Tuple[int, int, float]]:
    """Greedy Euclidean NMS on peaks sorted by descending score.

    Processes peaks in score-descending order; suppresses any peak whose
    Euclidean distance to an already-kept peak is < *min_distance*.
    """
    if min_distance <= 0:
        return list(peaks[:max_points]) if max_points else list(peaks)

    kept: List[Tuple[int, int, float]] = []
    for px, py, ps in peaks:
        ok = all(float(np.hypot(px - kx, py - ky)) >= min_distance
                 for kx, ky, _ in kept)
        if ok:
            kept.append((px, py, ps))
            if max_points and len(kept) >= max_points:
                break
    return kept


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def detect_points_from_map(
    pred_map: np.ndarray,
    roi_mask: Optional[np.ndarray] = None,
    config:   Optional[PostprocessConfig] = None,
) -> List[Tuple[int, int, float]]:
    """Full post-processing: prob map → list of (x, y, score) peaks.

    Pipeline (exact order per ARCHITECTURE.md design rule #4 and §2 of
    REWRITE_PLAN.md):
        1. Gaussian smooth
        2. Re-apply ROI mask *after* smoothing (avoids boundary bleed)
        3. Local maxima detection + absolute threshold
        4. Greedy Euclidean NMS

    Parameters
    ----------
    pred_map:
        2-D float32 sigmoid probability map in [0, 1].
    roi_mask:
        Optional 2-D binary mask (uint8 or bool, 1=inside ROI).
        Applied after smoothing when ``config.reapply_roi`` is True.
    config:
        ``PostprocessConfig``.  Uses frozen paper defaults if None.

    Returns
    -------
    List of ``(x, y, score)`` tuples sorted by score descending.
    """
    cfg = config or PostprocessConfig()
    _validate_config(cfg)

    pred = _ensure_2d(pred_map, "pred_map").astype(np.float32)

    # Step 1: Gaussian smooth
    if cfg.smooth_sigma > 0:
        pred = gaussian_filter(pred, sigma=float(cfg.smooth_sigma))

    # Step 2: Re-apply ROI mask AFTER smoothing
    if cfg.reapply_roi and roi_mask is not None:
        roi = _ensure_2d(roi_mask, "roi_mask")
        if pred.shape != roi.shape:
            raise ValueError(
                f"pred_map {pred.shape} vs roi_mask {roi.shape} shape mismatch"
            )
        pred = pred.copy()
        pred[roi <= 0] = 0.0

    # Step 3: Local maxima + absolute threshold
    peaks = find_local_maxima(
        pred, cfg.threshold_abs, cfg.peak_min_distance, cfg.exclude_border
    )

    # Step 4: Greedy NMS
    peaks = greedy_distance_suppression(peaks, cfg.nms_min_distance, cfg.max_points)

    return peaks


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def peaks_to_mask(
    peaks:       List[Tuple[int, int, float]],
    shape:       Tuple[int, int],
    disk_radius: int = 2,
) -> np.ndarray:
    """Render peaks as filled diamonds on a binary mask.

    Uses an L1-norm (Manhattan distance) criterion so the rendered shape
    matches the 7×7 diamond used for GT annotation (point_disk_radius=3
    in GT; use point_disk_radius=2 here for a 5×5 diamond).

    A diamond of radius r has tip-to-tip span (2r+1) in both axes:
        r=1 → 3×3 diamond  (13 pixels)  — very small
        r=2 → 5×5 diamond  (13 pixels)  — matches GT 7×7 style at half size
        r=3 → 7×7 diamond  (25 pixels)  — identical to GT diamond shape

    Parameters
    ----------
    peaks:
        List of ``(x, y, score)`` from ``detect_points_from_map``.
    shape:
        Output (H, W).
    disk_radius:
        L1 radius of each diamond in pixels.  0 = single pixel per peak.

    Returns
    -------
    np.ndarray, uint8, values {0, 255}.
    """
    H, W = shape
    out  = np.zeros((H, W), dtype=np.uint8)

    if disk_radius <= 0:
        for x, y, _ in peaks:
            if 0 <= y < H and 0 <= x < W:
                out[y, x] = 255
        return out

    # Vectorised diamond drawing via meshgrid + L1 norm
    ys_grid = np.arange(H)
    xs_grid = np.arange(W)
    xx, yy  = np.meshgrid(xs_grid, ys_grid)

    for x, y, _ in peaks:
        mask = np.abs(xx - x) + np.abs(yy - y) <= disk_radius
        out[mask] = 255

    return out


def prob_to_threshold_mask(
    prob_map:  np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Binarize a probability map at *threshold*.

    Parameters
    ----------
    prob_map:
        2-D float array of sigmoid probabilities in [0, 1].
    threshold:
        Pixels at or above this value become foreground (255).

    Returns
    -------
    np.ndarray, uint8, values {0, 255}.
    """
    _ensure_2d(prob_map, "prob_map")
    return (prob_map >= threshold).astype(np.uint8) * 255