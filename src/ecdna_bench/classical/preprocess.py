"""
ecdna_bench.classical.preprocess
=================================
Atomic preprocessing operators + registry + chain runner.

Design rules
------------
* Every operator has signature ``fn(gray: np.ndarray, **kwargs) -> np.ndarray``.
  All parameters are keyword-only with defaults so callers can pass an
  arbitrary param dict without needing to know the exact signature.
* Parameters are clamped to safe ranges before use so optimizer-supplied
  values can never crash cv2.
* Registry keys are identical to the legacy ``preprocessing_map`` names so
  the frozen Stage-3 JSON files (committed at ``configs/classical/``) remain
  valid without any migration.
* This module has NO disk I/O and NO argparse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import cv2
import numpy as np

__all__ = [
    # Helpers
    "oddize",
    # Grayscale / masking
    "rgb_to_gray_preserve",
    "mask_with_roi",
    # Atomic operators
    "apply_bilateral_filter",
    "remove_background",
    "top_hat",
    "apply_gamma",
    "apply_sharpening",
    "apply_sigmoid",
    "custom_clahe",
    # Registry + chain runner
    "PREPROCESSING_REGISTRY",
    "apply_chain",
    # Legacy default chain (kept for backward-compat)
    "default_chain",
]


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def oddize(v: int) -> int:
    """Return *v* rounded to the nearest odd integer (≥ 1)."""
    v = max(1, int(round(v)))
    return v if v % 2 == 1 else v + 1


def _clamp(v, lo, hi, cast=None):
    v = max(lo, min(hi, v))
    return cast(v) if cast is not None else v


def _odd_kernel_at_most(v: int, upper: int) -> int:
    upper = max(1, int(upper))
    if upper % 2 == 0:
        upper -= 1
    v = max(1, min(int(round(v)), upper))
    return v if v % 2 == 1 else max(1, v - 1)


# ---------------------------------------------------------------------------
# Grayscale conversion
# ---------------------------------------------------------------------------

def rgb_to_gray_preserve(
    rgb: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """Convert an RGB image to grayscale without fixed channel weights.

    Using fixed-weight luminance (cv2 default) down-weights red/green probes
    that carry ecDNA signal.  ``mode="max"`` keeps whichever channel is
    brightest per pixel, which preserves probe signal best.

    Parameters
    ----------
    rgb:
        H × W (already grayscale) or H × W × 3 array.
    mode:
        ``"max"``    — per-pixel max across channels  (default, best for ecDNA)
        ``"mean"``   — mean across channels
        ``"median"`` — median across channels

    Returns
    -------
    np.ndarray
        H × W, dtype uint8 when input is uint8, else float32.
    """
    if rgb.ndim == 2:
        return rgb.astype(np.uint8) if rgb.dtype == np.uint8 else rgb.astype(np.float32)

    x = rgb[..., :3]
    if mode == "max":
        g = x.max(axis=2)
    elif mode == "mean":
        g = x.mean(axis=2)
    elif mode == "median":
        g = np.median(x, axis=2)
    else:
        raise ValueError(f"mode must be 'max', 'mean', or 'median'; got {mode!r}")

    return g.astype(np.uint8) if rgb.dtype == np.uint8 else g.astype(np.float32)


def mask_with_roi(
    image: np.ndarray,
    roi: np.ndarray,
    mode: str = "zero",
) -> np.ndarray:
    """Zero out pixels outside the ROI (DAPI or drawn ROI mask).

    Parameters
    ----------
    image:
        H × W or H × W × C image to mask.
    roi:
        H × W or H × W × C mask.  Can be binary {0, 1}, {0, 255}, or
        continuous-valued.
    mode:
        ``"zero"``  — foreground = roi > 0 (strict; background must be exactly 0)
        ``"otsu"``  — foreground = Otsu threshold on the grayscale of *roi*

    Returns
    -------
    np.ndarray
        Same shape and dtype as *image*, background pixels zeroed.
    """
    if roi.ndim == 3:
        roi_gray = rgb_to_gray_preserve(roi, mode="max")
    else:
        roi_gray = roi

    roi_u8 = roi_gray.astype(np.uint8)

    if mode == "zero":
        mask = (roi_u8 > 0).astype(np.uint8)
    elif mode == "otsu":
        _, bm = cv2.threshold(roi_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = (bm > 0).astype(np.uint8)
    else:
        raise ValueError(f"mode must be 'zero' or 'otsu'; got {mode!r}")

    out = image.copy()
    if out.ndim == 3:
        for c in range(min(3, out.shape[2])):
            out[..., c] = out[..., c] * mask
    else:
        out = out * mask
    return out


# ---------------------------------------------------------------------------
# Atomic preprocessing operators
# ---------------------------------------------------------------------------

def apply_bilateral_filter(
    gray: np.ndarray,
    d: int = 9,
    sigmaColor: float = 75.0,
    sigmaSpace: float = 75.0,
    **kwargs: Any,
) -> np.ndarray:
    """Edge-preserving denoising (bilateral filter).

    All params may be overridden through *kwargs* for optimizer compatibility.
    """
    d          = _clamp(int(round(kwargs.get("d",          d))),          1,   25, int)
    sigmaColor = _clamp(float(kwargs.get("sigmaColor", sigmaColor)),  1.0, 200.0, float)
    sigmaSpace = _clamp(float(kwargs.get("sigmaSpace", sigmaSpace)),  1.0, 200.0, float)
    return cv2.bilateralFilter(gray.astype(np.uint8), d, sigmaColor, sigmaSpace)


def remove_background(
    gray: np.ndarray,
    kernel_size: int = 11,
    **kwargs: Any,
) -> np.ndarray:
    """Subtract morphological opening to remove broad low-frequency background."""
    ks = oddize(_clamp(int(round(kwargs.get("kernel_size", kernel_size))), 3, 51, int))
    k  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
    opened = cv2.morphologyEx(gray.astype(np.uint8), cv2.MORPH_OPEN, k)
    return cv2.subtract(gray.astype(np.uint8), opened)


def top_hat(
    gray: np.ndarray,
    k_size: int = 15,
    chrom_size: int = 25,
    **kwargs: Any,
) -> np.ndarray:
    """Top-hat style enhancement that suppresses chromosome-sized structures.

    Steps
    -----
    1. Morphological close with *k_size* kernel → normalize to [0, 255].
    2. Detect chromosome-sized blobs with *chrom_size* kernel + Otsu.
    3. Multiply chromosome-blob pixels by 0.5 (dampen, not remove).

    Parameters
    ----------
    k_size:
        Kernel size for the main morphological close (ecDNA spot scale).
    chrom_size:
        Kernel size for detecting larger chromosome-like regions.
    """
    max_kernel = min(gray.shape[:2])

    k_size = _odd_kernel_at_most(
        max(3, int(round(kwargs.get("k_size", k_size)))),
        max_kernel,
    )
    chrom_size = _odd_kernel_at_most(
        max(5, int(round(kwargs.get("chrom_size", chrom_size)))),
        max_kernel,
    )

    g = gray.astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, k)
    topn   = cv2.normalize(closed, None, 0, 255, cv2.NORM_MINMAX)

    ck      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (chrom_size, chrom_size))
    c_closed = cv2.morphologyEx(g, cv2.MORPH_CLOSE, ck)
    _, chrom_mask = cv2.threshold(c_closed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    tf = topn.astype(np.float32)
    tf[chrom_mask == 255] *= 0.5

    return np.clip(tf, 0, 255).astype(np.uint8)


def apply_gamma(
    gray: np.ndarray,
    gamma_val: float = 1.2,
    **kwargs: Any,
) -> np.ndarray:
    """Gamma correction.  *gamma_val* < 1 brightens; > 1 darkens."""
    gamma = _clamp(float(kwargs.get("gamma_val", gamma_val)), 0.4, 3.0, float)
    inv   = 1.0 / max(1e-6, gamma)
    table = np.array([(i / 255.0) ** inv * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(gray.astype(np.uint8), table)


def apply_sharpening(
    gray: np.ndarray,
    kernel_weight: float = 5.0,
    **kwargs: Any,
) -> np.ndarray:
    """Unsharp-mask style sharpening."""
    w = _clamp(float(kwargs.get("kernel_weight", kernel_weight)), 0.0, 15.0, float)
    k = np.array(
        [[0, -1, 0], [-1, 4 + w, -1], [0, -1, 0]],
        dtype=np.float32,
    )
    k /= (w + 4.0)
    return cv2.filter2D(gray.astype(np.uint8), -1, k)


def apply_sigmoid(
    gray: np.ndarray,
    gain: float = 10.0,
    threshold: float = 100 / 255.0,
    **kwargs: Any,
) -> np.ndarray:
    """Sigmoid-based soft binarization.

    Parameters
    ----------
    gain:
        Steepness of the sigmoid.
    threshold:
        Sigmoid center in [0, 1] (image is normalized to [0, 1] internally).
    """
    gain = _clamp(float(kwargs.get("gain",      gain)),      1.0,  25.0, float)
    thr  = _clamp(float(kwargs.get("threshold", threshold)), 0.0, 1.0, float)

    norm = gray.astype(np.float32) / 255.0
    out  = 1.0 / (1.0 + np.exp(-gain * (norm - thr)))
    return (out * 255.0).astype(np.uint8)


def custom_clahe(
    gray: np.ndarray,
    clipLimit: float = 1.0,
    tileGridSize: int = 30,
    **kwargs: Any,
) -> np.ndarray:
    """Contrast Limited Adaptive Histogram Equalization (CLAHE)."""
    clip = _clamp(float(kwargs.get("clipLimit",   clipLimit)),   0.1, 10.0, float)
    tile = oddize(_clamp(int(round(kwargs.get("tileGridSize", tileGridSize))), 4, 64, int))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray.astype(np.uint8))


# ---------------------------------------------------------------------------
# Registry (keys match legacy preprocessing_map for JSON compatibility)
# ---------------------------------------------------------------------------

PREPROCESSING_REGISTRY: Dict[str, Any] = {
    "apply_gamma":            apply_gamma,
    "apply_sharpening":       apply_sharpening,
    "apply_bilateral_filter": apply_bilateral_filter,
    "clahe":                  custom_clahe,
    "remove_background":      remove_background,
    "top_hat":                top_hat,
    "apply_sigmoid":          apply_sigmoid,
    # Short aliases accepted in YAML configs (not in frozen JSON)
    "gamma":                  apply_gamma,
    "sharpen":                apply_sharpening,
    "bilateral":              apply_bilateral_filter,
    "tophat":                 top_hat,
    "background_suppress":    remove_background,
    "sigmoid":                apply_sigmoid,
}


def apply_chain(
    gray: np.ndarray,
    combo: List[str],
    param_map: Dict[str, Dict[str, Any]] | None = None,
) -> np.ndarray:
    """Apply a list of named preprocessing ops in order.

    Parameters
    ----------
    gray:
        Input grayscale image (uint8 strongly preferred).
    combo:
        Ordered list of registry keys, e.g. ``["top_hat", "clahe", "apply_sigmoid"]``.
    param_map:
        Optional mapping of ``{op_name: {param: value}}``.  Missing entries
        use each operator's built-in defaults.

    Returns
    -------
    np.ndarray
        Enhanced grayscale image (uint8).
    """
    if param_map is None:
        param_map = {}

    x = gray.astype(np.uint8)
    for name in combo:
        if name not in PREPROCESSING_REGISTRY:
            raise KeyError(
                f"Preprocessing op {name!r} not in PREPROCESSING_REGISTRY. "
                f"Available: {sorted(PREPROCESSING_REGISTRY)}"
            )
        fn = PREPROCESSING_REGISTRY[name]
        x  = fn(x, **param_map.get(name, {}))
        # Ensure uint8 between steps (some ops return float32)
        if x.dtype != np.uint8:
            x = np.clip(x, 0, 255).astype(np.uint8)
    return x


def _unpack_flat_params(flat: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Convert ``{"clahe__clipLimit": 0.4, ...}`` → ``{"clahe": {"clipLimit": 0.4}}``.

    This is the format produced by the Stage-3 Bayesian optimiser.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for k, v in flat.items():
        if "__" not in k:
            continue
        fn, param = k.split("__", 1)
        out.setdefault(fn, {})[param] = v
    return out


# ---------------------------------------------------------------------------
# Legacy default chain (reproduce original count-only pipeline)
# ---------------------------------------------------------------------------

def default_chain(
    rgb_or_gray: np.ndarray,
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply the original fixed 4-step chain: top_hat → sharpen → CLAHE → sigmoid.

    This reproduces the pre-optimization classical pipeline behavior.
    Kept for backward compatibility and cross-check tests.

    Parameters
    ----------
    rgb_or_gray:
        Input image.  RGB is converted to grayscale first.
    params:
        Flat parameter dict with keys:
        ``kernel_size``, ``chrom_kernel_size``, ``strength``,
        ``clip_limit``, ``tile_grid_size``, ``cutoff``, ``gain``.

    Returns
    -------
    simple_gray : np.ndarray
        Grayscale before enhancement (for debug/overlay).
    enhanced : np.ndarray
        After full preprocessing chain.
    """
    required = [
        "kernel_size",
        "chrom_kernel_size",
        "strength",
        "clip_limit",
        "tile_grid_size",
        "cutoff",
        "gain",
    ]
    missing = [k for k in required if k not in params]
    if missing:
        raise KeyError(f"default_chain params missing required keys: {missing}")

    if rgb_or_gray.ndim == 3:
        simple_gray = rgb_to_gray_preserve(rgb_or_gray, mode="max")
    else:
        simple_gray = rgb_or_gray.astype(np.uint8)

    th    = top_hat(simple_gray,
                    k_size=params["kernel_size"],
                    chrom_size=params["chrom_kernel_size"])
    sharp = apply_sharpening(th, kernel_weight=params["strength"])
    clahe_img = custom_clahe(sharp,
                             clipLimit=params["clip_limit"],
                             tileGridSize=params["tile_grid_size"])
    enhanced = apply_sigmoid(clahe_img, gain=params["gain"], threshold=params["cutoff"] / 255.0)

    return simple_gray, enhanced
