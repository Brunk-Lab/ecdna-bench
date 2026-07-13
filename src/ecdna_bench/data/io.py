"""
ecdna_bench.data.io — low-level image and CSV I/O helpers.

Pure I/O only. No image processing, no metric computation. This module is
imported by every other workflow module; it must stay small and dependable.

Core functions
--------------
    ensure_dir          — mkdir -p
    load_image          — cv2.imread wrapper with clear error messages
    load_rgb            — 3-channel BGR
    load_gray           — single-channel grayscale
    load_mask           — binary mask loader with boolean dtype
    save_image          — cv2.imwrite with parent-dir creation
    save_mask           — save a binary/probability mask as 8-bit PNG
    load_csv / save_csv — thin pandas wrappers
    load_id_list        — read a one-column CSV of UIDs
    is_image_file       — extension check
    list_images         — enumerate images in a directory

Design notes
------------
- OpenCV is used throughout for image I/O (pandas for CSV). We keep images in
  OpenCV's native BGR ordering because downstream modules (classical
  preprocessing, ecCount dataset) are written against BGR. Conversion to RGB
  happens only at the point of display or PyTorch tensorization.
- For binary masks, `save_mask` writes an 8-bit PNG scaled so that foreground
  pixels are exactly 255 — this avoids the common footgun of PNG masks that
  store `1` and fail obvious visual inspection.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import pandas as pd  # noqa: F401 — imported lazily inside helpers
    _HAS_PANDAS = True
except ImportError:  # pragma: no cover
    _HAS_PANDAS = False


PathLike = str | Path

VALID_IMAGE_EXTS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
)


# ==============================================================================
# Small helpers
# ==============================================================================


def ensure_dir(path: PathLike) -> Path:
    """`mkdir -p path` with a pathlib return."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def is_image_file(path: PathLike) -> bool:
    """True if the path has an extension we know how to load."""
    return Path(path).suffix.lower() in VALID_IMAGE_EXTS


def list_images(folder: PathLike, *, recursive: bool = False) -> list[Path]:
    """
    Return a sorted list of image files in `folder`.

    Hidden files (starting with '.') are skipped. If `folder` is not a
    directory, an empty list is returned — callers can treat that as
    "no images found".
    """
    folder = Path(folder)
    if not folder.is_dir():
        return []

    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.iterdir()
    files = [
        p for p in iterator
        if p.is_file() and not p.name.startswith(".") and is_image_file(p)
    ]
    return sorted(files)


# ==============================================================================
# Image loading
# ==============================================================================


def _imread_or_raise(path: Path, mode: int) -> np.ndarray:
    img = cv2.imread(str(path), mode)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {path}")
    return img


def load_image(path: PathLike) -> np.ndarray:
    """
    Load an image preserving its original depth and channel count.

    Useful when you don't know in advance whether the file is single-channel,
    RGB, or RGBA (e.g., ROI masks that are sometimes stored as RGB PNGs).
    """
    return _imread_or_raise(Path(path), cv2.IMREAD_UNCHANGED)


def load_rgb(path: PathLike) -> np.ndarray:
    """Load a 3-channel BGR image. Shape (H, W, 3), dtype uint8."""
    return _imread_or_raise(Path(path), cv2.IMREAD_COLOR)


def load_gray(path: PathLike) -> np.ndarray:
    """Load a single-channel grayscale image. Shape (H, W), dtype uint8."""
    return _imread_or_raise(Path(path), cv2.IMREAD_GRAYSCALE)


def load_mask(path: PathLike, *, as_bool: bool = False, threshold: int = 0) -> np.ndarray:
    """
    Load a binary mask.

    Parameters
    ----------
    path
        Path to a mask file. PNGs with any bit depth are accepted; RGB masks
        are flattened to grayscale via the green channel (which is what
        cv2.IMREAD_GRAYSCALE does internally).
    as_bool
        If True, return a boolean array. If False (default), return a uint8
        array with values in {0, 1}.
    threshold
        Pixels strictly greater than `threshold` are foreground. Default 0,
        which means "any nonzero pixel is foreground".
    """
    img = _imread_or_raise(Path(path), cv2.IMREAD_GRAYSCALE)
    fg = img > threshold
    return fg if as_bool else fg.astype(np.uint8)


# ==============================================================================
# Image saving
# ==============================================================================


def save_image(path: PathLike, image: np.ndarray, *, create_dirs: bool = True) -> None:
    """
    Save an image with cv2.imwrite. Creates parent directories by default.
    Raises IOError if the write fails.
    """
    p = Path(path)
    if create_dirs:
        ensure_dir(p.parent)
    if not cv2.imwrite(str(p), image):
        raise OSError(f"Failed to save image to: {p}")


def save_mask(
    path: PathLike,
    mask: np.ndarray,
    *,
    scale_to_255: bool = True,
    create_dirs: bool = True,
) -> None:
    """
    Save a binary or boolean mask as an 8-bit PNG.

    Parameters
    ----------
    path
        Output path (PNG recommended; any cv2-supported extension works).
    mask
        Array of any numeric dtype. Nonzero pixels are treated as foreground.
    scale_to_255
        If True (default), foreground pixels are written as 255 so the mask
        is visually inspectable. If False, foreground is written as 1.
    """
    if mask.dtype == bool:
        arr = mask.astype(np.uint8)
    else:
        arr = (mask != 0).astype(np.uint8)
    if scale_to_255:
        arr = arr * 255
    save_image(path, arr, create_dirs=create_dirs)


def build_debug_image_path(
    output_dir: PathLike,
    uid: str,
    step: int,
    tag: str,
    ext: str = ".png",
) -> Path:
    """
    Build a debug-image path with the standardized naming scheme used
    throughout the classical pipeline and ecCount visualization scripts:

        <output_dir>/<uid>_<step>_<tag><ext>

    Spaces in `tag` are converted to underscores; path separators in `uid`
    are replaced with underscores to prevent writing outside `output_dir`.
    """
    output_dir = Path(output_dir)
    ensure_dir(output_dir)
    safe_uid = uid.replace("/", "_").replace("\\", "_")
    safe_tag = tag.replace(" ", "_")
    return output_dir / f"{safe_uid}_{step}_{safe_tag}{ext}"


def save_debug_image(
    image: np.ndarray,
    output_dir: PathLike,
    uid: str,
    step: int,
    tag: str,
    ext: str = ".png",
) -> Path:
    """Save a debug image with the standardized filename scheme."""
    p = build_debug_image_path(output_dir, uid, step, tag, ext=ext)
    save_image(p, image, create_dirs=True)
    return p


# ==============================================================================
# CSV helpers
# ==============================================================================


def _require_pandas() -> None:
    if not _HAS_PANDAS:
        raise ImportError(
            "pandas is required for CSV utilities in ecdna_bench.data.io "
            "but is not installed."
        )


def load_csv(path: PathLike, **read_csv_kwargs: Any):
    """Thin wrapper around pandas.read_csv."""
    _require_pandas()
    import pandas as pd
    return pd.read_csv(path, **read_csv_kwargs)


def save_csv(df, path: PathLike, *, index: bool = False, **to_csv_kwargs: Any) -> None:
    """Save a DataFrame to CSV, creating parent dirs as needed."""
    _require_pandas()
    p = Path(path)
    ensure_dir(p.parent)
    df.to_csv(p, index=index, **to_csv_kwargs)


def load_id_list(path: PathLike, column: str | None = None) -> list[str]:
    """
    Load a list of unique IDs from a one-column CSV.

    Parameters
    ----------
    path
        CSV path.
    column
        Column to read. If None, the first column in the file is used.
        This keeps `release/split_files/{train,val,test}_ids.csv` readable
        regardless of whether the header is "unique_id" or "image_id".
    """
    _require_pandas()
    import pandas as pd
    df = pd.read_csv(path)
    if df.empty:
        return []
    col_name = column or df.columns[0]
    if col_name not in df.columns:
        raise ValueError(f"Column {col_name!r} not found in {path}")
    return [str(x) for x in df[col_name].tolist()]


__all__ = [
    "PathLike",
    "VALID_IMAGE_EXTS",
    "ensure_dir",
    "is_image_file",
    "list_images",
    "load_image",
    "load_rgb",
    "load_gray",
    "load_mask",
    "save_image",
    "save_mask",
    "build_debug_image_path",
    "save_debug_image",
    "load_csv",
    "save_csv",
    "load_id_list",
]
