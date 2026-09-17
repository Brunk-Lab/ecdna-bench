"""
Image loading for the ROI model.

Same pixel semantics as River's code: TIFF files are read with tifffile (RGB
channel order as stored; channel-first arrays moved to channel-last; 2-D DAPI
kept as is), other files with OpenCV (RGB converted from BGR, DAPI read as
grayscale). The format is decided from the file's first bytes, not from its
extension, because some lab copies hold PNG bytes under .tif names.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

import cv2
import numpy as np

IMAGE_EXTENSIONS = (".tif", ".tiff", ".png")
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")


def is_tiff(path) -> bool:
    with open(path, "rb") as fh:
        return fh.read(4) in _TIFF_MAGIC


def _tif_image(path):
    import tifffile
    image = tifffile.imread(str(path))
    if image.ndim == 3 and image.shape[0] in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[2] == 1:
        image = image.squeeze(-1)
    return image


def _tif_mask(path):
    import tifffile
    mask = tifffile.imread(str(path))
    if mask.ndim == 3:
        mask = mask[..., 0] if mask.shape[0] > mask.shape[2] else mask[0]
    return mask


def _check_uint8(arr, path, what):
    if arr is None:
        raise RuntimeError(f"failed to read {what} image: {path}")
    if arr.dtype != np.uint8:
        raise ValueError(f"{what} image must be 8-bit, got {arr.dtype}: {path}")
    return arr


def read_rgb(path) -> np.ndarray:
    """H x W x 3 uint8, RGB order."""
    if is_tiff(path):
        img = _tif_image(path)
    else:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _check_uint8(img, path, "RGB")
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(f"RGB image must have 3 channels, got shape {img.shape}: {path}")
    return img


def read_dapi(path) -> np.ndarray:
    """H x W uint8."""
    img = _tif_mask(path) if is_tiff(path) else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    img = _check_uint8(img, path, "DAPI")
    if img.ndim != 2:
        raise ValueError(f"DAPI image must be single-channel, got shape {img.shape}: {path}")
    return img


def read_mask(path) -> np.ndarray:
    """H x W float32 in {0, 1} (any value > 0 is foreground)."""
    m = _tif_mask(path) if is_tiff(path) else cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise RuntimeError(f"failed to read mask: {path}")
    return (m > 0).astype(np.float32)


def stack_channels(rgb: Optional[np.ndarray], dapi: Optional[np.ndarray]) -> np.ndarray:
    """RGB (+ DAPI as a fourth channel), as the model was trained."""
    if rgb is None and dapi is None:
        raise ValueError("need RGB and/or DAPI")
    if dapi is None:
        return rgb
    dapi = np.expand_dims(dapi, axis=-1)
    if rgb is None:
        return dapi
    if rgb.shape[:2] != dapi.shape[:2]:
        raise ValueError(f"RGB {rgb.shape[:2]} and DAPI {dapi.shape[:2]} sizes differ")
    return np.concatenate((rgb, dapi), axis=-1)


def find_image(folder, uid: str) -> Optional[Path]:
    """<folder>/<uid>.<tif|tiff|png>, or None."""
    if folder is None:
        return None
    for ext in IMAGE_EXTENSIONS:
        p = Path(folder) / f"{uid}{ext}"
        if p.is_file():
            return p
    return None


def read_ids(csv_path) -> List[str]:
    """First column of a CSV with a header row (like train_ids.csv), sorted, unique."""
    import csv
    ids = set()
    with open(csv_path, newline="") as fh:
        reader = csv.reader(fh)
        next(reader, None)
        for row in reader:
            if row and row[0].strip():
                ids.add(row[0].strip())
    return sorted(ids)


def ids_in_folder(folder) -> List[str]:
    return sorted({p.stem for p in Path(folder).iterdir()
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS})


def load_pair(rgb_dir, dapi_dir, uid: str) -> np.ndarray:
    rgb = dapi = None
    if rgb_dir is not None:
        p = find_image(rgb_dir, uid)
        if p is None:
            raise FileNotFoundError(f"no RGB image for {uid} in {rgb_dir}")
        rgb = read_rgb(p)
    if dapi_dir is not None:
        p = find_image(dapi_dir, uid)
        if p is None:
            raise FileNotFoundError(f"no DAPI image for {uid} in {dapi_dir}")
        dapi = read_dapi(p)
    return stack_channels(rgb, dapi)
