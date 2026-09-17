"""
ROI inference.

Reproduces River's ``infer.py`` (albumentations 2.0.8) without albumentations:

1. resize the uint8 image to 1,024 x 1,224 px (bilinear, on the uint8 array);
2. scale every channel, DAPI included, to [0, 1] and standardize with mean 0.5
   and standard deviation 0.5 (in albumentations 2.0.8 the three-value mean and
   standard deviation passed by River's code were applied to all four channels;
   the lookup table below is bit-identical to that library path);
3. forward pass, sigmoid;
4. resize the probability map to 2,048 x 2,448 px (nearest neighbor);
5. Gaussian smoothing (sigma = 10 px), threshold 0.4, fill holes, keep the
   connected component under the image center (all components if the center is
   background); write 0/255 PNG named ``<unique_id>.png``.

River's ``morph.py`` also accepts ``min_size=200``, but the small-object removal
it would control is commented out; it is omitted here.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

from . import io as roi_io

logger = logging.getLogger(__name__)

TRAIN_SIZE = (1024, 1224)   # (height, width) used for training and inference
SAVE_SIZE = (2048, 2448)    # native acquisition size
THRESHOLD = 0.4
SMOOTH_SIGMA = 10.0

# albumentations 2.0.8 Normalize(mean=0.5, std=0.5, max_pixel_value=255) on uint8
_LUT = (np.arange(0, 256, dtype=np.float32) - 127.5) * (1 / 127.5)


def preprocess(image: np.ndarray, size: Sequence[int] = TRAIN_SIZE) -> np.ndarray:
    """uint8 H x W x C -> float32 C x h x w in [-1, 1]."""
    if image.dtype != np.uint8:
        raise ValueError(f"expected uint8, got {image.dtype}")
    h, w = int(size[0]), int(size[1])
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    if resized.ndim == 2:
        resized = resized[:, :, None]
    normed = cv2.LUT(resized, _LUT)
    if normed.ndim == 2:
        normed = normed[:, :, None]
    return np.ascontiguousarray(np.transpose(normed, (2, 0, 1)))


def postprocess_mask(probs: np.ndarray, threshold: float = THRESHOLD,
                     keep_center: bool = True) -> np.ndarray:
    """Probability map at native size -> uint8 mask with values 0 and 255."""
    probs = gaussian_filter(probs, sigma=SMOOTH_SIGMA)
    mask = (probs > threshold).astype(np.uint8)
    mask = binary_fill_holes(mask)
    if keep_center:
        labeled, num = label(mask)
        if num > 0:
            h, w = mask.shape
            center_label = labeled[h // 2, w // 2]
            if center_label != 0:
                mask = labeled == center_label
    return mask.astype(np.uint8) * 255


def load_model(checkpoint, in_ch: int = 4, device="cpu"):
    """UNet(in_ch) with the weights in ``checkpoint`` (strict), in eval mode."""
    import torch
    from .model import UNet
    model = UNet(in_ch)
    state = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    model.eval()
    return model


def predict_probability(model, image: np.ndarray, device="cpu",
                        train_size: Sequence[int] = TRAIN_SIZE,
                        save_size: Sequence[int] = SAVE_SIZE) -> np.ndarray:
    """Probability map (float32) at ``save_size``."""
    import torch
    x = torch.from_numpy(preprocess(image, train_size)).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))
    probs = probs.squeeze().cpu().numpy()
    return cv2.resize(probs, (int(save_size[1]), int(save_size[0])),
                      interpolation=cv2.INTER_NEAREST)


def predict_mask(model, image: np.ndarray, device="cpu", threshold: float = THRESHOLD,
                 train_size: Sequence[int] = TRAIN_SIZE,
                 save_size: Sequence[int] = SAVE_SIZE) -> np.ndarray:
    probs = predict_probability(model, image, device, train_size, save_size)
    return postprocess_mask(probs, threshold=threshold)


def predict_folder(rgb_dir, dapi_dir, checkpoint, out_dir, ids: Optional[Iterable[str]] = None,
                   device: Optional[str] = None, threshold: float = THRESHOLD,
                   train_size: Sequence[int] = TRAIN_SIZE, save_size: Sequence[int] = SAVE_SIZE,
                   overwrite: bool = False) -> int:
    """Write ``<out_dir>/<uid>.png`` for every image set. Returns the number written."""
    import torch
    if rgb_dir is None and dapi_dir is None:
        raise ValueError("need rgb_dir and/or dapi_dir")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    in_ch = 3 * (rgb_dir is not None) + 1 * (dapi_dir is not None)
    model = load_model(checkpoint, in_ch=in_ch, device=device)
    if ids is None:
        ids = roi_io.ids_in_folder(rgb_dir if rgb_dir is not None else dapi_dir)
    ids = sorted(set(ids))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i, uid in enumerate(ids, 1):
        dst = out_dir / f"{uid}.png"
        if dst.exists() and not overwrite:
            continue
        image = roi_io.load_pair(rgb_dir, dapi_dir, uid)
        mask = predict_mask(model, image, device, threshold, train_size, save_size)
        if not cv2.imwrite(str(dst), mask):
            raise RuntimeError(f"could not write {dst}")
        written += 1
        if i % 50 == 0 or i == len(ids):
            logger.info("%d / %d image sets", i, len(ids))
    return written
