"""
ecdna_bench.eccount.infer
==========================
End-to-end inference for ecCount: image → prediction masks + peak list.

Two output modes (from §2 of REWRITE_PLAN.md)
----------------------------------------------
1. **Threshold mask** — binarize the sigmoid probability map at 0.5.
2. **Peaks mask**     — disks of radius 3 px at retained peaks.

Design rules
------------
* ``infer_one`` is a pure function: no disk I/O, no multiprocessing.
  The CLI (run_eccount.py) loads the model ONCE and calls this per image.
* The caller is responsible for ``model.eval()`` and device placement.
* If *roi_mask* is provided, it is applied to the probability map AFTER
  sigmoid, inside the postprocess pipeline (smooth → ROI → maxima → NMS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ecdna_bench.eccount.postprocess import (
    PostprocessConfig,
    detect_points_from_map,
    peaks_to_mask,
    prob_to_threshold_mask,
)

__all__ = ["InferConfig", "EcCountInferResult", "infer_one", "load_checkpoint"]


@dataclass
class InferConfig:
    """Configuration for ecCount inference.

    All defaults are the frozen paper values.

    Attributes
    ----------
    postprocess:
        Peak-extraction post-processing config (passed to
        ``detect_points_from_map``).
    threshold_mask_cutoff:
        Sigmoid threshold for the binary threshold mask.
        0.5 is the standard decision boundary.
    peaks_disk_radius:
        Radius of the disk drawn at each retained peak in the peaks mask.
        Kept here (not in PostprocessConfig) because it affects only the
        rendered output mask, not the detection logic.
    """
    postprocess:            PostprocessConfig = field(default_factory=PostprocessConfig)
    threshold_mask_cutoff:  float = 0.5
    peaks_disk_radius:      int   = 3


@dataclass
class EcCountInferResult:
    """Return value of ``infer_one``.

    Attributes
    ----------
    prob_map:
        (H, W) float32 sigmoid probability map, range [0, 1].
    threshold_mask:
        (H, W) uint8 binary mask: 255 where prob ≥ threshold_mask_cutoff.
    peaks:
        List of (x, y, score) tuples after full post-processing pipeline.
    peaks_mask:
        (H, W) uint8 binary mask: disks at each retained peak centre.
    n_peaks:
        Number of detected peaks (= predicted ecDNA count).
    """
    prob_map:        np.ndarray
    threshold_mask:  np.ndarray
    peaks:           List[Tuple[int, int, float]]
    peaks_mask:      np.ndarray
    n_peaks:         int


def load_checkpoint(
    path:   str,
    model:  nn.Module,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Load a saved ecCount checkpoint into *model*.

    Parameters
    ----------
    path:
        Path to the ``.pt`` checkpoint file written by ``eccount.train``.
    model:
        ecCount U-Net to load weights into (already instantiated).
    device:
        Device string for ``torch.load``.

    Returns
    -------
    dict
        The full checkpoint payload (contains ``best_val_loss``,
        ``best_epoch``, ``run_config``, etc.).

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    KeyError
        If the checkpoint does not contain ``model_state_dict``.
    """
    ckpt_path = Path(str(path))
    if not ckpt_path.is_file():
        raise FileNotFoundError(
            f"ecCount checkpoint not found: {ckpt_path}\n"
            f"Either place the released weights (eccount_best.pt, an asset of "
            f"the GitHub release) at this path, or set paths.eccount_checkpoint "
            f"in configs/paths.local.yaml to the best_model.pt written by "
            f"train_eccount. Do not rename a trained model to eccount_best.pt."
        )

    ckpt = torch.load(str(ckpt_path), map_location=device)

    if "model_state_dict" not in ckpt:
        raise KeyError(
            f"Checkpoint at {ckpt_path} is missing 'model_state_dict'. "
            f"Keys found: {sorted(ckpt.keys())}"
        )

    model.load_state_dict(ckpt["model_state_dict"])

    import logging
    logging.getLogger(__name__).info(
        "Loaded ecCount checkpoint: best_epoch=%s  best_val_loss=%.4f  path=%s",
        ckpt.get("best_epoch", "?"),
        float(ckpt.get("best_val_loss", float("nan"))),
        ckpt_path,
    )
    return ckpt


def infer_one(
    image_tensor: torch.Tensor,
    roi_mask:     Optional[np.ndarray],
    model:        nn.Module,
    cfg:          Optional[InferConfig] = None,
) -> EcCountInferResult:
    """Run ecCount inference on one image.

    Parameters
    ----------
    image_tensor:
        Float tensor of shape ``(1, 3, H, W)`` or ``(3, H, W)``,
        values in [0, 1].  Must already be on the correct device.
    roi_mask:
        Optional (H, W) binary numpy mask (1 = inside ROI).
        Passed to the postprocess pipeline; applied AFTER smoothing.
    model:
        ecCount U-Net.  Caller must call ``model.eval()`` beforehand.
    cfg:
        ``InferConfig``.  Uses frozen paper defaults if None.

    Returns
    -------
    EcCountInferResult
    """
    cfg = cfg or InferConfig()

    x = image_tensor
    if x.ndim == 3:
        x = x.unsqueeze(0)   # (3, H, W) → (1, 3, H, W)

    with torch.no_grad():
        logits = model(x)                                 # (1, 1, H, W)

    probs = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)  # (H, W)

    # Threshold mask — simple binarization, no post-processing
    threshold_mask = prob_to_threshold_mask(
        probs, threshold=cfg.threshold_mask_cutoff
    )

    # Peaks via full postprocess pipeline (smooth → ROI → maxima → NMS)
    peaks = detect_points_from_map(
        probs, roi_mask=roi_mask, config=cfg.postprocess
    )

    # Peaks mask — disks at peak centres
    peaks_mask = peaks_to_mask(
        peaks, shape=probs.shape, disk_radius=cfg.peaks_disk_radius
    )

    return EcCountInferResult(
        prob_map       = probs,
        threshold_mask = threshold_mask,
        peaks          = peaks,
        peaks_mask     = peaks_mask,
        n_peaks        = len(peaks),
    )