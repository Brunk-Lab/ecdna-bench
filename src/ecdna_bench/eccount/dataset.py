"""
ecdna_bench.eccount.dataset
============================
PyTorch Dataset for ecCount training and evaluation.

Design rules
------------
* Soft target is generated AFTER resizing to the final training resolution
  (1024 × 1224), so σ is defined in model pixel space.
* All spatial augmentations are synchronized across image / GS / target / ROI.
* The dataset filters to ``count_mask_consistent == True`` rows.
* No disk paths are hard-coded; everything comes from the DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ecdna_bench.eccount.targets import SoftTargetConfig, make_centroid_gaussian_target

__all__ = ["DatasetConfig", "EcCountDataset"]

# Frozen training input size (paper §2)
_DEFAULT_IMAGE_SIZE: Tuple[int, int] = (1024, 1224)


@dataclass
class DatasetConfig:
    """Configuration for EcCountDataset.

    Attributes
    ----------
    image_size:
        Final (H, W) spatial size.  Defaults to the frozen training size.
    apply_roi_mask:
        Zero out RGB pixels outside the ROI mask before training.
    normalize_to_01:
        Scale uint8 RGB to [0, 1].
    return_gt_mask:
        Include binary GS mask in the returned dict.
    return_roi_mask:
        Include ROI mask in the returned dict.
    return_metadata:
        Include ``unique_id``, ``cell_line``, ``split`` strings.
    training:
        Enable data augmentation.
    hflip_prob, vflip_prob:
        Horizontal / vertical flip probabilities.
        Frozen training values: 0.5, 0.5.
    brightness_jitter_prob:
        Probability of brightness scaling.
        Frozen training value: 0.2.
    brightness_jitter_range:
        Multiplicative range for brightness scaling.
        Frozen training value: (0.9, 1.1).
    """
    image_size:               Optional[Tuple[int, int]] = _DEFAULT_IMAGE_SIZE
    apply_roi_mask:           bool  = True
    normalize_to_01:          bool  = True
    return_gt_mask:           bool  = True
    return_roi_mask:          bool  = False
    return_metadata:          bool  = True
    training:                 bool  = False
    # Augmentation (frozen paper values as defaults)
    hflip_prob:               float = 0.5
    vflip_prob:               float = 0.5
    brightness_jitter_prob:   float = 0.2
    brightness_jitter_range:  Tuple[float, float] = (0.9, 1.1)


class EcCountDataset(Dataset):
    """Full-image ecDNA soft-target dataset.

    Required DataFrame columns
    --------------------------
    ``unique_id``, ``cell_line``, ``split``, ``ecDNA_gt``,
    ``rgb_fullpath``, ``gt_fullpath``, ``count_mask_consistent``.
    Additionally ``roi_fullpath`` when ``apply_roi_mask`` or
    ``return_roi_mask`` is True.
    """

    REQUIRED_COLUMNS = {
        "unique_id", "cell_line", "split", "ecDNA_gt",
        "rgb_fullpath", "gt_fullpath", "count_mask_consistent",
    }

    def __init__(
        self,
        df:             pd.DataFrame,
        dataset_config: Optional[DatasetConfig]    = None,
        target_config:  Optional[SoftTargetConfig] = None,
    ) -> None:
        super().__init__()
        self.cfg        = dataset_config or DatasetConfig()
        self.target_cfg = target_config  or SoftTargetConfig()
        self._validate_config(self.cfg)

        missing = sorted(self.REQUIRED_COLUMNS - set(df.columns))
        if missing:
            raise ValueError(f"DataFrame missing required columns: {missing}")

        if (self.cfg.apply_roi_mask or self.cfg.return_roi_mask) and "roi_fullpath" not in df.columns:
            raise ValueError(
                "'roi_fullpath' column required when apply_roi_mask or return_roi_mask is True."
            )

        df = df.copy()
        df = df.loc[df["count_mask_consistent"].fillna(False)].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError("No rows remain after filtering count_mask_consistent == True.")

        self.df = df

    @staticmethod
    def _validate_config(cfg: DatasetConfig) -> None:
        for name in ("hflip_prob", "vflip_prob", "brightness_jitter_prob"):
            v = getattr(cfg, name)
            if not (0.0 <= v <= 1.0):
                raise ValueError(f"{name} must be in [0, 1], got {v}")
        lo, hi = cfg.brightness_jitter_range
        if lo <= 0 or hi <= 0 or lo > hi:
            raise ValueError(f"brightness_jitter_range invalid: {cfg.brightness_jitter_range}")

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _read_rgb(path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Cannot read RGB: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_gray(path: str) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Cannot read grayscale: {path}")
        return img

    # ------------------------------------------------------------------
    # Resize helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resize_rgb(rgb: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        H, W = size
        return cv2.resize(rgb, (W, H), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _resize_mask(mask: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
        H, W = size
        return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------------
    # Synchronized augmentation
    # ------------------------------------------------------------------

    def _augment(
        self,
        rgb:     np.ndarray,
        gt_mask: np.ndarray,
        target:  np.ndarray,
        roi:     Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not self.cfg.training:
            return rgb, gt_mask, target, roi

        if self.cfg.hflip_prob > 0.0 and np.random.rand() < self.cfg.hflip_prob:
            rgb     = np.flip(rgb,     axis=1).copy()
            gt_mask = np.flip(gt_mask, axis=1).copy()
            target  = np.flip(target,  axis=1).copy()
            if roi is not None:
                roi = np.flip(roi, axis=1).copy()

        if self.cfg.vflip_prob > 0.0 and np.random.rand() < self.cfg.vflip_prob:
            rgb     = np.flip(rgb,     axis=0).copy()
            gt_mask = np.flip(gt_mask, axis=0).copy()
            target  = np.flip(target,  axis=0).copy()
            if roi is not None:
                roi = np.flip(roi, axis=0).copy()

        if (self.cfg.brightness_jitter_prob > 0.0
                and np.random.rand() < self.cfg.brightness_jitter_prob):
            lo, hi = self.cfg.brightness_jitter_range
            factor = float(np.random.uniform(lo, hi))
            rgb = np.clip(rgb.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)

        return rgb, gt_mask, target, roi

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row      = self.df.iloc[idx]
        rgb_path = str(row["rgb_fullpath"])
        gt_path  = str(row["gt_fullpath"])
        roi_path = str(row["roi_fullpath"]) if "roi_fullpath" in self.df.columns else None

        rgb = self._read_rgb(rgb_path)
        gt  = self._read_gray(gt_path)
        gt_mask = (gt > 0).astype(np.uint8)

        roi: Optional[np.ndarray] = None
        if roi_path and (self.cfg.apply_roi_mask or self.cfg.return_roi_mask):
            roi_raw = self._read_gray(roi_path)
            roi     = (roi_raw > 0).astype(np.uint8)

        # Apply ROI masking before resize (preserves original boundary)
        if roi is not None and self.cfg.apply_roi_mask:
            rgb = rgb * roi[:, :, None]

        # Resize to training resolution
        if self.cfg.image_size is not None:
            rgb     = self._resize_rgb(rgb,     self.cfg.image_size)
            gt_mask = self._resize_mask(gt_mask, self.cfg.image_size)
            if roi is not None:
                roi = self._resize_mask(roi, self.cfg.image_size)

        # Generate soft target AFTER resize (σ defined in model pixel space)
        target = make_centroid_gaussian_target(gt_mask, config=self.target_cfg)

        # Synchronized augmentation
        rgb, gt_mask, target, roi = self._augment(rgb, gt_mask, target, roi)

        # Convert to float tensors
        rgb    = rgb.astype(np.float32)
        if self.cfg.normalize_to_01:
            rgb /= 255.0
        gt_mask = gt_mask.astype(np.float32)
        target  = target.astype(np.float32)

        # HWC → CHW; HW → 1HW
        image_t  = torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float()
        target_t = torch.from_numpy(target[None]).float()
        gt_t     = torch.from_numpy(gt_mask[None]).float()

        sample: Dict[str, Any] = {
            "image":  image_t,
            "target": target_t,
            "count":  torch.tensor(int(row["ecDNA_gt"]), dtype=torch.long),
        }

        if self.cfg.return_gt_mask:
            sample["gt_mask"] = gt_t

        if self.cfg.return_roi_mask and roi is not None:
            sample["roi_mask"] = torch.from_numpy(roi[None].astype(np.float32)).float()

        if self.cfg.return_metadata:
            sample["unique_id"] = str(row["unique_id"])
            sample["cell_line"] = str(row["cell_line"])
            sample["split"]     = str(row["split"])

        return sample
