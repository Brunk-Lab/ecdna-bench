"""
ROI model training, as run for the released model (River Summers).

Released run: ``--rgb River/input/rgb --dapi River/private/dapi_png
--roi_mask River/input/roi_mask --splits <train_ids.csv, val_ids.csv>
--resize 1024 1224 --num_epochs 200 --batch_size 2 --accum_steps 4 --num_cpus 5``
on one NVIDIA L40 GPU (23.7 h), albumentations 2.0.8, PyTorch 2.10.0.

Recorded behavior kept on purpose:

* seed 42 is set before the network is built;
* loss = 0.5 BCE-with-logits (pos_weight 1.0) + 0.5 soft Dice (smooth 1e-6);
* AdamW, lr 1e-4, weight decay 1e-4; ReduceLROnPlateau(factor 0.5, patience 8);
* ``autocast`` wraps only the loss bookkeeping, so the forward pass runs in
  float32 and the GradScaler has no numerical effect;
* the checkpoint with the lowest validation loss is saved as
  ``best_checkpoint.pth`` (a plain state_dict), the last one as
  ``final_checkpoint.pth``; ``train_history.csv`` has one row per epoch.

Differences from River's script that do not change a GPU run with several
CPUs: fused AdamW only on CUDA (the fused CPU optimizer is not available in
every PyTorch build), and persistent data-loader workers only when there are
workers (River's script fails with one CPU).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from . import io as roi_io


def _albumentations():
    try:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
    except ImportError as exc:
        raise ImportError("ROI training needs albumentations (the released model used 2.0.8): "
                          "python -m pip install albumentations==2.0.8") from exc
    return A, ToTensorV2


def build_transforms(img_size):
    import cv2
    A, ToTensorV2 = _albumentations()
    train_transform = A.Compose([
        A.Resize(img_size[0], img_size[1],
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        # --- Geometric ---
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-20, 20),
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST,
                 border_mode=cv2.BORDER_CONSTANT, p=0.5),
        # --- Elastic / Deformation ---
        A.ElasticTransform(interpolation=cv2.INTER_LINEAR,
                           mask_interpolation=cv2.INTER_NEAREST, p=0.1),
        A.GridDistortion(interpolation=cv2.INTER_LINEAR,
                         mask_interpolation=cv2.INTER_NEAREST, p=0.1),
        # --- Intensity (image only) ---
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        # --- Normalize (image only) ---
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    val_transform = A.Compose([
        A.Resize(img_size[0], img_size[1],
                 interpolation=cv2.INTER_LINEAR, mask_interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])
    return train_transform, val_transform


def _torch_parts():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    class SegmentationDataset(Dataset):
        def __init__(self, ids, rgb_dir, dapi_dir, mask_dir, transform):
            self.ids = sorted(ids)
            self.rgb_dir, self.dapi_dir, self.mask_dir = rgb_dir, dapi_dir, mask_dir
            self.transform = transform

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, idx):
            uid = self.ids[idx]
            image = roi_io.load_pair(self.rgb_dir, self.dapi_dir, uid)
            mpath = roi_io.find_image(self.mask_dir, uid)
            if mpath is None:
                raise FileNotFoundError(f"no ROI mask for {uid} in {self.mask_dir}")
            mask = roi_io.read_mask(mpath)
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            return image, mask

    class DiceLoss(nn.Module):
        def __init__(self, smooth=1e-6):
            super().__init__()
            self.smooth = smooth

        def forward(self, logits, targets):
            b = logits.shape[0]
            probs = torch.sigmoid(logits).view(b, -1)
            targets = targets.view(b, -1)
            inter = (probs * targets).sum(dim=1)
            dice = 1.0 - (2.0 * inter + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)
            return dice.mean()

    class BCEDiceLoss(nn.Module):
        def __init__(self, pos_weight, bce_weight=0.5, dice_weight=0.5):
            super().__init__()
            self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            self.dice = DiceLoss()
            total = bce_weight + dice_weight
            self.bce_weight, self.dice_weight = bce_weight / total, dice_weight / total

        def forward(self, logits, targets):
            return self.bce_weight * self.bce(logits, targets) + self.dice_weight * self.dice(logits, targets)

    return SegmentationDataset, BCEDiceLoss


HISTORY_COLUMNS = ["epoch", "train_loss", "val_loss", "lr", "best_val_loss", "best_epoch", "elapsed_s"]


def train_roi(rgb_dir, dapi_dir, mask_dir, splits_dir, output_dir, name="ROI_training",
              img_size=(1024, 1224), batch_size=2, accum_steps=4, num_epochs=200,
              lr=1e-4, weight_decay=1e-4, pos_weight=1.0, device=None, num_cpus=5,
              max_images=None):
    """Train the ROI model. ``splits_dir`` holds train_ids.csv and val_ids.csv.

    ``max_images`` (train and val each) is for smoke tests only.
    Returns the run folder.
    """
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from .model import UNet

    if (rgb_dir is None and dapi_dir is None) or mask_dir is None:
        raise ValueError("need rgb_dir and/or dapi_dir, and mask_dir")
    train_transform, val_transform = build_transforms(img_size)
    SegmentationDataset, BCEDiceLoss = _torch_parts()

    torch.manual_seed(42)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    num_workers = max(int(num_cpus) - 1, 0)
    in_ch = 3 * (rgb_dir is not None) + 1 * (dapi_dir is not None)
    model = UNet(in_ch)
    criterion = BCEDiceLoss(pos_weight=torch.tensor([pos_weight], device=dev))

    train_ids = roi_io.read_ids(Path(splits_dir) / "train_ids.csv")
    val_ids = roi_io.read_ids(Path(splits_dir) / "val_ids.csv")
    if max_images:
        train_ids, val_ids = train_ids[:max_images], val_ids[:max_images]
    loader_kw = dict(batch_size=batch_size, num_workers=num_workers,
                     pin_memory=(dev.type == "cuda"), persistent_workers=num_workers > 0)
    train_loader = DataLoader(SegmentationDataset(train_ids, rgb_dir, dapi_dir, mask_dir, train_transform),
                              shuffle=True, **loader_kw)
    val_loader = DataLoader(SegmentationDataset(val_ids, rgb_dir, dapi_dir, mask_dir, val_transform),
                            shuffle=False, **loader_kw)

    model = model.to(dev).to(memory_format=torch.channels_last)
    torch.backends.cudnn.benchmark = True
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay,
                            fused=(dev.type == "cuda"))
    scaler = torch.amp.GradScaler(dev.type)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=8)

    out_path = Path(output_dir) / name
    out_path.mkdir(parents=True, exist_ok=True)
    hist_path = out_path / "train_history.csv"
    with open(hist_path, "w", newline="") as fh:
        csv.writer(fh).writerow(HISTORY_COLUMNS)

    min_val_loss, best_epoch, start = float("inf"), -1, time.time()
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        train_loss = 0.0
        for step, (images, masks) in enumerate(train_loader):
            images = images.to(dev, non_blocking=True)
            masks = masks.to(dev, non_blocking=True)
            outputs = model(images)
            loss = criterion(outputs, masks)
            with torch.amp.autocast(device_type=dev.type):   # as in the released run
                train_loss += loss.item()
                loss = loss / accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        avg_train = train_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                val_loss += criterion(model(images.to(dev)), masks.to(dev)).item()
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < min_val_loss:
            min_val_loss, best_epoch = avg_val, epoch
            torch.save(model.state_dict(), out_path / "best_checkpoint.pth")
        with open(hist_path, "a", newline="") as fh:
            csv.writer(fh).writerow([epoch, avg_train, avg_val, optimizer.param_groups[0]["lr"],
                                     min_val_loss, best_epoch, time.time() - start])

    torch.save(model.state_dict(), out_path / "final_checkpoint.pth")
    return out_path
