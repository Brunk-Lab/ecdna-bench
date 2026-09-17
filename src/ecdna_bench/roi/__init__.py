"""
ecdna_bench.roi
===============
Region-of-interest (ROI) model: predicts the metaphase ROI from the RGB and DAPI
images, so that images without a manual mask can be analyzed.

Written by River Summers (Brunk Lab); merged into ecdna-bench without changes to
the model or the post-processing. The released weights are the release asset
``roi_model_best_checkpoint.pth`` (training run ``train_final_6-25``, best epoch 97,
validation loss 0.1741, 35,923,337 parameters, four input channels).

Modules
-------
model   residual U-Net (``UNet``)
io      image loading (RGB + DAPI, TIFF or PNG detected from the file content)
infer   pre-processing, prediction and post-processing (no albumentations needed)
train   training as run for the released model (needs albumentations 2.0.8)

Command line: ``python -m ecdna_bench.cli.roi predict|train --help``.
"""
from .model import UNet

N_PARAMS_RELEASED = 35_923_337
IN_CHANNELS_RELEASED = 4

__all__ = ["UNet", "N_PARAMS_RELEASED", "IN_CHANNELS_RELEASED"]
