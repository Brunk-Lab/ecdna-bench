"""
ecdna_bench.eccount
====================
ecCount — the novel deep-learning contribution of the paper.

Sub-modules
-----------
targets
    Soft Gaussian target map generation from GS masks.
    ``make_centroid_gaussian_target``, ``SoftTargetConfig``
dataset
    PyTorch Dataset with synchronized augmentation.
    ``EcCountDataset``, ``DatasetConfig``
model
    Compact U-Net (7,849,601 trainable parameters).
    ``build_model``, ``ModelConfig``
losses
    Weighted BCE + Soft Dice loss.
    ``build_loss``, ``LossConfig``
train
    Resumable training loop.
    ``train_eccount``, ``TrainConfig``
postprocess
    Peak extraction from predicted probability maps.
    ``detect_points_from_map``, ``PostprocessConfig``
infer
    End-to-end inference for one image.
    ``infer_one``, ``InferConfig``
tuning
    Ablation / grid-search helpers (no argparse):
    ``tuning.sigma``, ``tuning.loss``, ``tuning.schedule``, ``tuning.postprocess``
"""

from importlib import import_module as _imp

__all__ = [
    "SoftTargetConfig", "make_centroid_gaussian_target",
    "DatasetConfig", "EcCountDataset",
    "ModelConfig", "build_model",
    "LossConfig", "build_loss",
    "TrainConfig", "train_eccount",
    "PostprocessConfig", "detect_points_from_map",
    "InferConfig", "infer_one",
]

_MAP = {
    "SoftTargetConfig":            "targets",
    "make_centroid_gaussian_target": "targets",
    "DatasetConfig":               "dataset",
    "EcCountDataset":              "dataset",
    "ModelConfig":                 "model",
    "build_model":                 "model",
    "LossConfig":                  "losses",
    "build_loss":                  "losses",
    "TrainConfig":                 "train",
    "train_eccount":               "train",
    "PostprocessConfig":           "postprocess",
    "detect_points_from_map":      "postprocess",
    "InferConfig":                 "infer",
    "infer_one":                   "infer",
}


def __getattr__(name: str):
    if name in _MAP:
        mod = _imp(f"ecdna_bench.eccount.{_MAP[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'ecdna_bench.eccount' has no attribute {name!r}")
