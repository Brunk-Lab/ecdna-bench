"""
ecdna_bench.classical
=====================
Classical ecDNA detection pipeline — inference side.

The four modules in this sub-package implement the *inference* path of the
optimized classical pipeline that is compared against ecCount and the external
baselines in the benchmark.

Sub-modules
-----------
preprocess
    Atomic preprocessing operators (top-hat, CLAHE, gamma, …), the registry
    dict, and the chain runner ``apply_chain``.
detect
    Otsu-threshold → morphological close → connected components → area filter
    (``detect_objects``) and greedy centroid-based merge (``merge_close_objects``).
    Also contains the HSV-based ecDNA/chromosome classifier
    (``classify_hsv``, ``label_objects_hsv``).
postprocess
    Eval-time split heuristic (``split_large_predictions``) and the
    ``objects_to_mask`` helpers.
infer
    End-to-end ``infer_one(sample, params)`` that consumes a ``ClassicalParams``
    object and returns a prediction mask + per-image count dict.
    Multi-image parallelism lives in ``cli/run_classical.py``.
"""

from importlib import import_module as _imp

# Lazy re-exports: importing ecdna_bench.classical will surface these
# without pulling in cv2/numpy until they're actually used.
__all__ = [
    # preprocess
    "PREPROCESSING_REGISTRY",
    "apply_chain",
    "rgb_to_gray_preserve",
    "mask_with_roi",
    # detect
    "detect_objects",
    "merge_close_objects",
    "classify_hsv",
    "label_objects_hsv",
    # postprocess
    "split_large_predictions",
    "objects_to_mask",
    # infer
    "ClassicalParams",
    "load_frozen_params",
    "infer_one",
]


def __getattr__(name: str):
    _module_map = {
        "PREPROCESSING_REGISTRY": "preprocess",
        "apply_chain":            "preprocess",
        "rgb_to_gray_preserve":   "preprocess",
        "mask_with_roi":          "preprocess",
        "detect_objects":         "detect",
        "merge_close_objects":    "detect",
        "classify_hsv":           "detect",
        "label_objects_hsv":      "detect",
        "split_large_predictions": "postprocess",
        "objects_to_mask":        "postprocess",
        "ClassicalParams":        "infer",
        "load_frozen_params":     "infer",
        "infer_one":              "infer",
    }
    if name in _module_map:
        mod = _imp(f"ecdna_bench.classical.{_module_map[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'ecdna_bench.classical' has no attribute {name!r}")
