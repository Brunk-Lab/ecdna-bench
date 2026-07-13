"""
ecdna_bench.baselines
======================
Harmonization adapters for the three external baseline models.

Design contract
---------------
* Each module exposes one public ``harmonize_*(...)`` function.
* The function reads raw prediction files from *pred_dir*, normalizes them
  to a common binary uint8 mask format (foreground = 255, background = 0),
  applies a minimum connected-component area filter (default 3 px, matching
  the main evaluation framework), and writes the result to *output_dir*.
* Returns ``{uid: "ok" | "missing" | "error: <msg>"}`` for every requested UID.
* These adapters do NOT run the third-party models; they only harmonize
  already-produced prediction files.

Sub-modules
-----------
ecseg
    ecSeg multi-channel output → binary mask (channel 3 = ecDNA class).
mia
    MIA binary/near-binary output → canonical binary mask.
label_engine
    Label Engine probability map or binary mask → canonical binary mask.
"""

from importlib import import_module as _imp

__all__ = [
    "harmonize_ecseg",
    "harmonize_mia",
    "harmonize_label_engine",
    "resolve_pred_path",
    "load_and_binarize",
    "apply_min_area_filter",
    "write_binary_mask",
]


def __getattr__(name: str):
    _map = {
        "harmonize_ecseg":        "ecseg",
        "harmonize_mia":          "mia",
        "harmonize_label_engine": "label_engine",
        "resolve_pred_path":      "label_engine",
        "load_and_binarize":      "ecseg",
        "apply_min_area_filter":  "ecseg",
        "write_binary_mask":      "ecseg",
    }
    if name in _map:
        mod = _imp(f"ecdna_bench.baselines.{_map[name]}")
        return getattr(mod, name)
    raise AttributeError(f"module 'ecdna_bench.baselines' has no attribute {name!r}")
