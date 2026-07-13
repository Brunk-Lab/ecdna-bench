"""
ecdna_bench.classical.infer
============================
End-to-end classical inference for one image.

Design rules
------------
* ``infer_one(sample, params)`` is a **pure function**: it takes a
  ``data.samples.Sample`` and a ``ClassicalParams`` dataclass, and returns
  an ``InferResult`` dataclass.  No global state, no multiprocessing.

* Parallelism lives in ``cli/run_classical.py``.  ``infer_one`` is the unit
  of work that the CLI distributes across processes.

* JSON parsing lives here via ``load_frozen_params`` and
  ``frozen_params_for_cell_line``.  The optimizer stages produce a
  Stage-3 JSON; this module reads it into ``ClassicalParams`` objects.

* HSV classification (ecDNA vs chromosome) is included in every inference
  run.  The counts are returned in ``InferResult``; the CLI decides whether
  to write them.

* The prediction mask is bbox-fill (stable, cheap).  If the caller wants
  per-object masks, they can call ``postprocess.objects_to_mask(...,
  use_object_masks=True)`` on ``result.pred_objects``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from ecdna_bench.classical.preprocess import (
    rgb_to_gray_preserve,
    mask_with_roi,
    apply_chain,
    default_chain,
    _unpack_flat_params,
)
from ecdna_bench.classical.detect import (
    detect_objects,
    merge_close_objects,
    label_objects_hsv,
)
from ecdna_bench.classical.postprocess import objects_to_mask_bbox_fill

__all__ = [
    "ClassicalParams",
    "InferResult",
    "load_frozen_params",
    "frozen_params_for_cell_line",
    "infer_one",
]


# ---------------------------------------------------------------------------
# Typed parameters dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassicalParams:
    """All parameters needed to run ``infer_one`` on one image.

    Attributes
    ----------
    combo:
        Ordered list of preprocessing op names (registry keys).
    preproc_params:
        ``{op_name: {param: value}}`` — unpacked from the Stage-3 flat dict.
    threshold_factor:
        Otsu multiplier for thresholding.
    morph_close_kernel:
        Diameter of the morphological closing kernel.
    min_area, max_area:
        Area filter bounds (px).
    merge_distance:
        Centroid-distance threshold for greedy merging.
    white_value_threshold, white_saturation_threshold:
        HSV thresholds for the chromosome classifier.
    cell_line:
        Informational; not used by the algorithm.
    """
    combo:                      List[str]
    preproc_params:             Dict[str, Dict[str, Any]]

    preprocess_mode:            str = "chain"
    default_chain_params:       Dict[str, Any] = field(default_factory=dict)

    threshold_factor:           float = 1.0
    morph_close_kernel:         int   = 5
    min_area:                   float = 3.0
    max_area:                   float = 900.0
    merge_distance:             float = 5.0
    white_value_threshold:      float = 170.0
    white_saturation_threshold: float = 50.0
    cell_line:                  Optional[str] = None


@dataclass
class InferResult:
    """Return value of ``infer_one``.

    Attributes
    ----------
    pred_mask:
        H×W uint8 binary mask {0, 255} from bbox fill.
    pred_objects:
        List of detected + merged + labelled object dicts.
    n_total:
        Total number of predicted objects.
    n_ecdna:
        ecDNA count (HSV-labelled).
    n_chromosome:
        Chromosome count (HSV-labelled).
    status:
        ``"ok"`` on success, error string on failure.
    debug:
        Optional dict of intermediate arrays (filled when caller passes
        ``return_debug=True`` to ``infer_one``).
    """
    pred_mask:    np.ndarray
    pred_objects: List[Dict]
    n_total:      int
    n_ecdna:      int
    n_chromosome: int
    status:       str = "ok"
    debug:        Optional[Dict[str, Any]] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# JSON loading helpers
# ---------------------------------------------------------------------------

def load_frozen_params(path: Path) -> Dict[str, Any]:
    """Load the Stage-3 best-params JSON from *path*.

    The file is produced by ``classical_opt.freeze.write_frozen_json`` and
    committed at ``configs/classical/stage3_frozen_params.json``.

    Parameters
    ----------
    path:
        Path to the JSON file.

    Returns
    -------
    dict
        Raw JSON object (cell-line-keyed nested dict).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Frozen params JSON not found: {path}")
    with open(path) as f:
        return json.load(f)


def _pick_best_payload(cell_block: Dict[str, Any]) -> Dict[str, Any]:
    """Select the combo payload with the highest ``best_test_f1``.

    The Stage-3 JSON may list several combo keys per cell line (if multiple
    combos tied during optimisation).  We deterministically pick the best.
    """
    best: Optional[Dict[str, Any]] = None
    best_f1 = -1.0
    for payload in cell_block.values():
        raw_f1 = payload.get("best_test_f1", -1.0)
        f1 = -1.0 if raw_f1 is None else float(raw_f1)
        if best is None or f1 > best_f1:
            best_f1 = f1
            best = payload
    if best is None:
        raise ValueError("Empty cell_block — no payload found.")
    return best


def frozen_params_for_cell_line(
    frozen_json: Dict[str, Any],
    cell_line: str,
) -> ClassicalParams:
    """Extract and parse the best ``ClassicalParams`` for *cell_line*.

    Parameters
    ----------
    frozen_json:
        The dict returned by ``load_frozen_params``.
    cell_line:
        Cell-line name (must be a key in *frozen_json*).

    Returns
    -------
    ClassicalParams
    """
    if cell_line in frozen_json:
        cell_block = frozen_json[cell_line]
    elif "__global__" in frozen_json:
        cell_block = frozen_json["__global__"]
    else:
        available = sorted(frozen_json.keys())
        raise KeyError(
            f"Cell line {cell_line!r} not found in frozen params and no "
            f"'__global__' fallback exists. Available: {available}"
        )

    payload = _pick_best_payload(cell_block)

    combo: List[str]             = payload.get("combo", [])
    flat_preproc: Dict[str, Any] = payload.get("fixed_preproc_params", {})
    fixed_hsv:    Dict[str, Any] = payload.get("fixed_hsv_params", {})
    theta:        Dict[str, Any] = payload["best_theta"]

    return ClassicalParams(
        combo                      = combo,
        preproc_params             = _unpack_flat_params(flat_preproc),
        preprocess_mode            = payload.get("preprocess_mode", "chain"),
        default_chain_params       = payload.get("default_chain_params", {}),
        threshold_factor           = float(theta["threshold_factor"]),
        morph_close_kernel         = int(round(float(theta["morph_close_kernel"]))),
        min_area                   = float(theta["min_area"]),
        max_area                   = float(theta["max_area"]),
        merge_distance             = float(theta["merge_distance"]),
        white_value_threshold      = float(fixed_hsv.get("white_value_threshold", 170.0)),
        white_saturation_threshold = float(fixed_hsv.get("white_saturation_threshold", 50.0)),
        cell_line                  = cell_line,
    )


# ---------------------------------------------------------------------------
# End-to-end inference
# ---------------------------------------------------------------------------

def infer_one(
    rgb: np.ndarray,
    roi: Optional[np.ndarray],
    params: ClassicalParams,
    return_debug: bool = False,
) -> InferResult:
    """Run the full classical pipeline on one image.

    Parameters
    ----------
    rgb:
        H×W×3 BGR image (OpenCV convention).  Must be uint8.
    roi:
        Optional H×W ROI / DAPI mask.  Pixels outside the mask are zeroed
        before preprocessing.
    params:
        Fully-specified ``ClassicalParams`` for this image's cell line.
    return_debug:
        If ``True``, ``InferResult.debug`` contains intermediate arrays
        ``{"simple_gray", "enhanced"}``.

    Returns
    -------
    InferResult
    """
    H, W = rgb.shape[:2]
    empty_mask = np.zeros((H, W), dtype=np.uint8)

    # 1. Apply ROI mask if provided
    if roi is not None:
        rgb_masked = mask_with_roi(rgb, roi, mode="zero")
    else:
        rgb_masked = rgb

    # 2. Grayscale (probe-preserving)
    try:
        if params.preprocess_mode == "default_chain":
            simple_gray, enhanced = default_chain(
                rgb_masked,
                params.default_chain_params,
            )
        elif params.preprocess_mode == "chain":
            simple_gray = rgb_to_gray_preserve(rgb_masked, mode="max")
            if simple_gray.dtype != np.uint8:
                simple_gray = np.clip(simple_gray, 0, 255).astype(np.uint8)

            # 3. Preprocessing chain
            enhanced = apply_chain(simple_gray, params.combo, params.preproc_params)
        else:
            raise ValueError(f"Unknown preprocess_mode: {params.preprocess_mode!r}")
    except Exception as exc:
        return InferResult(
            pred_mask=empty_mask, pred_objects=[], n_total=0,
            n_ecdna=0, n_chromosome=0, status=f"preproc_error:{exc}",
        )

    # 4. Detect + merge
    pred_objs = detect_objects(
        enhanced,
        threshold_factor   = params.threshold_factor,
        morph_close_kernel = params.morph_close_kernel,
        min_area           = params.min_area,
        max_area           = params.max_area,
    )
    merged_objs = merge_close_objects(pred_objs, merge_distance=params.merge_distance)

    # 5. HSV labelling (ecDNA vs chromosome)
    labelled_objs, counts = label_objects_hsv(
        merged_objs,
        rgb_masked,
        white_value_threshold      = params.white_value_threshold,
        white_saturation_threshold = params.white_saturation_threshold,
    )

    # 6. Prediction mask (bbox fill — cheap and stable)
    pred_mask = objects_to_mask_bbox_fill(labelled_objs, shape_hw=(H, W))

    debug: Optional[Dict[str, Any]] = (
        {"simple_gray": simple_gray, "enhanced": enhanced}
        if return_debug else None
    )

    return InferResult(
        pred_mask    = pred_mask,
        pred_objects = labelled_objs,
        n_total      = len(labelled_objs),
        n_ecdna      = counts["ecDNA"],
        n_chromosome = counts["chromosome"],
        status       = "ok",
        debug        = debug,
    )
