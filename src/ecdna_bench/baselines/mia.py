"""
ecdna_bench.baselines.mia
==========================
Harmonization adapter for MIA (Mask-based Image Analysis) predictions.

MIA background
--------------
MIA prediction masks were supplied directly by the MIA authors from their
top-performing configuration.  No retraining was performed.  The masks are
delivered as grayscale or binary PNGs where foreground pixels represent
predicted ecDNA.  Some masks may use values {0, 1} or {0, 255}; both are
handled correctly by thresholding at 0.

Harmonization pipeline per image
---------------------------------
1. Load the prediction PNG (grayscale or single-channel).
2. Binarize: foreground = pixel_value > 0.
3. Apply min-area-3 connected-component filter.
4. Save as uint8 PNG with values {0, 255}.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from ecdna_bench.baselines.ecseg import (
    apply_min_area_filter,
    load_and_binarize,
    write_binary_mask,
    _build_file_index,
    _locate_file,
)

__all__ = ["harmonize_mia"]

logger = logging.getLogger(__name__)


def harmonize_mia(
    pred_dir:   Path,
    output_dir: Path,
    uid_list:   List[str],
    min_area:   int = 3,
    threshold:  float = 0.0,
) -> Dict[str, str]:
    """Harmonize MIA predictions to canonical binary masks.

    For each UID, the function:
    1. Looks for a prediction PNG named ``{uid}.png`` or ``{uid}_mia.png``
       in *pred_dir*.
    2. Binarizes at *threshold* (default 0 → any nonzero pixel is foreground).
    3. Applies min-area-3 filter.
    4. Writes ``{output_dir}/{uid}.png`` with pixel values {0, 255}.

    Parameters
    ----------
    pred_dir:
        Directory containing MIA output PNGs.
    output_dir:
        Where to write the harmonized binary masks.
    uid_list:
        UIDs to process.
    min_area:
        Minimum connected-component area (default 3).
    threshold:
        Binarization threshold.  Pixels strictly above this are foreground.
        Default 0 handles both {0, 1} and {0, 255} encoded masks.

    Returns
    -------
    dict ``{uid: "ok" | "missing" | "error: <msg>"}``
    """
    pred_dir   = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_index = _build_file_index(pred_dir)
    results: Dict[str, str] = {}

    for uid in uid_list:
        out_path  = output_dir / f"{uid}.png"
        pred_path = _locate_file(uid, file_index, pred_dir,
                                 suffixes=["", "_mia"])
        if pred_path is None:
            results[uid] = "missing"
            logger.debug("MIA | missing prediction for uid=%s", uid)
            continue

        try:
            binary   = load_and_binarize(pred_path, threshold=threshold, channel=None)
            filtered = apply_min_area_filter(binary, min_area=min_area)
            write_binary_mask(filtered, out_path)
            results[uid] = "ok"
        except Exception as exc:
            results[uid] = f"error: {exc}"
            logger.warning("MIA | uid=%s error: %s", uid, exc)

    ok      = sum(1 for v in results.values() if v == "ok")
    missing = sum(1 for v in results.values() if v == "missing")
    errors  = sum(1 for v in results.values() if v.startswith("error"))
    logger.info("MIA harmonize | ok=%d missing=%d errors=%d", ok, missing, errors)
    return results
