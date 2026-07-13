"""
ecdna_bench.baselines.label_engine
====================================
Harmonization adapter for Label Engine predictions.

Label Engine background
-----------------------
Label Engine was trained on the same dataset and splits as ecCount, by a
former lab member.  Prediction masks are delivered as PNGs.  The filename
convention is non-trivial: Label Engine appended ``_LE`` to the UID stem
and sometimes applied minor UID normalizations (``control`` → ``ctrl``,
trailing suffixes ``_a``, ``_b``, ``_f``, ``_s`` dropped).

This module contains the key canonicalization logic extracted from
``figure1_label_engine_plots.py`` (``build_labelengine_index`` and
``resolve_le_pred_path``) and re-exposes it as a clean library function.

Harmonization pipeline per image
---------------------------------
1. Build a canonicalized filename index from *pred_dir*.
2. Look up the prediction PNG for each UID using the canonicalized key
   (with suffix fallback).
3. Binarize at *threshold* (default 0.5).
4. Apply min-area-3 connected-component filter.
5. Save as uint8 PNG with values {0, 255}.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from ecdna_bench.baselines.ecseg import (
    apply_min_area_filter,
    load_and_binarize,
    write_binary_mask,
)

__all__ = [
    "harmonize_label_engine",
    "build_label_engine_index",
    "resolve_pred_path",
    "canonicalize_le_key",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# UID canonicalization (migrated from figure1_label_engine_plots.py)
# ---------------------------------------------------------------------------

def canonicalize_le_key(name: str) -> str:
    """Normalize a filename stem or UID to the Label Engine index key.

    Rules (matching the original ``figure1_label_engine_plots.py``):
    * Strip ``_LE`` suffix (case-insensitive).
    * ``control`` → ``ctrl``.
    * Remove trailing ``_bad``.
    * Collapse multiple underscores; strip leading/trailing underscores.
    """
    base = name.strip()
    # Remove trailing _LE suffix (case-insensitive)
    if base.lower().endswith("_le"):
        base = base[:-3]
    base = base.replace("control", "ctrl")
    base = re.sub(r"_bad$", "", base)
    base = re.sub(r"__+", "_", base).strip("_")
    return base.lower()


def build_label_engine_index(pred_dir: Path) -> Dict[str, Path]:
    """Build a canonicalized stem → path index for a Label Engine prediction dir.

    Parameters
    ----------
    pred_dir:
        Directory of Label Engine PNG files.

    Returns
    -------
    dict mapping canonicalized UID key → absolute Path.
    """
    idx: Dict[str, Path] = {}
    if not pred_dir.exists():
        return idx
    for p in sorted(pred_dir.glob("*.png")):
        key = canonicalize_le_key(p.stem)
        # First match wins (don't overwrite)
        idx.setdefault(key, p.resolve())
    return idx


def resolve_pred_path(
    uid:        str,
    le_index:   Dict[str, Path],
    pred_dir:   Path,
) -> Optional[Path]:
    """Locate the Label Engine prediction file for *uid*.

    Strategy (matching the original resolution logic):
    1. Direct key lookup.
    2. Key lookup after stripping a single trailing letter suffix
       (``_a``, ``_b``, ``_f``, ``_s``).
    3. Fallback: ``{pred_dir}/{uid}_LE.png``.

    Returns None only when the fallback path also does not exist.
    """
    key = canonicalize_le_key(uid)

    # 1. Direct
    p = le_index.get(key)
    if p is not None and p.is_file():
        return p

    # 2. Strip trailing letter suffix
    toks = key.split("_")
    if toks and toks[-1] in {"a", "b", "f", "s"}:
        short_key = "_".join(toks[:-1])
        p = le_index.get(short_key)
        if p is not None and p.is_file():
            return p

    # 3. Prefix-match fallback within the index
    uid_lower = uid.lower()
    for stem_key, path in le_index.items():
        if stem_key.startswith(uid_lower):
            return path

    # 4. Explicit fallback path (may not exist — caller handles)
    fallback = (pred_dir / f"{uid}_LE.png").resolve()
    if fallback.is_file():
        return fallback

    return None


# ---------------------------------------------------------------------------
# Label Engine harmonizer
# ---------------------------------------------------------------------------

def harmonize_label_engine(
    pred_dir:   Path,
    output_dir: Path,
    uid_list:   List[str],
    min_area:   int   = 3,
    threshold:  float = 0.5,
) -> Dict[str, str]:
    """Harmonize Label Engine predictions to canonical binary masks.

    For each UID, the function:
    1. Resolves the prediction PNG using the canonicalized LE filename index.
    2. Binarizes at *threshold*.
    3. Applies min-area-3 filter.
    4. Writes ``{output_dir}/{uid}.png`` with pixel values {0, 255}.

    Parameters
    ----------
    pred_dir:
        Directory containing Label Engine output PNGs.
    output_dir:
        Where to write the harmonized binary masks.
    uid_list:
        UIDs to process.
    min_area:
        Minimum connected-component area (default 3).
    threshold:
        Binarization threshold (default 0.5).

    Returns
    -------
    dict ``{uid: "ok" | "missing" | "error: <msg>"}``
    """
    pred_dir   = Path(pred_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    le_index = build_label_engine_index(pred_dir)
    results: Dict[str, str] = {}

    for uid in uid_list:
        out_path  = output_dir / f"{uid}.png"
        pred_path = resolve_pred_path(uid, le_index, pred_dir)

        if pred_path is None or not pred_path.is_file():
            results[uid] = "missing"
            logger.debug("LabelEngine | missing prediction for uid=%s", uid)
            continue

        try:
            binary   = load_and_binarize(pred_path, threshold=threshold, channel=None)
            filtered = apply_min_area_filter(binary, min_area=min_area)
            write_binary_mask(filtered, out_path)
            results[uid] = "ok"
        except Exception as exc:
            results[uid] = f"error: {exc}"
            logger.warning("LabelEngine | uid=%s error: %s", uid, exc)

    ok      = sum(1 for v in results.values() if v == "ok")
    missing = sum(1 for v in results.values() if v == "missing")
    errors  = sum(1 for v in results.values() if v.startswith("error"))
    logger.info("LabelEngine harmonize | ok=%d missing=%d errors=%d", ok, missing, errors)
    return results
