"""
ecdna_bench.classical_opt.freeze
==================================
Commit the best per-cell-line Stage-3 parameters to the frozen JSON that
is version-controlled in ``configs/classical/stage3_frozen_params.json``.

This module reads all per-cell-line Stage-3 JSON files produced by
``stage3_detmerge.run_stage3``, selects the best combo per cell line
(highest ``best_test_f1``), and writes a single merged JSON.

The frozen JSON is the single source of truth for published results.
It is what ``classical.infer.load_frozen_params`` reads at inference time.

JSON schema (frozen output)
----------------------------
{
  "NCIH2170": {
    "top_hat__clahe__apply_sigmoid": {
      "combo":                ["top_hat", "clahe", "apply_sigmoid"],
      "fixed_preproc_params": {"top_hat__k_size": 15.0, ...},
      "fixed_hsv_params":     {"white_value_threshold": 170.0, ...},
      "best_theta":           {"threshold_factor": 0.9, ...},
      "best_train_f1":        0.81,
      "best_test_f1":         0.79
    }
  },
  ...
}

Only the single best combo per cell line is kept; unused combos are dropped.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import re

__all__ = ["freeze_best_params", "load_and_merge_stage3_jsons"]

logger = logging.getLogger(__name__)

def _norm_cell_line_name(s: str) -> str:
    """Normalize cell-line names for matching only.

    Examples:
    NCI-H2170, NCI_H2170, NCIH2170 -> NCIH2170
    """
    return re.sub(r"[^A-Za-z0-9]+", "", str(s)).upper()

def load_and_merge_stage3_jsons(stage3_json_dir: Path) -> Dict[str, Any]:
    """Read all ``*.json`` files in *stage3_json_dir* and merge into one dict.

    Each file is expected to map ``cell_line → {combo_id → payload}``.

    Parameters
    ----------
    stage3_json_dir:
        Directory containing per-cell-line Stage-3 JSON files
        (``{safe_cell_line}.json``).

    Returns
    -------
    dict
        ``{cell_line: {combo_id: payload}}``.
    """
    stage3_json_dir = Path(stage3_json_dir)
    if not stage3_json_dir.exists():
        raise FileNotFoundError(f"Stage-3 JSON dir not found: {stage3_json_dir}")

    merged: Dict[str, Any] = {}
    json_files = sorted(stage3_json_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {stage3_json_dir}")

    for fpath in json_files:
        try:
            with open(fpath) as f:
                data = json.load(f)
        except Exception as exc:
            logger.warning("Could not read %s: %s", fpath, exc)
            continue

        if not isinstance(data, dict):
            logger.warning("Skipping %s: top-level is not a dict", fpath)
            continue

        # Each JSON is produced by stage3_detmerge.run_stage3 as:
        #   {combo_id: payload}  (single cell-line file, filename = safe cell-line name)
        #
        # A payload dict has the key "combo" (list) and "best_theta" (dict).
        # We detect whether a value is a payload by checking for "combo" key.
        def _is_payload(v: Any) -> bool:
            return isinstance(v, dict) and "combo" in v and "best_theta" in v

        all_payloads = all(_is_payload(v) for v in data.values())

        if all_payloads:
            # Single-cell-line file: {combo_id: payload}
            cl_key = fpath.stem   # e.g. "NCIH2170" or "SNU16"
            existing = merged.get(cl_key, {})
            existing.update(data)
            merged[cl_key] = existing
        else:
            # Nested file: {cell_line: {combo_id: payload}}
            for cl, block in data.items():
                if isinstance(block, dict):
                    existing = merged.get(cl, {})
                    existing.update(block)
                    merged[cl] = existing

    return merged


def _pick_best_combo(cell_block: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return the single best payload from a cell block.

    Selection metric (Decision D2): ``best_val_f1``.
    Falls back to ``best_test_f1`` with a warning if ``best_val_f1`` is
    absent (e.g., legacy partial CSVs from before A4).
    """
    best_payload: Optional[Dict[str, Any]] = None
    best_f1 = -1.0

    # Determine which key to sort by
    has_val_f1 = any(
        "best_val_f1" in payload
        for payload in cell_block.values()
        if isinstance(payload, dict)
    )
    if not has_val_f1:
        logger.warning(
            "No 'best_val_f1' found in cell block — falling back to "
            "'best_test_f1' for selection. Re-run BO with the updated "
            "stage2/stage3 code to get val-based selection."
        )
    sort_key = "best_val_f1" if has_val_f1 else "best_test_f1"

    for combo_id, payload in cell_block.items():
        if not isinstance(payload, dict):
            continue
        f1 = float(payload.get(sort_key, -1.0))
        if f1 > best_f1:
            best_f1 = f1
            best_payload = {combo_id: payload}
    return best_payload


def freeze_best_params(
    stage3_json_dir: Path,
    output_path: Path,
    cell_lines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Read Stage-3 per-cell-line JSONs, pick the best combo, write frozen JSON.

    Parameters
    ----------
    stage3_json_dir:
        Directory containing per-cell-line Stage-3 JSON files.
    output_path:
        Destination path for the frozen JSON
        (``configs/classical/stage3_frozen_params.json``).
    cell_lines:
        Optional whitelist of cell-line names.  If None, all cell lines
        found in the JSON dir are included.

    Returns
    -------
    dict
        The frozen params dict that was written to *output_path*.
    """
    merged = load_and_merge_stage3_jsons(Path(stage3_json_dir))

    if cell_lines is not None:
        available_by_norm = {
            _norm_cell_line_name(cl): cl
            for cl in merged.keys()
        }

        resolved: Dict[str, Any] = {}
        missing: List[str] = []

        for requested_cl in cell_lines:
            norm = _norm_cell_line_name(requested_cl)
            available_cl = available_by_norm.get(norm)
            if available_cl is None:
                missing.append(requested_cl)
                continue

            # Preserve the requested/public cell-line name in the frozen JSON,
            # but read the payload from the safe filename-derived key.
            resolved[requested_cl] = merged[available_cl]

        if missing:
            raise KeyError(
                f"Requested cell lines not found in Stage-3 JSONs: {sorted(missing)}. "
                f"Available: {sorted(merged.keys())}"
            )

        merged = resolved

    frozen: Dict[str, Any] = {}
    for cl, cell_block in sorted(merged.items()):
        best = _pick_best_combo(cell_block)
        if best is None:
            logger.warning("No valid payload for cell line %s; skipping.", cl)
            continue
        frozen[cl] = best
        combo_id = next(iter(best))
        payload  = best[combo_id]
        val_f1   = float(payload.get("best_val_f1",  -1.0))
        test_f1  = float(payload.get("best_test_f1", -1.0))
        if val_f1 >= 0:
            logger.info("Frozen | %s | best_combo=%s | val_f1=%.4f | test_f1=%.4f",
                        cl, combo_id, val_f1, test_f1)
        else:
            logger.info("Frozen | %s | best_combo=%s | test_f1=%.4f (no val_f1)",
                        cl, combo_id, test_f1)

    if not frozen:
        raise RuntimeError("freeze_best_params produced an empty frozen dict — check Stage-3 outputs.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(output_path) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(frozen, f, indent=2)
    os.replace(tmp, str(output_path))
    logger.info("Wrote frozen params → %s (%d cell lines)", output_path, len(frozen))

    return frozen