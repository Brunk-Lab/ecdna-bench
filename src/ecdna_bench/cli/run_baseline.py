"""
ecdna_bench.cli.run_baseline
==============================
Harmonize one external baseline model's prediction masks.

Usage
-----
    python -m ecdna_bench.cli.run_baseline --model ecseg --config configs/default.yaml
    python -m ecdna_bench.cli.run_baseline --model mia
    python -m ecdna_bench.cli.run_baseline --model label_engine

Required config keys (under ``paths``)
---------------------------------------
* ``consistency_csv``
* ``{model}_raw_masks``        — input: raw prediction mask dir
* ``{model}_masks``            — output: harmonized binary mask dir
  (e.g. ``ecseg_raw_masks``, ``ecseg_masks``)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from ecdna_bench.cli._common import (
    get_path, load_config, require_file, resolve_path, setup_logging,
)

logger = logging.getLogger(__name__)

_HARMONIZERS = {
    "ecseg":         "ecdna_bench.baselines.ecseg:harmonize_ecseg",
    "mia":           "ecdna_bench.baselines.mia:harmonize_mia",
    "label_engine":  "ecdna_bench.baselines.label_engine:harmonize_label_engine",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Harmonize an external baseline's prediction masks.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, choices=list(_HARMONIZERS),
                   help="Which baseline to harmonize.")
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--min-area", type=int, default=3)
    p.add_argument("--force", action="store_true")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    model = args.model

    consistency_csv = get_path(cfg, "consistency_csv")
    raw_dir         = get_path(cfg, f"{model}_raw_masks")
    out_dir         = get_path(cfg, f"{model}_masks", required=False) or \
                      resolve_path(f"release/harmonized_masks/{model}")
    require_file(consistency_csv, "consistency CSV")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load uid list
    import pandas as pd
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]
    uid_list = df["unique_id"].astype(str).tolist()

    logger.info("Harmonizing %s: %d UIDs", model, len(uid_list))
    logger.info("  raw_dir = %s", raw_dir)
    logger.info("  out_dir = %s", out_dir)

    # Import harmonizer
    mod_name, fn_name = _HARMONIZERS[model].split(":")
    import importlib
    mod = importlib.import_module(mod_name)
    harmonize_fn = getattr(mod, fn_name)

    # Check if already done
    if not args.force:
        existing = list(out_dir.glob("*.png"))
        if len(existing) >= len(uid_list):
            logger.info("Already %d masks in %s — skipping (use --force to redo).",
                        len(existing), out_dir)
            return

    results = harmonize_fn(raw_dir, out_dir, uid_list, min_area=args.min_area)

    ok      = sum(1 for v in results.values() if v == "ok")
    missing = sum(1 for v in results.values() if v == "missing")
    errors  = sum(1 for v in results.values() if v.startswith("error"))
    logger.info("Harmonization complete — ok=%d missing=%d errors=%d", ok, missing, errors)


if __name__ == "__main__":
    main()
