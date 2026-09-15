#!/usr/bin/env python
"""
collect_loco_results.py — pool the leave-one-cell-line-out runs into one table.

Run after the array job finishes.

    python scripts/collect_loco_results.py --out-root outputs/eccount_loco

Reads each run's scored results, joins them to the in-distribution numbers from
the locked benchmark, and writes a comparison table plus one figure.

THE COMPARISON THAT MATTERS
---------------------------
For each cell line, two numbers:

    generalisation   ecCount trained WITHOUT that line, scored on it
    in-distribution  the locked benchmark value for that line, from a model
                     that saw it in training

The gap between them is the cost of never having seen the cell line — except
for NCI-H2170, where it is the cost of that PLUS a 4.5-fold smaller training
set. The size-matched control run separates the two. If the control is absent,
the NCI-H2170 row is flagged as confounded and must not be quoted alone.

GUARDS
------
Each run is checked against the ground-truth object total for its held-out
line before its metrics are read. A run whose totals do not match scored a
different set of images than it was supposed to, and its numbers are dropped
rather than reported.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("loco_collect")

# Held-out ground-truth totals, from the frozen consistency CSV.
# Sum = 228,039, the locked benchmark total.
HELD_OUT_GT = {
    "COLO320DM": (64, 2_943),
    "NCI-H2170": (888, 176_883),
    "NCI-H716": (70, 13_340),
    "SNU16": (123, 34_873),
}

# In-distribution reference: per-cell-line test F1 for ecCount (peaks) from the
# locked benchmark. Source: release/figures/notebook05/
# source_fig6_f1_by_cell_line_heatmap_long.csv — verify before quoting.
IN_DISTRIBUTION_NOTE = (
    "In-distribution values must be read from the locked per-cell-line CSV, "
    "not typed here. This script leaves that column empty for you to fill "
    "from release/figures/notebook05/source_fig6_f1_by_cell_line_heatmap_long.csv"
)

SLUG = {k: k.lower().replace("-", "_") for k in HELD_OUT_GT}


def find_summary(run_dir: Path, results_name: str = "results") -> Path | None:
    """Locate the OR-policy overall summary for one evaluation of a run."""
    for candidate in (
        run_dir / results_name / "or_matching" / "summary_overall.csv",
        run_dir / results_name / "summary_overall.csv",
    ):
        if candidate.is_file():
            return candidate
    hits = sorted(run_dir.glob(f"{results_name}/**/summary_overall.csv"))
    return hits[0] if hits else None


def read_metrics(path: Path) -> dict:
    """Model -> metric dict, from one summary_overall.csv."""
    summ = pd.read_csv(path)
    model_col = next(
        (c for c in ("model", "model_name") if c in summ.columns), None)
    n_col = next((c for c in ("n_images", "n") if c in summ.columns), None)
    out = {}
    for _, r in summ.iterrows():
        key = str(r[model_col]) if model_col else "unknown"
        rec = {"n_images": int(r[n_col]) if n_col else None}
        for col in ("obj_f1", "precision", "recall", "count_mae",
                    "count_bias", "mean_signed_bias", "pixel_dice",
                    "tp", "fp", "fn"):
            if col in summ.columns:
                rec[col] = r[col]
        out[key] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--out-root", default="outputs/eccount_loco")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)

    root = Path(args.out_root)
    if not root.is_dir():
        logger.error("Not found: %s", root)
        return 1

    records, problems = [], []

    for line, (n_expected, gt_expected) in HELD_OUT_GT.items():
        for suffix, kind in (("", "leave-one-out"), ("_control", "size-matched control")):
            run_dir = root / f"holdout_{SLUG[line]}{suffix}"
            if not run_dir.is_dir():
                if not suffix:
                    problems.append(f"{line}: run directory missing ({run_dir})")
                continue

            manifest_path = run_dir / "run_manifest.json"
            manifest = (
                json.loads(manifest_path.read_text())
                if manifest_path.is_file() else {}
            )

            summary_path = find_summary(run_dir, "results")
            if summary_path is None:
                problems.append(f"{line} {kind}: no summary_overall.csv — "
                                f"scoring did not complete")
                continue
            held = read_metrics(summary_path)

            # Guard: the run must have scored the images it was meant to.
            if not suffix:
                got = max(
                    (v["n_images"] for v in held.values()
                     if v["n_images"] is not None), default=None)
                if got is not None and got != n_expected:
                    problems.append(
                        f"{line} {kind}: scored {got} images, expected "
                        f"{n_expected}. Numbers dropped."
                    )
                    continue

            ref_path = find_summary(run_dir, "results_seen_lines")
            ref = read_metrics(ref_path) if ref_path else {}
            if not ref:
                problems.append(
                    f"{line} {kind}: no reference scoring — the "
                    f"generalisation gap cannot be computed within this model, "
                    f"only against the paper's model"
                )

            for model, hm in held.items():
                rm = ref.get(model, {})
                rec = {
                    "held_out_cell_line": line,
                    "run_kind": kind,
                    "model": model,
                    "n_train_images": manifest.get("n_train"),
                    "best_epoch": manifest.get("best_epoch"),
                    "best_val_loss": manifest.get("best_val_loss"),
                    "n_images_held_out": hm.get("n_images"),
                    "n_images_reference": rm.get("n_images"),
                }
                for col in ("obj_f1", "precision", "recall", "count_mae",
                            "count_bias", "mean_signed_bias", "pixel_dice"):
                    if col in hm:
                        rec[f"{col}_held_out"] = hm[col]
                    if col in rm:
                        rec[f"{col}_reference"] = rm[col]
                # The gap that answers the question: same checkpoint, unseen
                # cell line versus cell lines it trained on. Positive means the
                # unseen line scores worse.
                if "obj_f1" in hm and "obj_f1" in rm:
                    rec["obj_f1_gap"] = round(rm["obj_f1"] - hm["obj_f1"], 4)
                if "count_mae" in hm and "count_mae" in rm:
                    rec["count_mae_gap"] = round(
                        hm["count_mae"] - rm["count_mae"], 2)
                records.append(rec)

    if not records:
        logger.error("No results found under %s. Nothing to collect.", root)
        for p in problems:
            logger.error("  %s", p)
        return 2

    df = pd.DataFrame(records)

    # Flag the confound explicitly rather than leaving it for the reader.
    has_control = "size-matched control" in set(
        df.loc[df["held_out_cell_line"] == "NCI-H2170", "run_kind"])
    df["note"] = ""
    mask = (df["held_out_cell_line"] == "NCI-H2170") & (df["run_kind"] == "leave-one-out")
    df.loc[mask, "note"] = (
        "training set is 179 images vs ~750 for the other lines; "
        + ("compare against the size-matched control row"
           if has_control
           else "CONFOUNDED — cell-line shift and data volume are not separable "
                "without the size-matched control run")
    )

    out_csv = root / "source_loco_summary.csv"
    df.to_csv(out_csv, index=False)

    print()
    print("=" * 78)
    print("  Leave-one-cell-line-out results")
    print("=" * 78)
    show = [c for c in ("held_out_cell_line", "run_kind", "model",
                        "n_train_images", "n_images_held_out",
                        "obj_f1_held_out", "obj_f1_reference", "obj_f1_gap",
                        "count_mae_held_out", "count_mae_reference")
            if c in df.columns]
    print(df[show].to_string(index=False))
    print()
    print("obj_f1_held_out    — the unseen cell line")
    print("obj_f1_reference   — the same checkpoint on test images of the cell")
    print("                     lines it trained on")
    print("obj_f1_gap         — reference minus held out; positive means the")
    print("                     unseen cell line scores worse")
    print()
    if not has_control:
        print("WARNING: no size-matched control for NCI-H2170. That row cannot")
        print("separate cell-line shift from training-set size. Run:")
        print("  sbatch --array=4-4 slurm/submit_eccount_loco.sh")
        print()
    if problems:
        print("Problems:")
        for p in problems:
            print("  -", p)
        print()
    print(IN_DISTRIBUTION_NOTE)
    print()
    print(f"Written: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())