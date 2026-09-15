#!/usr/bin/env python
"""
loco_paper_table.py — turn the leave-one-cell-line-out runs into paper numbers.

Run after collect_loco_results.py.

    python scripts/loco_paper_table.py --out-root outputs/eccount_loco

WHAT THIS FIXES
---------------
collect_loco_results.py answers "did the runs work". It does not produce
numbers that can go in a manuscript. Three things are wrong for that purpose:

1.  THE NCI-H2170 COMPARISON IS NOT LIKE-FOR-LIKE. The leave-one-out run is
    scored on all 888 NCI-H2170 images; the size-matched control is scored on
    the 134 test rows only, because the control saw some of that line's train
    and val images. Subtracting one from the other mixes cell-line novelty
    with a change of image set. This script rescores the leave-one-out run on
    the same 134 rows and reports that row separately.

2.  obj_f1_gap IS NOT THE COST OF NOT HAVING SEEN A LINE. It compares one
    checkpoint on the unseen line against the same checkpoint on the lines it
    trained on. That is a training-health check — useful, and it says nothing
    about how hard the held-out line is. NCI-H716's negative gap most likely
    means H716 is an easy line, not that the model prefers unseen data. The
    manuscript claim needs the paper model's F1 ON THAT SAME LINE, which lives
    in the locked per-cell-line CSV.

3.  count_mae IS NOT SCALE-FREE. COLO320DM averages ~46 objects per image and
    SNU16 ~283, so a raw MAE ranking is misleading. Relative MAE and the mean
    ground-truth count are both reported.

It also adds mean signed count bias, which collect_loco_results.py does not
compute at all. ecCount peaks being the only near-zero-bias model (+0.4) is a
headline claim of the paper; whether that survives on an unseen cell line is
directly relevant and nothing currently measures it.

AGGREGATION CONVENTION
----------------------
Pooled F1 (from summed TP/FP/FN) and mean-of-per-image F1 are different
numbers, and quoting the wrong one puts a value in the paper that does not
match any frozen CSV. Rather than assume, this script computes both, compares
them against summary_overall.csv on image sets it has not touched, and adopts
whichever the benchmark pipeline actually uses. If neither matches it says so
and refuses to write the table.

NOTHING IS INVENTED
-------------------
Every number written here is recomputed from per_image_metrics.csv or read
from a locked CSV. If the in-distribution CSV cannot be parsed with
confidence, that column is left empty and the file's structure is printed,
rather than a guess being written into a manuscript table.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("loco_table")

CELL_LINES = ("COLO320DM", "NCI-H2170", "NCI-H716", "SNU16")
SLUG = {c: c.lower().replace("-", "_") for c in CELL_LINES}

# (run_id suffix, run_kind label)
RUN_KINDS = (("", "leave-one-out"), ("_control", "size-matched control"))

# Expected held-out image counts, from the frozen consistency CSV.
EXPECTED_N_HELD_OUT = {
    "COLO320DM": 64,
    "NCI-H2170": 888,
    "NCI-H716": 70,
    "SNU16": 123,
}
# The control is scored on the held-out line's test rows only.
EXPECTED_N_CONTROL = {"NCI-H2170": 134}

TOL = 1e-6


# ---------------------------------------------------------------------------
# metric helpers
# ---------------------------------------------------------------------------

def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    """Object-level F1 from pooled TP/FP/FN."""
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    return 2 * p * r / (p + r) if (p + r) else 0.0


def normalise_model(name: object) -> str:
    """Map registry keys and display names onto one canonical label.

    'eccount_peaks', 'ecCount (peaks)'          -> ecCount (peaks)
    'eccount_mask', 'ecCount (threshold mask)'  -> ecCount (threshold mask)

    Order matters: the display name for the threshold model contains both
    'thresh' and 'mask', and the peaks name contains neither, so 'peak' is
    tested first.
    """
    s = str(name).lower()
    if "peak" in s:
        return "ecCount (peaks)"
    if "mask" in s or "thresh" in s:
        return "ecCount (threshold mask)"
    return str(name)


def block_metrics(sub: pd.DataFrame) -> Dict[str, float]:
    """Every metric this script reports, for one model on one image set."""
    tp = float(sub["obj_tp"].sum())
    fp = float(sub["obj_fp"].sum())
    fn = float(sub["obj_fn"].sum())

    mean_gt = float(sub["gt_count"].mean())
    mae = float(sub["count_abs_error"].mean())

    return {
        "n_images": int(len(sub)),
        "obj_f1_pooled": f1_from_counts(tp, fp, fn),
        "obj_f1_mean": float(sub["obj_f1"].mean()),
        "precision_pooled": tp / (tp + fp) if (tp + fp) else 0.0,
        "recall_pooled": tp / (tp + fn) if (tp + fn) else 0.0,
        "count_mae": mae,
        "mean_gt_count": mean_gt,
        "rel_mae": mae / mean_gt if mean_gt else float("nan"),
        "mean_signed_bias": float((sub["pred_count"] - sub["gt_count"]).mean()),
        "gt_objects": int(sub["gt_count"].sum()),
        "n_pred_missing": int((~sub["pred_exists"].astype(bool)).sum())
        if "pred_exists" in sub.columns else 0,
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def per_image_path(run_dir: Path, results_dir: str, policy: str) -> Optional[Path]:
    p = run_dir / results_dir / policy / "per_image_metrics.csv"
    return p if p.is_file() else None


def summary_path(run_dir: Path, results_dir: str, policy: str) -> Optional[Path]:
    p = run_dir / results_dir / policy / "summary_overall.csv"
    return p if p.is_file() else None


def read_summary(path: Path) -> Dict[str, float]:
    """summary_overall.csv as {canonical model name: obj_f1}."""
    df = pd.read_csv(path)
    model_col = next(
        (c for c in ("model", "model_name", "method") if c in df.columns), None
    )
    f1_col = next((c for c in ("obj_f1", "f1", "object_f1") if c in df.columns), None)
    if model_col is None or f1_col is None:
        logger.warning("Cannot read %s (columns: %s)", path, list(df.columns))
        return {}
    return {
        normalise_model(r[model_col]): float(r[f1_col]) for _, r in df.iterrows()
    }


# ---------------------------------------------------------------------------
# aggregation convention
# ---------------------------------------------------------------------------

def detect_convention(blocks: list) -> Tuple[str, float]:
    """Decide whether the pipeline pools F1 or averages per-image F1.

    ``blocks`` is a list of (label, computed_metrics, summary_f1) for image
    sets this script scored in full — i.e. sets where the recomputation should
    reproduce summary_overall.csv exactly.

    Returns (convention, worst_deviation). Exits if neither convention fits,
    because guessing here silently changes every number in the table.
    """
    dev_pooled, dev_mean = [], []
    for label, m, s in blocks:
        dev_pooled.append(abs(m["obj_f1_pooled"] - s))
        dev_mean.append(abs(m["obj_f1_mean"] - s))

    worst_pooled = max(dev_pooled) if dev_pooled else float("inf")
    worst_mean = max(dev_mean) if dev_mean else float("inf")

    logger.info(
        "Convention check over %d model/image-set pairs: "
        "worst deviation pooled=%.2e, mean-of-per-image=%.2e",
        len(blocks), worst_pooled, worst_mean,
    )

    if worst_pooled <= TOL and worst_mean > TOL:
        return "pooled", worst_pooled
    if worst_mean <= TOL and worst_pooled > TOL:
        return "mean", worst_mean
    if worst_pooled <= TOL and worst_mean <= TOL:
        # Both fit — only possible if they coincide. Pooled is the paper's
        # convention for count-weighted quantities, so prefer it.
        logger.info("Both conventions fit; using pooled.")
        return "pooled", worst_pooled

    logger.error(
        "Neither aggregation convention reproduces summary_overall.csv "
        "(pooled off by %.2e, mean off by %.2e). The benchmark aggregates "
        "obj_f1 some other way. Stopping rather than writing numbers that do "
        "not match the frozen CSVs — inspect "
        "ecdna_bench.benchmark.aggregate and re-run.",
        worst_pooled, worst_mean,
    )
    for label, m, s in blocks:
        logger.error(
            "  %-46s summary=%.6f pooled=%.6f mean=%.6f",
            label, s, m["obj_f1_pooled"], m["obj_f1_mean"],
        )
    sys.exit(3)


# ---------------------------------------------------------------------------
# in-distribution reference
# ---------------------------------------------------------------------------

def load_in_distribution(path: Path) -> Optional[pd.DataFrame]:
    """Read the locked per-cell-line ecCount values, or return None.

    The schema of this file is not assumed. Columns are identified by their
    contents: the cell-line column is the one whose values match the four
    benchmark cell lines, the model column is the one naming ecCount variants.
    If either cannot be identified the function reports the structure and
    returns None, so the caller leaves the column empty instead of writing a
    guess into a manuscript table.
    """
    if not path.is_file():
        logger.warning("In-distribution CSV not found: %s", path)
        return None

    df = pd.read_csv(path)
    logger.info("In-distribution CSV columns: %s", list(df.columns))

    line_col = None
    for c in df.columns:
        vals = set(str(v) for v in df[c].dropna().unique())
        if len(vals & set(CELL_LINES)) >= 3:
            line_col = c
            break

    model_col = None
    for c in df.columns:
        if c == line_col:
            continue
        vals = " ".join(str(v).lower() for v in df[c].dropna().unique()[:50])
        if "eccount" in vals or "peak" in vals:
            model_col = c
            break

    # Value column: an explicit metric/value pair, or a named f1 column.
    value_col, metric_col = None, None
    for c in df.columns:
        if str(c).lower() in ("metric", "measure", "statistic"):
            metric_col = c
    if metric_col is not None:
        value_col = next(
            (c for c in df.columns if str(c).lower() in ("value", "val", "score")),
            None,
        )
    else:
        value_col = next(
            (c for c in df.columns
             if str(c).lower() in ("obj_f1", "f1", "object_f1", "value")),
            None,
        )

    if line_col is None or model_col is None or value_col is None:
        logger.warning(
            "Could not identify columns in %s "
            "(cell line=%s, model=%s, value=%s). Leaving the in-distribution "
            "column empty. First rows:\n%s",
            path.name, line_col, model_col, value_col, df.head(8).to_string(),
        )
        return None

    out = df.copy()
    if metric_col is not None:
        mask = out[metric_col].astype(str).str.lower().str.contains("f1")
        if not mask.any():
            logger.warning(
                "No F1 rows in %s under column %r. Leaving the column empty.",
                path.name, metric_col,
            )
            return None
        out = out[mask]

    out = out[[line_col, model_col, value_col]].copy()
    out.columns = ["cell_line", "model_raw", "in_distribution_obj_f1"]
    out["model"] = out["model_raw"].map(normalise_model)
    out = out[out["model"].isin(
        ["ecCount (peaks)", "ecCount (threshold mask)"])]

    if out.empty:
        logger.warning(
            "No ecCount rows found in %s after normalisation. Model values "
            "present: %s", path.name,
            sorted(set(str(v) for v in df[model_col].dropna().unique()))[:12],
        )
        return None

    logger.info(
        "In-distribution values matched: %d rows "
        "(columns used — cell line: %r, model: %r, value: %r)",
        len(out), line_col, model_col, value_col,
    )
    return out[["cell_line", "model", "in_distribution_obj_f1"]]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=__doc__.split("WHAT THIS FIXES")[0].strip(),
    )
    ap.add_argument("--out-root", default="outputs/eccount_loco")
    ap.add_argument("--policy", default="or_matching",
                    choices=["or_matching", "and_matching"])
    ap.add_argument(
        "--in-distribution-csv",
        default="release/figures/notebook05/"
                "source_fig6_f1_by_cell_line_heatmap_long.csv",
        help="Locked per-cell-line ecCount values.",
    )
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(levelname)-8s %(message)s", force=True)

    root = Path(args.out_root)
    if not root.is_dir():
        logger.error("Not found: %s", root)
        return 1

    # -- load every run's per-image metrics -------------------------------
    # loaded[(cell_line, kind)] = {"held": df or None, "ref": df or None,
    #                              "summary_held": {...}}
    loaded: Dict[Tuple[str, str], dict] = {}

    for line in CELL_LINES:
        for suffix, kind in RUN_KINDS:
            run_dir = root / f"holdout_{SLUG[line]}{suffix}"
            if not run_dir.is_dir():
                continue
            held_p = per_image_path(run_dir, "results", args.policy)
            ref_p = per_image_path(run_dir, "results_seen_lines", args.policy)
            summ_p = summary_path(run_dir, "results", args.policy)
            if held_p is None:
                logger.warning("%s: no per-image metrics under results/%s",
                               run_dir.name, args.policy)
                continue
            loaded[(line, kind)] = {
                "run_dir": run_dir,
                "held": pd.read_csv(held_p),
                "ref": pd.read_csv(ref_p) if ref_p else None,
                "summary_held": read_summary(summ_p) if summ_p else {},
            }
            logger.info("%s: %d held-out rows, %s reference rows",
                        run_dir.name, len(loaded[(line, kind)]["held"]),
                        len(loaded[(line, kind)]["ref"])
                        if ref_p else "no")

    if not loaded:
        logger.error("No runs found under %s.", root)
        return 2

    # -- guard: did each run score the images it was meant to? ------------
    problems = []
    for (line, kind), d in loaded.items():
        n = d["held"]["uid"].nunique()
        expected = (EXPECTED_N_HELD_OUT.get(line) if kind == "leave-one-out"
                    else EXPECTED_N_CONTROL.get(line))
        if expected is not None and n != expected:
            problems.append(
                f"{line} {kind}: scored {n} unique images, expected {expected}"
            )
    for p in problems:
        logger.error("GUARD: %s", p)
    if problems:
        logger.error("Refusing to build a table from runs that scored the "
                     "wrong image sets.")
        return 4

    # -- work out how the pipeline aggregates obj_f1 ----------------------
    check_blocks = []
    for (line, kind), d in loaded.items():
        for model, s_f1 in d["summary_held"].items():
            sub = d["held"][d["held"]["model"].map(normalise_model) == model]
            if len(sub):
                check_blocks.append(
                    (f"{line} / {kind} / {model}", block_metrics(sub), s_f1)
                )
    convention, worst = detect_convention(check_blocks)
    f1_key = "obj_f1_pooled" if convention == "pooled" else "obj_f1_mean"
    logger.info("Using %s obj_f1 (reproduces summary_overall.csv to %.2e).",
                convention, worst)

    # -- build the rows ---------------------------------------------------
    rows = []

    def add_row(line, kind, scope, model, held_sub, ref_sub):
        m = block_metrics(held_sub)
        rec = {
            "held_out_cell_line": line,
            "run_kind": kind,
            "scope": scope,
            "model": model,
            "n_images": m["n_images"],
            "gt_objects": m["gt_objects"],
            "mean_gt_count": m["mean_gt_count"],
            "obj_f1": m[f1_key],
            "precision": m["precision_pooled"],
            "recall": m["recall_pooled"],
            "count_mae": m["count_mae"],
            "rel_mae": m["rel_mae"],
            "mean_signed_bias": m["mean_signed_bias"],
            "n_pred_missing": m["n_pred_missing"],
        }
        if ref_sub is not None and len(ref_sub):
            r = block_metrics(ref_sub)
            rec["reference_n_images"] = r["n_images"]
            rec["reference_obj_f1"] = r[f1_key]
            rec["reference_count_mae"] = r["count_mae"]
            rec["reference_mean_signed_bias"] = r["mean_signed_bias"]
        return rows.append(rec)

    for (line, kind), d in loaded.items():
        held, ref = d["held"], d["ref"]
        for model in ("ecCount (peaks)", "ecCount (threshold mask)"):
            hs = held[held["model"].map(normalise_model) == model]
            rs = (ref[ref["model"].map(normalise_model) == model]
                  if ref is not None else None)
            if len(hs):
                scope = ("held-out line, all rows" if kind == "leave-one-out"
                         else "held-out line, test rows")
                add_row(line, kind, scope, model, hs, rs)

    # -- the like-for-like NCI-H2170 comparison ---------------------------
    loco = loaded.get(("NCI-H2170", "leave-one-out"))
    ctrl = loaded.get(("NCI-H2170", "size-matched control"))
    if loco is not None and ctrl is not None:
        loco_test = loco["held"][loco["held"]["split"] == "test"]
        ctrl_uids = set(ctrl["held"]["uid"])
        loco_uids = set(loco_test["uid"])
        if loco_uids != ctrl_uids:
            logger.warning(
                "Test-row UID sets differ: %d in the leave-one-out run, %d in "
                "the control, %d shared. Using the intersection.",
                len(loco_uids), len(ctrl_uids), len(loco_uids & ctrl_uids),
            )
            shared = loco_uids & ctrl_uids
            loco_test = loco_test[loco_test["uid"].isin(shared)]
        else:
            logger.info(
                "Like-for-like check: both NCI-H2170 runs cover the same %d "
                "test images.", len(loco_uids),
            )
        for model in ("ecCount (peaks)", "ecCount (threshold mask)"):
            hs = loco_test[loco_test["model"].map(normalise_model) == model]
            rs = (loco["ref"][loco["ref"]["model"].map(normalise_model) == model]
                  if loco["ref"] is not None else None)
            if len(hs):
                add_row("NCI-H2170", "leave-one-out (test-restricted)",
                        "held-out line, test rows", model, hs, rs)
    else:
        logger.warning("Both NCI-H2170 runs are needed for the like-for-like "
                       "comparison; one is missing.")

    df = pd.DataFrame(rows)

    # -- join the locked in-distribution values ---------------------------
    ind = load_in_distribution(Path(args.in_distribution_csv))
    if ind is not None:
        df = df.merge(
            ind.rename(columns={"cell_line": "held_out_cell_line"}),
            on=["held_out_cell_line", "model"], how="left",
        )
        df["cost_of_novelty"] = df["in_distribution_obj_f1"] - df["obj_f1"]
    else:
        df["in_distribution_obj_f1"] = np.nan
        df["cost_of_novelty"] = np.nan

    order = {"leave-one-out": 0, "leave-one-out (test-restricted)": 1,
             "size-matched control": 2}
    df["_k"] = df["run_kind"].map(order).fillna(9)
    df = (df.sort_values(["held_out_cell_line", "model", "_k"], kind="stable")
            .drop(columns="_k")
            .reset_index(drop=True))

    out_csv = root / f"source_loco_paper_table_{args.policy}.csv"
    df.to_csv(out_csv, index=False)

    # -- print, formatted to the manuscript's conventions -----------------
    show = df.copy()
    for c in ("obj_f1", "precision", "recall", "reference_obj_f1",
              "in_distribution_obj_f1"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    for c in ("count_mae", "reference_count_mae", "mean_gt_count"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
    for c in ("mean_signed_bias", "reference_mean_signed_bias"):
        if c in show.columns:
            show[c] = show[c].map(lambda v: "" if pd.isna(v) else f"{v:+.1f}")
    for c in ("rel_mae", "cost_of_novelty"):
        if c in show.columns:
            show[c] = show[c].map(
                lambda v: "" if pd.isna(v)
                else (f"{v:.1%}" if c == "rel_mae" else f"{v:+.3f}"))

    cols = [c for c in (
        "held_out_cell_line", "run_kind", "model", "n_images",
        "mean_gt_count", "obj_f1", "in_distribution_obj_f1",
        "cost_of_novelty", "count_mae", "rel_mae", "mean_signed_bias",
        "reference_obj_f1",
    ) if c in show.columns]

    print()
    print("=" * 100)
    print(f"  LOCO paper table — {args.policy}, obj_f1 aggregated {convention}")
    print("=" * 100)
    print(show[cols].to_string(index=False))
    print()
    if "n_pred_missing" in df.columns and df["n_pred_missing"].sum():
        print(f"NOTE: {int(df['n_pred_missing'].sum())} image/model pairs had "
              f"no prediction mask. Investigate before quoting.")
        print()
    if ind is None:
        print("in_distribution_obj_f1 is empty — the locked per-cell-line CSV")
        print("could not be parsed. Its structure is logged above. Do not fill")
        print("this column from CLAUDE_INSTRUCTIONS.md §3: those four")
        print("per-cell-line values (0.828 / 0.740 / 0.908 / 0.839) are")
        print("Classic (after opt), not ecCount.")
        print()
    print("Read the NCI-H2170 cell-line-shift effect from the")
    print("'leave-one-out (test-restricted)' row against the")
    print("'size-matched control' row. Both cover the same 134 test images.")
    print()
    print(f"Written: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
