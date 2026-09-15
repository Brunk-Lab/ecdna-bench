#!/usr/bin/env python
"""Pooled, benchmark-only summary of the River-ROI run, gt_full variant.

The printed per-cell-line table from benchmark_river_roi.py pools benchmark and
extension images together, which is why COLO320DM reads 0.554 there against a
benchmark value of 0.915. This separates them and pools F1 from TP/FP/FN rather
than averaging per-image F1, matching the paper's convention.
"""
from pathlib import Path

import pandas as pd

RUN = Path("/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/release/river_roi_run")
PIM = RUN / "benchmark" / "per_image_metrics_river.csv"
COLS = ["uid", "obj_tp", "obj_fp", "obj_fn", "pred_count", "gt_count"]
EXT = "extension (not in benchmark)"

df = pd.read_csv(PIM)
print(f"read {PIM}\nshape {df.shape}\n")

# The subset labels must be the repaired ones, not the all-benchmark labels.
n = df[df.model == "ecCount (peaks)"].drop_duplicates("uid").subset.value_counts()
print("subset labels (unique images)")
print(n.to_string())
assert n.get("benchmark train (SEEN by ROI model)") == 800, "subset labels are stale"
assert n.get("benchmark val+test (held out)") == 345, "subset labels are stale"
print("subset labels OK\n")


def pooled(g):
    tp, fp, fn = g.obj_tp.sum(), g.obj_fp.sum(), g.obj_fn.sum()
    p = tp / (tp + fp) if tp + fp else float("nan")
    r = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * p * r / (p + r) if p + r else float("nan")
    d = g.pred_count - g.gt_count
    return pd.Series({
        "n": g.uid.nunique(), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(p, 3), "recall": round(r, 3), "object_f1": round(f1, 3),
        "count_mae": round(d.abs().mean(), 1), "count_bias": round(d.mean(), 1),
    })


for model in ["ecCount (peaks)", "ecCount (threshold mask)"]:
    d = df[(df.model == model) & (df.gt_variant == "gt_full")]
    print("=" * 78)
    print(f"{model}  ·  gt_full  (end-to-end, not conditional on the ROI)")
    print("=" * 78)
    print("\nby subset, pooled")
    print(d.groupby("subset")[COLS].apply(pooled).to_string())
    print("\nby cell line x subset, pooled")
    print(d.groupby(["cell_line", "subset"])[COLS].apply(pooled).to_string())
    bench = d[d.subset != EXT]
    print("\nthe 1,145 benchmark images only, pooled  <-- compare this to the paper")
    print(pooled(bench).to_string())
    print()
