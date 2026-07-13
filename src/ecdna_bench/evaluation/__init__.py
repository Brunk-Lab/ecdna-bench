"""
ecdna_bench.evaluation — the unified evaluation framework.

This sub-package is the core contribution of the paper. It harmonizes
predictions from any model (classical, Label Engine, MIA, ecSeg, ecCount)
and evaluates them through a single Hungarian-matching-based object
framework, plus pixel and count metric families.

Layout
------
- `objects`      : connected-component extraction and the standard object
                   schema used everywhere downstream.
- `matching`     : Hungarian assignment under OR or AND validity policies,
                   with a first-class `ignored` bucket for predictions that
                   had a valid candidate but lost the 1-to-1 assignment.
- `metrics`      : object / pixel / count metric families. Pure functions.
- `sensitivity`  : (d_max × IoU_min × policy) grid sweep powered by cached
                   pairwise tensors so repeated grid points are cheap.
- `density`      : density-bin assignment and stratified aggregation.
- `stats`        : Wilcoxon signed-rank, Friedman, Benjamini–Hochberg FDR.

Design invariants
-----------------
1. Every module here is importable without torch, matplotlib, or any other
   heavy optional dependency. The evaluation framework is meant to be usable
   by anyone wanting to benchmark their own model on the released data,
   regardless of what their model is written in.
2. No module here writes to disk. The benchmark orchestrator in
   `ecdna_bench.benchmark` handles that.
3. Object dicts follow the schema laid out in `objects.Object`, which is
   compatible with every downstream consumer in the codebase.
"""

from __future__ import annotations

from ecdna_bench.evaluation.density import (
    DEFAULT_DENSITY_EDGES,
    DEFAULT_DENSITY_LABELS,
    assign_density_bin,
    bin_counts,
    stratify_by_density,
)
from ecdna_bench.evaluation.matching import (
    BIG_COST,
    MatchResult,
    PairwiseTensors,
    match_boxes_to_centers,
    match_objects,
    precompute_pairwise,
    resolve_matching_from_pairwise,
)
from ecdna_bench.evaluation.metrics import (
    count_metrics,
    object_metrics,
    object_metrics_from_counts,
    pixel_confusion,
    pixel_metrics,
)
from ecdna_bench.evaluation.objects import (
    bbox_iou,
    centroid_distance,
    objects_from_mask,
)
from ecdna_bench.evaluation.sensitivity import (
    SweepRow,
    sweep_matching_grid,
    sweep_one_image,
)
from ecdna_bench.evaluation.stats import (
    benjamini_hochberg,
    friedman_test,
    wilcoxon_signed_rank,
)

__all__ = [
    # objects
    "objects_from_mask",
    "bbox_iou",
    "centroid_distance",
    # matching
    "BIG_COST",
    "MatchResult",
    "PairwiseTensors",
    "match_objects",
    "match_boxes_to_centers",
    "precompute_pairwise",
    "resolve_matching_from_pairwise",
    # metrics
    "object_metrics",
    "object_metrics_from_counts",
    "pixel_confusion",
    "pixel_metrics",
    "count_metrics",
    # sensitivity
    "SweepRow",
    "sweep_matching_grid",
    "sweep_one_image",
    # density
    "DEFAULT_DENSITY_EDGES",
    "DEFAULT_DENSITY_LABELS",
    "assign_density_bin",
    "bin_counts",
    "stratify_by_density",
    # stats
    "wilcoxon_signed_rank",
    "friedman_test",
    "benjamini_hochberg",
]
