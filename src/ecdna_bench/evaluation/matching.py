"""
ecdna_bench.evaluation.matching — Hungarian 1-to-1 assignment between
predicted and gold-standard objects, with first-class OR / AND validity
policies and a dedicated "ignored prediction" bucket.

What this module does
---------------------
Given two lists of object dicts (see :mod:`ecdna_bench.evaluation.objects`
for the schema), decide which predictions correspond to which gold-standard objects,
and bucket every un-paired prediction as either a true false positive (``fp``)
or an *ignored* duplicate (``ignored_pred``). The latter is a prediction that
*had* a valid geometric candidate but was passed over by the 1-to-1 Hungarian
assignment because a better-scoring candidate won; we explicitly refuse to
count such predictions as false positives because in ecDNA microscopy they
almost always represent over-segmentation of a single real spot into two or
more close neighbours, not genuine spurious detections.

The OR / AND policy is the other key contribution of this module. Any
candidate pair (pred i, gt j) has two binary gates:

  * ``dist_ok[i,j]``  — the centroid distance is at most ``d_max`` pixels.
  * ``overlap_ok[i,j]`` — the prediction bounding box "hits" the GS mask
                          *and* their true IoU is at least ``min_iou``.

Under the **OR** policy (the paper's main-text convention), a pair is a
valid candidate if either gate passes. Under the **AND** policy, both must
pass. Valid pairs are scored by

    C_ij = alpha * (1 - overlap_score_ij) + (1 - alpha) * d_ij / d_max

and the Hungarian algorithm picks the minimum-cost 1-to-1 assignment.
Invalid pairs receive a sentinel ``BIG_COST`` so the Hungarian solver
never selects them.

Why a two-step API
------------------
Sensitivity analysis and benchmark runs re-evaluate the same image under
dozens of ``(d_max, min_iou, policy)`` combinations. Re-scanning pixels to
rebuild the geometry from scratch for each combination would be wasteful:
the geometry is invariant to matching parameters. We therefore expose the
pipeline in two pieces:

  1. :func:`precompute_pairwise`     (slow; one call per image)
         → ``PairwiseTensors`` holding ``dist``, ``overlap``, and ``hit`` for
           every (pred, gt) pair.
  2. :func:`resolve_matching_from_pairwise`  (fast; one call per param set)
         → ``MatchResult`` under a specific ``(d_max, min_iou, alpha, policy)``.

For a single evaluation, :func:`match_objects` is a convenience wrapper that
calls both in sequence.

Output schema
-------------
Every matching call returns a :class:`MatchResult` with four lists that
partition the predictions and the gold-standard objects:

  * ``matched``        : list of ``(pred_idx, gt_idx)`` tuples (the TPs).
  * ``unmatched_pred`` : predictions with *no* valid candidate at all → FP.
  * ``ignored_pred``   : predictions that had a candidate but lost → ignored.
  * ``unmatched_gt``   : GS objects that nobody matched to → FN.

``len(matched) + len(unmatched_pred) + len(ignored_pred)`` always equals the
total number of predictions; ``len(matched) + len(unmatched_gt)`` always
equals the total number of GS objects. These identities are enforced by the tests
in ``tests/test_matching.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from ecdna_bench.evaluation.objects import bbox_iou, centroid_distance


__all__ = [
    "BIG_COST",
    "MatchPolicy",
    "MatchResult",
    "PairwiseTensors",
    "match_objects",
    "match_boxes_to_centers",
    "precompute_pairwise",
    "resolve_matching_from_pairwise",
]


# Sentinel cost that Hungarian must never pick. Any realistic cost from the
# formula above is bounded above by 1.0, so 1e6 is safely "infinite" while
# still being finite (``linear_sum_assignment`` rejects true ``inf``).
BIG_COST: float = 1.0e6

# Type alias used in signatures.
MatchPolicy = Literal["OR", "AND"]


# ==============================================================================
# Result containers.
# ==============================================================================


@dataclass(frozen=True)
class MatchResult:
    """
    Immutable record of a single matching call.

    Attributes
    ----------
    matched : list[tuple[int, int]]
        Each tuple is ``(pred_idx, gt_idx)``; these are the TPs.
    unmatched_pred : list[int]
        Predictions with no valid candidate at all → counted as FP.
    ignored_pred : list[int]
        Predictions that had at least one valid candidate but were not
        selected by the 1-to-1 Hungarian assignment → ignored in
        precision/recall/F1. This is the paper's "ignored" bucket.
    unmatched_gt : list[int]
        GS objects that nobody matched to → counted as FN.
    n_pred : int
        Total number of predictions supplied to the matcher.
    n_gt : int
        Total number of gold-standard objects supplied to the matcher.
    """

    matched: list[tuple[int, int]] = field(default_factory=list)
    unmatched_pred: list[int] = field(default_factory=list)
    ignored_pred: list[int] = field(default_factory=list)
    unmatched_gt: list[int] = field(default_factory=list)
    n_pred: int = 0
    n_gt: int = 0

    # Convenience counters. We expose them as properties rather than caching
    # at construction time so that a MatchResult remains trivially picklable
    # and comparable.
    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)

    @property
    def ignored(self) -> int:
        return len(self.ignored_pred)

    @property
    def ignored_count(self) -> int:
        """Alias for .ignored — used by the test suite."""
        return len(self.ignored_pred)


@dataclass(frozen=True)
class PairwiseTensors:
    """
    Matching-parameter-independent geometry between predictions and GS objects.

    This is the output of :func:`precompute_pairwise` and the input of
    :func:`resolve_matching_from_pairwise`. Reusing the same tensors across
    many ``(d_max, min_iou, alpha, policy)`` combinations is what makes
    the sensitivity sweep in :mod:`ecdna_bench.evaluation.sensitivity`
    affordable.

    Attributes
    ----------
    dist : np.ndarray, shape ``(n_pred, n_gt)``
        Euclidean centroid distance in pixels. Entries outside the
        ``max_precompute_dist`` cap used during precompute are set to
        ``np.inf`` — the sensitivity sweep must not request a ``d_max``
        larger than that cap, or it will under-report distant candidates.
    overlap : np.ndarray, shape ``(n_pred, n_gt)``
        Overlap score in ``[0, 1]``. The exact definition follows the
        Supplementary §13 recipe and is documented in :func:`precompute_pairwise`.
    hit : np.ndarray, shape ``(n_pred, n_gt)``, dtype bool
        True iff the prediction bbox contains at least one GS-positive
        pixel (or, when no GS mask is available, iff bbox IoU > 0). This
        is the "hit gate" used together with ``min_iou`` to decide
        ``overlap_ok`` downstream.
    n_pred : int
    n_gt : int
    max_precompute_dist : float
        The distance cap used when this precompute was built. Any
        ``d_max`` greater than this is inconsistent with the precompute
        and will be rejected by :func:`resolve_matching_from_pairwise`.
    """

    dist: np.ndarray
    overlap: np.ndarray
    hit: np.ndarray
    n_pred: int
    n_gt: int
    max_precompute_dist: float


# ==============================================================================
# Pairwise precompute — slow path, one call per image.
# ==============================================================================


def precompute_pairwise(
    pred_objs: Sequence[dict[str, Any]],
    gt_objs: Sequence[dict[str, Any]],
    *,
    max_precompute_dist: float,
) -> PairwiseTensors:
    """
    Compute all matching-parameter-*independent* pairwise geometry between
    predictions and GS objects in one pass over the data.

    The work here is dominated by the small-bbox ROI extraction for IoU,
    which is the single most expensive operation in the evaluation. Doing
    it once per image rather than once per matching-parameter combination
    is the critical optimization that makes the sensitivity sweep cheap.

    Parameters
    ----------
    pred_objs, gt_objs : sequence of dict
        Standard object dicts. See :mod:`ecdna_bench.evaluation.objects`.
    max_precompute_dist : float
        Distance cap used to short-circuit the inner loop. Pairs with
        centroid distance greater than this are recorded as ``np.inf``
        in ``dist`` and never contribute overlap or hit entries (they are
        stored as 0 / False). The sensitivity sweep must not request a
        ``d_max`` larger than this value.

        Choosing ``max_precompute_dist`` is a time / accuracy trade-off:
        larger values make precompute slower but let the sweep explore
        larger ``d_max``. For the paper's main analysis (``d_max = 20``)
        we use ``max_precompute_dist = 20``. For the sensitivity
        supplementary figure (grid up to ``d_max = 100``) we use
        ``max_precompute_dist = 100``.

    Returns
    -------
    PairwiseTensors
    """
    n_pred = len(pred_objs)
    n_gt = len(gt_objs)

    dist = np.full((n_pred, n_gt), np.inf, dtype=np.float32)
    overlap = np.zeros((n_pred, n_gt), dtype=np.float32)
    hit = np.zeros((n_pred, n_gt), dtype=bool)

    if n_pred == 0 or n_gt == 0:
        return PairwiseTensors(
            dist=dist,
            overlap=overlap,
            hit=hit,
            n_pred=n_pred,
            n_gt=n_gt,
            max_precompute_dist=float(max_precompute_dist),
        )

    max_d = float(max_precompute_dist)

    for i, p in enumerate(pred_objs):
        p_bbox = p["bbox"]
        p_cent = p["centroid"]
        p_mask = p.get("mask")
        py1, px1, py2, px2 = (int(v) for v in p_bbox)

        for j, g in enumerate(gt_objs):
            g_cent = g["centroid"]
            g_mask = g.get("mask")

            # Distance gate (cheap, skip everything else if too far).
            d = centroid_distance(p_cent, g_cent)
            if d <= max_d:
                dist[i, j] = float(d)

            # Overlap + hit gate. We compute these even when d > max_d so
            # that the OR-policy path (which only needs one of the two gates
            # to pass) is not silently starved of candidates when the hit
            # gate alone would have admitted the pair.
            overlap_score = 0.0
            hit_gate = False

            if g_mask is not None:
                g_h, g_w = g_mask.shape[:2]
                # Clip the prediction bbox into the GS mask frame.
                y1 = max(0, min(py1, g_h))
                y2 = max(0, min(py2, g_h))
                x1 = max(0, min(px1, g_w))
                x2 = max(0, min(px2, g_w))

                if y2 > y1 and x2 > x1:
                    g_roi = g_mask[y1:y2, x1:x2] > 0
                    gt_pos_in_roi = int(np.count_nonzero(g_roi))

                    if gt_pos_in_roi > 0:
                        # Hit gate passes if *any* GS-positive pixel lies
                        # inside the prediction bbox.
                        hit_gate = True

                        if (
                            p_mask is not None
                            and hasattr(p_mask, "shape")
                            and p_mask.shape == g_mask.shape
                        ):
                            # CASE A: prediction carries a mask of the same
                            # shape as the GS mask. Use true mask-IoU inside
                            # the prediction bbox ROI. This penalizes huge
                            # prediction bounding boxes that happen to
                            # contain a small GS spot: their area_p is
                            # large, so union grows, so IoU falls.
                            p_roi = p_mask[y1:y2, x1:x2] > 0
                            inter = int(np.count_nonzero(p_roi & g_roi))
                            if inter > 0:
                                area_p = int(np.count_nonzero(p_roi))
                                area_g = gt_pos_in_roi
                                union = area_p + area_g - inter
                                if union > 0:
                                    overlap_score = float(inter / union)
                        else:
                            # CASE B: prediction has no mask (e.g., we only
                            # have its bbox). Fall back to treating the
                            # bbox itself as the prediction footprint:
                            # intersection = GS-positive pixels inside bbox,
                            # area_p     = full bbox ROI area,
                            # area_g     = gt_pos_in_roi,
                            # union      = area_p (because g_roi ⊆ bbox).
                            # → IoU reduces to gt_pos_in_roi / bbox_area.
                            area_p = int((y2 - y1) * (x2 - x1))
                            if area_p > 0:
                                overlap_score = float(gt_pos_in_roi / float(area_p))
            else:
                # GS mask missing → fallback to bbox IoU only. The hit gate
                # reduces to "do the bboxes overlap at all".
                overlap_score = float(bbox_iou(p_bbox, g["bbox"]))
                if overlap_score > 0.0:
                    hit_gate = True

            overlap[i, j] = overlap_score
            hit[i, j] = hit_gate

    return PairwiseTensors(
        dist=dist,
        overlap=overlap,
        hit=hit,
        n_pred=n_pred,
        n_gt=n_gt,
        max_precompute_dist=max_d,
    )


# ==============================================================================
# Fast resolver — one call per matching-parameter combination.
# ==============================================================================


def _validate_params(
    d_max: float,
    min_iou: float,
    alpha: float,
    policy: str,
    pw: PairwiseTensors,
) -> None:
    """Reject obviously-invalid parameter combinations early with a clear error."""
    if policy not in ("OR", "AND"):
        raise ValueError(f"policy must be 'OR' or 'AND', got {policy!r}")
    if d_max <= 0:
        raise ValueError(f"d_max must be > 0, got {d_max}")
    if d_max > pw.max_precompute_dist + 1e-9:
        raise ValueError(
            f"d_max={d_max} exceeds the precompute cap max_precompute_dist="
            f"{pw.max_precompute_dist}. Recompute PairwiseTensors with a "
            f"larger max_precompute_dist."
        )
    if not 0.0 <= min_iou <= 1.0:
        raise ValueError(f"min_iou must be in [0, 1], got {min_iou}")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")


def resolve_matching_from_pairwise(
    pw: PairwiseTensors,
    *,
    d_max: float,
    min_iou: float,
    alpha: float = 0.5,
    policy: MatchPolicy = "OR",
    big_cost: float = BIG_COST,
) -> MatchResult:
    """
    Given precomputed pairwise geometry, resolve the 1-to-1 assignment
    under a specific ``(d_max, min_iou, alpha, policy)`` configuration.

    This is the fast path used by the sensitivity sweep. Under the hood it
    is just:

      1. Re-derive ``dist_ok`` and ``overlap_ok`` from the cached tensors.
      2. Combine them under OR or AND to get ``valid[i,j]``.
      3. Build a cost matrix with ``BIG_COST`` on invalid entries and the
         paper's cost formula on valid ones.
      4. Run ``scipy.optimize.linear_sum_assignment``.
      5. Partition predictions into matched / ignored / unmatched by the
         definition documented on :class:`MatchResult`.

    Returns
    -------
    MatchResult
    """
    _validate_params(d_max, min_iou, alpha, policy, pw)

    n_pred, n_gt = pw.n_pred, pw.n_gt

    if n_pred == 0 and n_gt == 0:
        return MatchResult(n_pred=0, n_gt=0)
    if n_pred == 0:
        return MatchResult(
            unmatched_gt=list(range(n_gt)), n_pred=0, n_gt=n_gt
        )
    if n_gt == 0:
        return MatchResult(
            unmatched_pred=list(range(n_pred)), n_pred=n_pred, n_gt=0
        )

    dist_ok = np.isfinite(pw.dist) & (pw.dist <= float(d_max))
    overlap_ok = pw.hit & (pw.overlap >= float(min_iou))

    if policy == "OR":
        valid = dist_ok | overlap_ok
    else:  # "AND"
        valid = dist_ok & overlap_ok

    cost = np.full((n_pred, n_gt), float(big_cost), dtype=np.float64)
    if valid.any():
        alpha_f = float(alpha)
        beta_f = 1.0 - alpha_f
        # Where valid, fill in the paper's cost formula. Where invalid,
        # leave BIG_COST so Hungarian never picks the pair.
        cost[valid] = (
            alpha_f * (1.0 - pw.overlap[valid])
            + beta_f * (pw.dist[valid] / float(d_max))
        )

    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    matched_pred_set: set[int] = set()
    matched_gt_set: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < big_cost:
            matched.append((int(i), int(j)))
            matched_pred_set.add(int(i))
            matched_gt_set.add(int(j))

    # A prediction has a candidate iff some column in its row is valid.
    has_candidate = valid.any(axis=1)

    ignored_pred = [
        i
        for i in range(n_pred)
        if i not in matched_pred_set and bool(has_candidate[i])
    ]
    unmatched_pred = [
        i
        for i in range(n_pred)
        if i not in matched_pred_set and not bool(has_candidate[i])
    ]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt_set]

    return MatchResult(
        matched=matched,
        unmatched_pred=unmatched_pred,
        ignored_pred=ignored_pred,
        unmatched_gt=unmatched_gt,
        n_pred=n_pred,
        n_gt=n_gt,
    )


# ==============================================================================
# Convenience wrapper — full pipeline for a single (d_max, min_iou, policy).
# ==============================================================================


def match_objects(
    pred_objs: Sequence[dict[str, Any]],
    gt_objs: Sequence[dict[str, Any]],
    *,
    d_max: float | None = None,
    max_dist: float | None = None,
    min_iou: float,
    alpha: float = 0.5,
    policy: MatchPolicy = "OR",
    big_cost: float = BIG_COST,
) -> MatchResult:
    """
    End-to-end matcher: precompute pairwise geometry, then resolve.
    Accepts either d_max or max_dist as the distance threshold.
    Policy is normalised to uppercase so "or" and "OR" both work.
    """
    if d_max is None and max_dist is None:
        raise TypeError("match_objects() requires either 'd_max' or 'max_dist'")
    effective_d_max = float(d_max if d_max is not None else max_dist)
    policy = policy.upper()
    pw = precompute_pairwise(
        pred_objs, gt_objs, max_precompute_dist=effective_d_max
    )
    return resolve_matching_from_pairwise(
        pw,
        d_max=effective_d_max,
        min_iou=min_iou,
        alpha=alpha,
        policy=policy,
        big_cost=big_cost,
    )


# ==============================================================================
# Center-only matcher — used when GS is points, not masks.
# ==============================================================================


def match_boxes_to_centers(
    pred_objs: Sequence[dict[str, Any]],
    gt_centers: Sequence[tuple[float, float]],
    *,
    d_max: float,
    require_inside: bool = True,
    big_cost: float = BIG_COST,
) -> MatchResult:
    """
    Match predicted objects (with bboxes + centroids) against a list of
    gold-standard centroid points.

    This is the evaluation mode used when a dataset provides only point
    annotations (e.g., the refined centroid-only GS used for some
    supplementary analyses). The policy here is implicitly "AND": a pair
    must satisfy *both* the optional "point lies inside prediction bbox"
    gate and the centroid-distance gate. This is intentional: with no
    mask to compute IoU against, we have only two gates to work with and
    it is safer to require both than either.

    Parameters
    ----------
    pred_objs : sequence of dict
        Standard prediction objects.
    gt_centers : sequence of (cy, cx) tuples
        Gold-standard point annotations.
    d_max : float
        Maximum Euclidean centroid distance for a candidate pair.
    require_inside : bool, default True
        If True, the GS point must lie within the prediction bbox to be
        considered a candidate. If False, the distance gate alone is used.
    big_cost : float, default BIG_COST
        Sentinel for invalid pairs in the cost matrix.

    Returns
    -------
    MatchResult
    """
    n_pred = len(pred_objs)
    n_gt = len(gt_centers)

    if n_pred == 0 and n_gt == 0:
        return MatchResult(n_pred=0, n_gt=0)
    if n_pred == 0:
        return MatchResult(
            unmatched_gt=list(range(n_gt)), n_pred=0, n_gt=n_gt
        )
    if n_gt == 0:
        return MatchResult(
            unmatched_pred=list(range(n_pred)), n_pred=n_pred, n_gt=0
        )

    if d_max <= 0:
        raise ValueError(f"d_max must be > 0, got {d_max}")

    cost = np.full((n_pred, n_gt), float(big_cost), dtype=np.float64)
    d_max_f = float(d_max)

    for i, p in enumerate(pred_objs):
        y1, x1, y2, x2 = p["bbox"]
        cy_p, cx_p = p["centroid"]

        for j, (gy, gx) in enumerate(gt_centers):
            if require_inside and not (y1 <= gy <= y2 and x1 <= gx <= x2):
                continue

            d = centroid_distance((cy_p, cx_p), (gy, gx))
            if d > d_max_f:
                continue

            cost[i, j] = d / d_max_f

    row_ind, col_ind = linear_sum_assignment(cost)

    matched: list[tuple[int, int]] = []
    matched_pred_set: set[int] = set()
    matched_gt_set: set[int] = set()

    for i, j in zip(row_ind, col_ind):
        if cost[i, j] < big_cost:
            matched.append((int(i), int(j)))
            matched_pred_set.add(int(i))
            matched_gt_set.add(int(j))

    has_candidate = (cost < big_cost).any(axis=1)

    ignored_pred = [
        i
        for i in range(n_pred)
        if i not in matched_pred_set and bool(has_candidate[i])
    ]
    unmatched_pred = [
        i
        for i in range(n_pred)
        if i not in matched_pred_set and not bool(has_candidate[i])
    ]
    unmatched_gt = [j for j in range(n_gt) if j not in matched_gt_set]

    return MatchResult(
        matched=matched,
        unmatched_pred=unmatched_pred,
        ignored_pred=ignored_pred,
        unmatched_gt=unmatched_gt,
        n_pred=n_pred,
        n_gt=n_gt,
    )
