"""
tests/test_matching.py
=======================
Tests for ecdna_bench.evaluation.matching.

These tests verify the invariants documented in §5 of REWRITE_PLAN.md,
which are the most important invariants for reviewer trust.

Critical invariants tested
--------------------------
1.  tp + fp + fn + ignored == n_pred + n_gt   (total partition)
2.  tp + fp + ignored == n_pred               (every pred accounted for)
3.  tp + fn == n_gt                           (every GT accounted for)
4.  Perfect match → tp=N, fp=fn=ignored=0
5.  No overlap / far apart → no TP
6.  OR policy F1 ≥ AND policy F1             (OR is more permissive)
7.  3 preds 1 GT → tp=1, ignored=2 (OR, close distance)
8.  Empty pred + non-empty GT → tp=0, fn=N, fp=0
9.  Empty pred + empty GT → all zeros
10. Non-empty pred + empty GT → tp=0, fp=N, fn=0
"""

from __future__ import annotations

from typing import List, Dict, Tuple

import numpy as np
import pytest

# We import the evaluation matching module under test
from ecdna_bench.evaluation.matching import match_objects, MatchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _f1(tp: int, fp: int, fn: int) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d > 0 else 0.0


def _check_partitions(result: MatchResult, n_pred: int, n_gt: int) -> None:
    """Assert all partition invariants for one MatchResult."""
    assert result.tp + result.fp + result.ignored_count == n_pred, (
        f"pred partition: tp={result.tp} fp={result.fp} ign={result.ignored_count} "
        f"!= n_pred={n_pred}"
    )
    assert result.tp + result.fn == n_gt, (
        f"gt partition: tp={result.tp} fn={result.fn} != n_gt={n_gt}"
    )


# ---------------------------------------------------------------------------
# Fixture shorthand builders
# ---------------------------------------------------------------------------

def _obj(cy: float, cx: float, shape: Tuple[int, int] = (64, 64),
          h: int = 5, w: int = 5) -> Dict:
    y1 = max(0, int(cy) - h // 2)
    x1 = max(0, int(cx) - w // 2)
    y2 = min(shape[0], y1 + h)
    x2 = min(shape[1], x1 + w)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[y1:y2, x1:x2] = 1
    return {
        "centroid": (float(cy), float(cx)),
        "bbox":     (y1, x1, y2, x2),
        "area":     int((y2 - y1) * (x2 - x1)),
        "mask":     mask,
    }


# ---------------------------------------------------------------------------
# 1–3. Partition invariants
# ---------------------------------------------------------------------------

class TestPartitionInvariants:

    def test_partition_perfect_match(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=2, n_gt=2)

    def test_partition_no_overlap(self, no_overlap_objects):
        pred, gt = no_overlap_objects
        r = match_objects(pred, gt, max_dist=5, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=len(pred), n_gt=len(gt))

    def test_partition_empty_pred(self, two_gt_objects):
        r = match_objects([], two_gt_objects, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=0, n_gt=2)

    def test_partition_empty_gt(self, two_pred_objects):
        r = match_objects(two_pred_objects, [], max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=2, n_gt=0)

    def test_partition_both_empty(self):
        r = match_objects([], [], max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=0, n_gt=0)

    def test_partition_large_sets(self):
        pred = [_obj(5 + i * 10, 5 + i * 10) for i in range(5)]
        gt   = [_obj(5 + i * 10, 5 + i * 10) for i in range(5)]
        r = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=5, n_gt=5)

    def test_partition_more_preds_than_gt(self):
        gt   = [_obj(20, 20)]
        pred = [_obj(20, 20), _obj(20.5, 20.5), _obj(5, 5)]
        r = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=3, n_gt=1)

    def test_partition_more_gt_than_pred(self):
        pred = [_obj(20, 20)]
        gt   = [_obj(20, 20), _obj(40, 40), _obj(5, 5)]
        r = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        _check_partitions(r, n_pred=1, n_gt=3)


# ---------------------------------------------------------------------------
# 4. Perfect match
# ---------------------------------------------------------------------------

class TestPerfectMatch:

    def test_perfect_match_two_objects(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert r.tp == 2
        assert r.fp == 0
        assert r.fn == 0
        assert r.ignored_count == 0

    def test_perfect_match_single_object(self):
        gt   = [_obj(15, 15)]
        pred = [_obj(15, 15)]
        r = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert r.tp == 1 and r.fp == 0 and r.fn == 0 and r.ignored_count == 0

    def test_perfect_match_f1_equals_one(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert abs(_f1(r.tp, r.fp, r.fn) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# 5. No overlap / far apart
# ---------------------------------------------------------------------------

class TestNoOverlap:

    def test_far_apart_no_tp(self, no_overlap_objects):
        pred, gt = no_overlap_objects
        r = match_objects(pred, gt, max_dist=5, min_iou=0.1, alpha=0.5, policy="or")
        assert r.tp == 0
        # pred is far → FP (not ignored, no valid candidate)
        assert r.fp == 1
        assert r.fn == 1
        assert r.ignored_count == 0

    def test_far_apart_and_policy(self, no_overlap_objects):
        pred, gt = no_overlap_objects
        r = match_objects(pred, gt, max_dist=5, min_iou=0.5, alpha=0.5, policy="and")
        assert r.tp == 0

    def test_zero_pred_all_fn(self, two_gt_objects):
        r = match_objects([], two_gt_objects, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert r.tp == 0
        assert r.fn == 2
        assert r.fp == 0
        assert r.ignored_count == 0


# ---------------------------------------------------------------------------
# 6. OR ≥ AND
# ---------------------------------------------------------------------------

class TestORvsAND:

    def _f1_policy(self, policy: str) -> float:
        gt   = [_obj(20, 20)]
        pred = [_obj(22, 20)]  # slightly offset — may fail AND but pass OR
        r = match_objects(pred, gt, max_dist=20, min_iou=0.05, alpha=0.5, policy=policy)
        return _f1(r.tp, r.fp, r.fn)

    def test_or_geq_and(self):
        f1_or  = self._f1_policy("or")
        f1_and = self._f1_policy("and")
        assert f1_or >= f1_and, f"OR F1={f1_or:.4f} < AND F1={f1_and:.4f}"

    def test_or_geq_and_many_objects(self):
        """Randomised: for every scenario, OR F1 ≥ AND F1."""
        rng = np.random.default_rng(42)
        for _ in range(20):
            n = rng.integers(1, 6)
            positions = rng.uniform(5, 55, size=(n, 2))
            noise     = rng.uniform(-3, 3, size=(n, 2))
            gt   = [_obj(float(r[0]), float(r[1])) for r in positions]
            pred = [_obj(float(r[0]+n[0]), float(r[1]+n[1]))
                    for r, n in zip(positions, noise)]
            r_or  = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
            r_and = match_objects(pred, gt, max_dist=20, min_iou=0.1, alpha=0.5, policy="and")
            assert _f1(r_or.tp, r_or.fp, r_or.fn) >= _f1(r_and.tp, r_and.fp, r_and.fn)


# ---------------------------------------------------------------------------
# 7. 3 preds on 1 GT → tp=1, ignored=2
# ---------------------------------------------------------------------------

class TestIgnoredBucket:

    def test_three_preds_one_gt(self, three_preds_one_gt):
        pred, gt = three_preds_one_gt
        # Large d_max so all three preds are valid candidates for the GT
        r = match_objects(pred, gt, max_dist=40, min_iou=0.0, alpha=0.5, policy="or")
        assert r.tp == 1, f"expected tp=1, got {r.tp}"
        # The 2 losing candidates should be 'ignored', not FP
        assert r.ignored_count == 2, f"expected ignored=2, got {r.ignored_count}"
        assert r.fp == 0, f"expected fp=0, got {r.fp}"
        _check_partitions(r, n_pred=3, n_gt=1)

    def test_ignored_not_fp_or_policy(self):
        """Candidates that lose the assignment under OR should be ignored."""
        gt   = [_obj(15, 15)]
        pred = [_obj(15, 15), _obj(16, 15), _obj(15, 16)]
        r = match_objects(pred, gt, max_dist=30, min_iou=0.0, alpha=0.5, policy="or")
        assert r.tp == 1
        assert r.fp == 0
        assert r.ignored_count == 2
        _check_partitions(r, n_pred=3, n_gt=1)

    def test_isolated_pred_is_fp_not_ignored(self):
        """A pred with NO valid candidates at all must be FP, not ignored."""
        gt   = [_obj(5, 5)]           # GT near origin
        pred = [_obj(5, 5), _obj(50, 50)]  # second pred far from any GT
        # With d_max=5 and min_iou=0.1: pred@(50,50) has dist≈63 > d_max
        # and IoU=0 < 0.1 → no valid candidate → must be FP under OR policy
        r = match_objects(pred, gt, max_dist=5, min_iou=0.1, alpha=0.5, policy="or")
        # First pred matches GT → tp=1; second has no candidates → fp=1
        assert r.tp == 1
        assert r.fp == 1
        assert r.ignored_count == 0
        _check_partitions(r, n_pred=2, n_gt=1)


# ---------------------------------------------------------------------------
# 8 & 9. Empty cases
# ---------------------------------------------------------------------------

class TestEmptyCases:

    def test_empty_pred(self, two_gt_objects):
        r = match_objects([], two_gt_objects, max_dist=20, min_iou=0.1,
                          alpha=0.5, policy="or")
        assert r.tp == 0 and r.fp == 0 and r.fn == 2 and r.ignored_count == 0

    def test_empty_gt(self, two_pred_objects):
        r = match_objects(two_pred_objects, [], max_dist=20, min_iou=0.1,
                          alpha=0.5, policy="or")
        assert r.tp == 0 and r.fp == 2 and r.fn == 0 and r.ignored_count == 0

    def test_both_empty(self):
        r = match_objects([], [], max_dist=20, min_iou=0.1,
                          alpha=0.5, policy="or")
        assert r.tp == 0 and r.fp == 0 and r.fn == 0 and r.ignored_count == 0

    def test_both_empty_partition(self):
        r = match_objects([], [], max_dist=20, min_iou=0.1,
                          alpha=0.5, policy="or")
        _check_partitions(r, n_pred=0, n_gt=0)


# ---------------------------------------------------------------------------
# Extra: policy names are case-insensitive
# ---------------------------------------------------------------------------

class TestPolicyNames:

    def test_or_case_insensitive(self, two_gt_objects, two_pred_objects):
        r_lower = match_objects(two_pred_objects, two_gt_objects,
                                max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        r_upper = match_objects(two_pred_objects, two_gt_objects,
                                max_dist=20, min_iou=0.1, alpha=0.5, policy="OR")
        assert r_lower.tp == r_upper.tp

    def test_invalid_policy_raises(self, two_gt_objects, two_pred_objects):
        with pytest.raises((ValueError, KeyError)):
            match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="xor")


# ---------------------------------------------------------------------------
# Extra: MatchResult properties
# ---------------------------------------------------------------------------

class TestMatchResultProperties:

    def test_result_is_matchresult(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert isinstance(r, MatchResult)

    def test_result_attributes_non_negative(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        assert r.tp >= 0 and r.fp >= 0 and r.fn >= 0 and r.ignored_count >= 0

    def test_frozen_result_immutable(self, two_gt_objects, two_pred_objects):
        r = match_objects(two_pred_objects, two_gt_objects,
                          max_dist=20, min_iou=0.1, alpha=0.5, policy="or")
        # MatchResult should be frozen/immutable
        with pytest.raises((AttributeError, TypeError)):
            r.tp = 999  # type: ignore[misc]
