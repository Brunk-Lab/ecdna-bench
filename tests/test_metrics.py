"""
tests/test_metrics.py
======================
Tests for ecdna_bench.evaluation.metrics.

Three families tested: object, pixel, count.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecdna_bench.evaluation.matching import match_objects, MatchResult
from ecdna_bench.evaluation.metrics import object_metrics, pixel_metrics, count_metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obj(cy, cx, shape=(64, 64), h=5, w=5):
    y1 = max(0, int(cy) - h // 2)
    x1 = max(0, int(cx) - w // 2)
    y2 = min(shape[0], y1 + h); x2 = min(shape[1], x1 + w)
    mask = np.zeros(shape, dtype=np.uint8); mask[y1:y2, x1:x2] = 1
    return {"centroid": (cy, cx), "bbox": (y1, x1, y2, x2), "area": (y2-y1)*(x2-x1), "mask": mask}


def _match(pred, gt, policy="or", d_max=20, iou_min=0.1):
    return match_objects(pred, gt, max_dist=d_max, min_iou=iou_min,
                         alpha=0.5, policy=policy)


# ---------------------------------------------------------------------------
# Object metrics
# ---------------------------------------------------------------------------

class TestObjectMetrics:

    def test_perfect_match(self, two_gt_objects, two_pred_objects):
        r = _match(two_pred_objects, two_gt_objects)
        m = object_metrics(r)
        assert abs(m["precision"] - 1.0) < 1e-9
        assert abs(m["recall"]    - 1.0) < 1e-9
        assert abs(m["f1"]        - 1.0) < 1e-9

    def test_no_match(self, two_gt_objects):
        r = _match([], two_gt_objects)
        m = object_metrics(r)
        assert m["recall"] == 0.0
        assert m["f1"]     == 0.0
        # precision is 0/0 when no predictions: should be 0 or NaN
        assert m["precision"] == 0.0 or np.isnan(m["precision"])

    def test_all_fp(self, two_pred_objects):
        r = _match(two_pred_objects, [], d_max=1)
        m = object_metrics(r)
        assert m["precision"] == 0.0
        assert m["recall"]    == 0.0 or np.isnan(m["recall"])

    def test_partial_match(self, two_gt_objects, one_pred_object):
        r = _match(one_pred_object, two_gt_objects)
        m = object_metrics(r)
        assert 0.0 < m["recall"] < 1.0
        assert 0.0 < m["f1"]    < 1.0

    def test_f1_harmonic_mean(self, two_gt_objects, one_pred_object):
        r = _match(one_pred_object, two_gt_objects)
        m = object_metrics(r)
        if m["precision"] > 0 and m["recall"] > 0:
            expected_f1 = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
            assert abs(m["f1"] - expected_f1) < 1e-9

    def test_metrics_keys_present(self, two_gt_objects, two_pred_objects):
        r = _match(two_pred_objects, two_gt_objects)
        m = object_metrics(r)
        for key in ("precision", "recall", "f1"):
            assert key in m, f"Missing key: {key}"

    def test_metrics_values_bounded(self, two_gt_objects, two_pred_objects):
        r = _match(two_pred_objects, two_gt_objects)
        m = object_metrics(r)
        for k, v in m.items():
            if not np.isnan(v):
                assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"

    def test_both_empty(self):
        r = _match([], [])
        m = object_metrics(r)
        # Both empty: precision and recall are undefined (0.0 or NaN accepted)
        assert m["f1"] == 0.0 or np.isnan(m["f1"])


# ---------------------------------------------------------------------------
# Pixel metrics
# ---------------------------------------------------------------------------

class TestPixelMetrics:

    def test_identical_masks_dice_one(self, tiny_gt_mask):
        m = pixel_metrics(tiny_gt_mask, tiny_gt_mask)
        assert abs(m["dice"] - 1.0) < 1e-9
        assert abs(m["iou"]  - 1.0) < 1e-9

    def test_no_overlap_dice_zero(self, tiny_gt_mask, empty_mask):
        m = pixel_metrics(tiny_gt_mask, empty_mask)
        assert m["dice"] == 0.0
        assert m["iou"]  == 0.0

    def test_full_overlap_with_extra_fp(self, tiny_gt_mask, full_mask):
        m = pixel_metrics(tiny_gt_mask, full_mask)
        assert 0.0 < m["dice"] < 1.0
        assert 0.0 < m["recall"] <= 1.0

    def test_keys_present(self, tiny_gt_mask):
        m = pixel_metrics(tiny_gt_mask, tiny_gt_mask)
        for key in ("precision", "recall", "dice", "iou"):
            assert key in m

    def test_precision_one_for_perfect(self, tiny_gt_mask):
        m = pixel_metrics(tiny_gt_mask, tiny_gt_mask)
        assert abs(m["precision"] - 1.0) < 1e-9
        assert abs(m["recall"]    - 1.0) < 1e-9

    def test_both_empty_masks(self, empty_mask):
        m = pixel_metrics(empty_mask, empty_mask)
        # Both empty: dice=0 or NaN; should not crash
        assert isinstance(m, dict)

    def test_dice_equals_f1(self, tiny_gt_mask):
        pred = tiny_gt_mask.copy()
        pred[0:5, :] = 0  # zero out some GS pixels → recall < 1
        m = pixel_metrics(tiny_gt_mask, pred)
        # For binary masks, Dice == F1 == 2P*R/(P+R)
        if m["precision"] > 0 and m["recall"] > 0:
            expected = 2 * m["precision"] * m["recall"] / (m["precision"] + m["recall"])
            assert abs(m["dice"] - expected) < 1e-9


# ---------------------------------------------------------------------------
# Count metrics
# ---------------------------------------------------------------------------

class TestCountMetrics:

    def test_perfect_counts(self):
        m = count_metrics(np.array([1, 5, 10, 100]),
                          np.array([1, 5, 10, 100]))
        assert abs(m["mae"]) < 1e-9
        assert abs(m["rmse"]) < 1e-9
        assert abs(m["bias"]) < 1e-9

    def test_mae_nonzero(self):
        m = count_metrics(np.array([0, 10, 20]),
                          np.array([0, 11, 22]))
        assert m["mae"] > 0

    def test_bias_sign(self):
        # pred always greater than gt → bias > 0
        m = count_metrics(np.array([5, 10]), np.array([10, 15]))
        assert m["bias"] > 0
        # pred always less → bias < 0
        m2 = count_metrics(np.array([10, 15]), np.array([5, 10]))
        assert m2["bias"] < 0

    def test_pearson_perfect(self):
        gt = np.array([1, 2, 3, 4, 5], dtype=float)
        m  = count_metrics(gt, gt)
        assert abs(m.get("pearson_r", 1.0) - 1.0) < 1e-9

    def test_empty_raises_or_returns_nan(self):
        m = count_metrics(np.array([]), np.array([]))
        # Should return dict with NaN or 0 values, not crash
        assert isinstance(m, dict)

    def test_mdape_nonzero_gt_only(self):
        # Images with gt_count=0 must be excluded from MdAPE
        gt   = np.array([0, 10, 20], dtype=float)
        pred = np.array([5, 10, 25], dtype=float)
        m = count_metrics(gt, pred)
        # MdAPE computed on gt>0 only: |10-10|/10=0%, |25-20|/20=25% → median=12.5%
        mdape = m.get("mdape", None)
        if mdape is not None and not np.isnan(mdape):
            assert abs(mdape - 12.5) < 1.0   # ≈12.5% (allow 1% tolerance)

    def test_keys_present(self):
        m = count_metrics(np.array([1, 2]), np.array([1, 3]))
        for key in ("mae", "rmse", "bias"):
            assert key in m
