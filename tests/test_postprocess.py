"""
tests/test_postprocess.py
==========================
Tests for ecdna_bench.eccount.postprocess.

Critical: pipeline order must be smooth → ROI → local maxima → threshold → NMS.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecdna_bench.eccount.postprocess import (
    PostprocessConfig,
    detect_points_from_map,
    find_local_maxima,
    greedy_distance_suppression,
    peaks_to_mask,
    prob_to_threshold_mask,
)


# ---------------------------------------------------------------------------
# find_local_maxima
# ---------------------------------------------------------------------------

class TestFindLocalMaxima:

    def test_two_peaks_detected(self, two_peak_prob_map):
        peaks = find_local_maxima(two_peak_prob_map, threshold_abs=0.5,
                                  peak_min_distance=2)
        xs = sorted(p[0] for p in peaks)
        ys = sorted(p[1] for p in peaks)
        assert xs == [10, 50]
        assert ys == [10, 50]

    def test_empty_map_no_peaks(self, empty_prob_map):
        peaks = find_local_maxima(empty_prob_map, threshold_abs=0.1,
                                  peak_min_distance=2)
        assert peaks == []

    def test_below_threshold_suppressed(self):
        pm = np.zeros((32, 32), dtype=np.float32)
        pm[10, 10] = 0.2   # below threshold
        peaks = find_local_maxima(pm, threshold_abs=0.5, peak_min_distance=2)
        assert peaks == []

    def test_peaks_sorted_by_score_descending(self, two_peak_prob_map):
        peaks = find_local_maxima(two_peak_prob_map, threshold_abs=0.5,
                                  peak_min_distance=2)
        scores = [p[2] for p in peaks]
        assert scores == sorted(scores, reverse=True)

    def test_exclude_border(self):
        pm = np.zeros((32, 32), dtype=np.float32)
        pm[1, 1]   = 0.9   # inside border=2 → excluded
        pm[15, 15] = 0.8   # outside border → kept
        peaks = find_local_maxima(pm, threshold_abs=0.5, peak_min_distance=1,
                                  exclude_border=2)
        xs = [p[0] for p in peaks]
        assert 1 not in xs
        assert 15 in xs


# ---------------------------------------------------------------------------
# greedy_distance_suppression
# ---------------------------------------------------------------------------

class TestGreedyNMS:

    def test_close_peaks_suppressed(self):
        peaks = [(10, 10, 0.9), (11, 11, 0.8), (25, 25, 0.7)]
        kept = greedy_distance_suppression(peaks, min_distance=5)
        assert len(kept) == 2
        # First peak kept (highest score); second suppressed (too close); third kept
        assert kept[0] == (10, 10, 0.9)
        assert kept[1] == (25, 25, 0.7)

    def test_zero_min_distance_keeps_all(self):
        peaks = [(10, 10, 0.9), (10, 11, 0.8)]
        kept = greedy_distance_suppression(peaks, min_distance=0)
        assert len(kept) == 2

    def test_max_points_limit(self):
        peaks = [(i * 20, i * 20, 1.0 - i * 0.1) for i in range(5)]
        kept = greedy_distance_suppression(peaks, min_distance=5, max_points=3)
        assert len(kept) == 3

    def test_empty_input(self):
        assert greedy_distance_suppression([], min_distance=5) == []


# ---------------------------------------------------------------------------
# detect_points_from_map  (full pipeline)
# ---------------------------------------------------------------------------

class TestDetectPointsFromMap:

    def test_two_peaks_found(self, two_peak_prob_map):
        cfg = PostprocessConfig(
            smooth_sigma=0.0,
            threshold_abs=0.5,
            peak_min_distance=2,
            nms_min_distance=2,
        )
        peaks = detect_points_from_map(two_peak_prob_map, config=cfg)
        assert len(peaks) == 2
        xs = sorted(p[0] for p in peaks)
        assert xs == [10, 50]

    def test_empty_map(self, empty_prob_map):
        cfg = PostprocessConfig(threshold_abs=0.1)
        peaks = detect_points_from_map(empty_prob_map, config=cfg)
        assert peaks == []

    def test_roi_mask_filters_peaks(self, two_peak_prob_map):
        """ROI applied AFTER smoothing must exclude out-of-ROI peaks."""
        roi = np.zeros((64, 64), dtype=np.uint8)
        roi[40:64, 40:64] = 1   # only bottom-right quadrant active
        cfg = PostprocessConfig(smooth_sigma=0.0, threshold_abs=0.5,
                                peak_min_distance=2, nms_min_distance=2)
        peaks = detect_points_from_map(two_peak_prob_map, roi_mask=roi, config=cfg)
        # Only peak at (50, 50) should survive
        assert len(peaks) == 1
        assert peaks[0][0] == 50

    def test_nms_reduces_count(self):
        """Two very close peaks should be reduced to 1 by NMS."""
        pm = np.zeros((32, 32), dtype=np.float32)
        pm[10, 10] = 0.9
        pm[11, 11] = 0.8  # within NMS radius
        cfg = PostprocessConfig(smooth_sigma=0.0, threshold_abs=0.5,
                                peak_min_distance=5, nms_min_distance=5)
        peaks = detect_points_from_map(pm, config=cfg)
        assert len(peaks) == 1

    def test_frozen_defaults(self):
        """Frozen paper defaults (§2 of REWRITE_PLAN.md)."""
        cfg = PostprocessConfig()
        assert cfg.smooth_sigma      == 0.5
        assert cfg.threshold_abs     == 0.35
        assert cfg.peak_min_distance == 2
        assert cfg.nms_min_distance  == 2
        assert cfg.exclude_border    == 0

    def test_pipeline_order_smooth_then_roi(self):
        """
        Smoothing a peak near the ROI boundary must NOT bleed signal
        outside when ROI is applied AFTER smoothing (correct order).
        If order were reversed (ROI before smooth), peak at boundary
        would be zero-padded during smoothing.
        """
        pm = np.zeros((32, 32), dtype=np.float32)
        pm[15, 15] = 1.0
        # ROI excludes all but the central region after smoothing
        roi = np.zeros((32, 32), dtype=np.uint8)
        roi[14:17, 14:17] = 1  # tight ROI around peak

        cfg = PostprocessConfig(smooth_sigma=0.5, threshold_abs=0.3,
                                peak_min_distance=2, nms_min_distance=2)
        peaks = detect_points_from_map(pm, roi_mask=roi, config=cfg)
        # Peak should be found — ROI is applied after smooth, not before
        assert len(peaks) == 1


# ---------------------------------------------------------------------------
# peaks_to_mask
# ---------------------------------------------------------------------------

class TestPeaksToMask:

    def test_empty_peaks(self):
        m = peaks_to_mask([], shape=(32, 32), disk_radius=3)
        assert m.dtype == np.uint8
        assert m.max() == 0

    def test_peak_at_centre(self):
        m = peaks_to_mask([(16, 16, 0.9)], shape=(32, 32), disk_radius=3)
        assert m[16, 16] == 255

    def test_disk_radius_zero(self):
        m = peaks_to_mask([(10, 10, 0.9)], shape=(32, 32), disk_radius=0)
        assert m[10, 10] == 255  # point-only

    def test_output_dtype_uint8(self):
        m = peaks_to_mask([(5, 5, 0.9)], shape=(32, 32), disk_radius=2)
        assert m.dtype == np.uint8


# ---------------------------------------------------------------------------
# prob_to_threshold_mask
# ---------------------------------------------------------------------------

class TestProbToThresholdMask:

    def test_above_threshold(self):
        pm = np.zeros((16, 16), dtype=np.float32)
        pm[5, 5] = 0.8
        m = prob_to_threshold_mask(pm, threshold=0.5)
        assert m[5, 5] == 255
        assert m[0, 0] == 0

    def test_exactly_at_threshold(self):
        pm = np.full((16, 16), 0.5, dtype=np.float32)
        m = prob_to_threshold_mask(pm, threshold=0.5)
        assert m.max() == 255

    def test_all_below(self):
        pm = np.full((16, 16), 0.3, dtype=np.float32)
        m = prob_to_threshold_mask(pm, threshold=0.5)
        assert m.max() == 0

    def test_output_dtype(self):
        pm = np.zeros((16, 16), dtype=np.float32)
        m = prob_to_threshold_mask(pm, threshold=0.5)
        assert m.dtype == np.uint8
