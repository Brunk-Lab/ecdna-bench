"""
tests/test_targets.py
======================
Tests for ecdna_bench.eccount.targets.
"""

from __future__ import annotations

import numpy as np
import pytest

from ecdna_bench.eccount.targets import (
    SoftTargetConfig,
    make_centroid_gaussian_target,
    extract_component_centroids,
    gaussian_at_points,
    resize_target,
)


class TestExtractCentroids:

    def test_empty_mask_returns_empty(self, empty_mask):
        pts = extract_component_centroids(empty_mask)
        assert pts.shape == (0, 2)

    def test_single_blob(self, single_blob_mask):
        pts = extract_component_centroids(single_blob_mask)
        assert pts.shape == (1, 2)

    def test_two_blobs(self, tiny_gt_mask):
        pts = extract_component_centroids(tiny_gt_mask)
        assert pts.shape == (2, 2)

    def test_centroid_dtype(self, tiny_gt_mask):
        pts = extract_component_centroids(tiny_gt_mask)
        assert pts.dtype == np.float32


class TestGaussianAtPoints:

    def test_empty_points_returns_zeros(self):
        out = gaussian_at_points((32, 32), np.zeros((0, 2), dtype=np.float32),
                                 sigma=1.0, normalize=True)
        assert out.max() == 0.0
        assert out.shape == (32, 32)

    def test_single_point_peak_at_centroid(self):
        pts = np.array([[16.0, 16.0]], dtype=np.float32)  # (x=16, y=16)
        out = gaussian_at_points((32, 32), pts, sigma=1.0, normalize=True)
        assert abs(out.max() - 1.0) < 1e-6
        # Peak should be at row 16, col 16 (y=16, x=16)
        assert out[16, 16] == out.max()

    def test_normalize_max_equals_one(self):
        pts = np.array([[10.0, 10.0], [20.0, 20.0]], dtype=np.float32)
        out = gaussian_at_points((32, 32), pts, sigma=1.0, normalize=True)
        assert abs(out.max() - 1.0) < 1e-6

    def test_max_mode_does_not_exceed_one(self):
        pts = np.array([[16.0, 16.0], [16.5, 16.5]], dtype=np.float32)
        out = gaussian_at_points((32, 32), pts, sigma=2.0,
                                 merge_mode="max", normalize=True)
        assert out.max() <= 1.0 + 1e-9

    def test_sum_mode_can_exceed_one_before_normalize(self):
        pts = np.array([[16.0, 16.0], [16.5, 16.5]], dtype=np.float32)
        out = gaussian_at_points((32, 32), pts, sigma=3.0,
                                 merge_mode="sum", normalize=False)
        assert out.max() > 1.0

    def test_invalid_sigma_raises(self):
        pts = np.array([[10.0, 10.0]], dtype=np.float32)
        with pytest.raises((ValueError, AssertionError)):
            gaussian_at_points((32, 32), pts, sigma=0.0)


class TestMakeCentroidGaussianTarget:

    def test_empty_mask_returns_zeros(self, empty_mask):
        t = make_centroid_gaussian_target(empty_mask, SoftTargetConfig())
        assert t.max() == 0.0
        assert t.shape == empty_mask.shape

    def test_single_centroid_peak_is_one(self, single_blob_mask):
        t = make_centroid_gaussian_target(single_blob_mask, SoftTargetConfig(sigma=1.0))
        assert abs(t.max() - 1.0) < 1e-6

    def test_two_centroids_normalized(self, tiny_gt_mask):
        t = make_centroid_gaussian_target(tiny_gt_mask, SoftTargetConfig(sigma=1.0))
        assert abs(t.max() - 1.0) < 1e-6
        assert t.min() >= 0.0

    def test_max_merge_mode_does_not_accumulate(self, tiny_gt_mask):
        t = make_centroid_gaussian_target(
            tiny_gt_mask, SoftTargetConfig(sigma=3.0, merge_mode="max", normalize=False)
        )
        assert t.max() <= 1.0 + 1e-9

    def test_output_dtype_float32(self, tiny_gt_mask):
        t = make_centroid_gaussian_target(tiny_gt_mask)
        assert t.dtype == np.float32

    def test_output_shape_matches_input(self, tiny_gt_mask):
        t = make_centroid_gaussian_target(tiny_gt_mask)
        assert t.shape == tiny_gt_mask.shape

    def test_frozen_sigma_default(self):
        """Frozen paper default σ = 1.0 (§2 of REWRITE_PLAN.md)."""
        cfg = SoftTargetConfig()
        assert cfg.sigma == 1.0

    def test_non_normalize_mode(self, single_blob_mask):
        """Without normalize, peak should be ≤ 1.0 for max-mode."""
        t = make_centroid_gaussian_target(
            single_blob_mask,
            SoftTargetConfig(sigma=1.0, normalize=False, merge_mode="max"),
        )
        assert t.max() <= 1.0 + 1e-9


class TestResizeTarget:

    def test_resize_shape(self):
        t = np.ones((32, 32), dtype=np.float32)
        out = resize_target(t, (16, 24))
        assert out.shape == (16, 24)

    def test_resize_dtype(self):
        t = np.ones((32, 32), dtype=np.float32)
        out = resize_target(t, (16, 16))
        assert out.dtype == np.float32

    def test_resize_identity(self):
        t = np.random.rand(32, 32).astype(np.float32)
        out = resize_target(t, (32, 32))
        np.testing.assert_array_equal(out, t)
