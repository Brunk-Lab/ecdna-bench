"""
tests/test_io.py
=================
Tests for ecdna_bench.baselines.ecseg I/O helpers
(load_and_binarize, apply_min_area_filter, write_binary_mask)
and channel extraction.

These helpers are re-used by mia.py and label_engine.py.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ecdna_bench.baselines.ecseg import (
    apply_min_area_filter,
    load_and_binarize,
    write_binary_mask,
)

FIXTURES = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# load_and_binarize
# ---------------------------------------------------------------------------

class TestLoadAndBinarize:

    def test_grayscale_png_round_trip(self, tmp_png_dir):
        # Write {0, 255} mask and read back
        m = np.zeros((16, 16), dtype=np.uint8)
        m[4:12, 4:12] = 255
        path = tmp_png_dir / "test.png"
        cv2.imwrite(str(path), m)

        b = load_and_binarize(path, threshold=0.0)
        assert b.dtype == np.uint8
        assert b.shape == (16, 16)
        assert b[8, 8] == 1    # foreground
        assert b[0, 0] == 0    # background

    def test_values_are_binary(self, tmp_png_dir):
        m = np.random.randint(0, 256, (16, 16), dtype=np.uint8)
        path = tmp_png_dir / "rand.png"
        cv2.imwrite(str(path), m)
        b = load_and_binarize(path, threshold=100.0)
        unique = set(b.flatten().tolist())
        assert unique.issubset({0, 1})

    def test_channel_extraction_4ch(self, tmp_png_dir):
        img = np.zeros((8, 8, 4), dtype=np.uint8)
        img[2:6, 2:6, 3] = 200   # ecDNA signal in channel 3
        path = tmp_png_dir / "4ch.png"
        cv2.imwrite(str(path), img)

        b = load_and_binarize(path, channel=3)
        assert b[4, 4] == 1    # inside ecDNA blob
        assert b[0, 0] == 0    # background

    def test_channel_extraction_wrong_channel_raises(self, tmp_png_dir):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        path = tmp_png_dir / "3ch.png"
        cv2.imwrite(str(path), img)
        with pytest.raises((ValueError, cv2.error, Exception)):
            load_and_binarize(path, channel=5)

    def test_file_not_found_raises(self, tmp_png_dir):
        with pytest.raises((FileNotFoundError, Exception)):
            load_and_binarize(tmp_png_dir / "nonexistent.png")

    def test_fixture_mask_16x16(self):
        path = FIXTURES / "mask_16x16.png"
        if not path.exists():
            pytest.skip("fixture not available")
        b = load_and_binarize(path, threshold=0.0)
        assert b.dtype == np.uint8
        assert set(b.flatten().tolist()).issubset({0, 1})

    def test_fixture_ecseg_4ch(self):
        path = FIXTURES / "ecseg_8x8.png"
        if not path.exists():
            pytest.skip("fixture not available")
        b = load_and_binarize(path, channel=3)
        assert b[4, 4] == 1   # ecDNA signal blob


# ---------------------------------------------------------------------------
# apply_min_area_filter
# ---------------------------------------------------------------------------

class TestApplyMinAreaFilter:

    def test_removes_tiny_blob(self):
        m = np.zeros((32, 32), dtype=np.uint8)
        m[1, 1] = 1   # 1-px blob — below min_area=3
        m[10:15, 10:15] = 1  # 25-px blob — kept
        out = apply_min_area_filter(m, min_area=3)
        assert out[1, 1] == 0         # removed
        assert out[12, 12] == 1       # kept

    def test_keeps_blob_at_min_area(self):
        m = np.zeros((32, 32), dtype=np.uint8)
        m[5:8, 5:7] = 1  # 6-px blob
        out = apply_min_area_filter(m, min_area=3)
        assert out[6, 6] == 1

    def test_empty_mask_unchanged(self, empty_mask):
        out = apply_min_area_filter((empty_mask // 255).astype(np.uint8), min_area=3)
        assert out.max() == 0

    def test_min_area_zero_keeps_all(self):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[5, 5] = 1
        out = apply_min_area_filter(m, min_area=0)
        assert out[5, 5] == 1

    def test_output_binary_values(self):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[4:8, 4:8] = 1
        out = apply_min_area_filter(m, min_area=3)
        assert set(out.flatten().tolist()).issubset({0, 1})

    def test_all_removed_when_too_small(self):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[5, 5] = 1  # 1-px blob
        out = apply_min_area_filter(m, min_area=5)
        assert out.max() == 0


# ---------------------------------------------------------------------------
# write_binary_mask
# ---------------------------------------------------------------------------

class TestWriteBinaryMask:

    def test_writes_255_for_ones(self, tmp_png_dir):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[4:12, 4:12] = 1
        path = tmp_png_dir / "out.png"
        write_binary_mask(m, path)
        assert path.exists()
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        assert loaded[8, 8]  == 255  # foreground → 255
        assert loaded[0, 0]  == 0    # background → 0

    def test_only_0_and_255_in_output(self, tmp_png_dir):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[4:12, 4:12] = 1
        path = tmp_png_dir / "vals.png"
        write_binary_mask(m, path)
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        unique = set(loaded.flatten().tolist())
        assert unique.issubset({0, 255})

    def test_creates_parent_dir(self, tmp_path):
        m = np.zeros((8, 8), dtype=np.uint8)
        m[2:6, 2:6] = 1
        path = tmp_path / "a" / "b" / "c" / "out.png"
        write_binary_mask(m, path)
        assert path.exists()

    def test_round_trip(self, tmp_png_dir):
        m = np.zeros((16, 16), dtype=np.uint8)
        m[3:8, 3:8] = 1
        path = tmp_png_dir / "rt.png"
        write_binary_mask(m, path)
        loaded = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        # loaded should be 255 where m=1
        expected = (m * 255).astype(np.uint8)
        np.testing.assert_array_equal(loaded, expected)


# ---------------------------------------------------------------------------
# Integration: load → filter → write round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:

    def test_full_round_trip(self, tmp_png_dir):
        # Create a mask with a tiny blob and a larger one
        m = np.zeros((32, 32), dtype=np.uint8)
        m[1, 1] = 255   # 1-px (should be removed by min_area=3)
        m[10:15, 10:15] = 255  # 25-px (should be kept)

        path = tmp_png_dir / "input.png"
        cv2.imwrite(str(path), m)

        binary   = load_and_binarize(path, threshold=0.0)
        filtered = apply_min_area_filter(binary, min_area=3)

        out_path = tmp_png_dir / "output.png"
        write_binary_mask(filtered, out_path)

        result = cv2.imread(str(out_path), cv2.IMREAD_GRAYSCALE)
        assert result[1, 1]   == 0    # tiny blob removed
        assert result[12, 12] == 255  # large blob kept
        assert set(result.flatten().tolist()).issubset({0, 255})
