"""
tests/conftest.py
==================
Shared pytest fixtures for the ecdna-bench test suite.

All fixtures use synthetic numpy arrays and temporary directories.
No disk paths to real images are used anywhere in the test suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Image / mask fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tiny_gt_mask() -> np.ndarray:
    """32×32 uint8 mask with two 5×5 blobs (areas = 25 each).

    Blob 1: rows 4–8,   cols 4–8
    Blob 2: rows 20–24, cols 20–24
    """
    m = np.zeros((32, 32), dtype=np.uint8)
    m[4:9, 4:9]   = 255
    m[20:25, 20:25] = 255
    return m


@pytest.fixture()
def tiny_pred_mask(tiny_gt_mask: np.ndarray) -> np.ndarray:
    """Identical copy of tiny_gt_mask (perfect prediction)."""
    return tiny_gt_mask.copy()


@pytest.fixture()
def single_blob_mask() -> np.ndarray:
    """32×32 mask with a single 6×6 blob (area=36)."""
    m = np.zeros((32, 32), dtype=np.uint8)
    m[10:16, 10:16] = 255
    return m


@pytest.fixture()
def empty_mask() -> np.ndarray:
    """32×32 all-zero mask."""
    return np.zeros((32, 32), dtype=np.uint8)


@pytest.fixture()
def full_mask() -> np.ndarray:
    """32×32 all-255 mask."""
    return np.full((32, 32), 255, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Object list fixtures (standard dict format)
# ---------------------------------------------------------------------------

def _make_obj(cy: float, cx: float, h: int = 5, w: int = 5,
               shape: Tuple[int, int] = (32, 32)) -> Dict:
    """Build one standard object dict."""
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


@pytest.fixture()
def two_gt_objects() -> List[Dict]:
    """Two GS objects at (6, 6) and (22, 22) on a 32×32 canvas."""
    return [_make_obj(6, 6), _make_obj(22, 22)]


@pytest.fixture()
def two_pred_objects() -> List[Dict]:
    """Two pred objects at (6, 6) and (22, 22) — perfect overlap with GS."""
    return [_make_obj(6, 6), _make_obj(22, 22)]


@pytest.fixture()
def one_pred_object() -> List[Dict]:
    """Single pred object at (6, 6) — matches first GS only."""
    return [_make_obj(6, 6)]


@pytest.fixture()
def three_preds_one_gt() -> Tuple[List[Dict], List[Dict]]:
    """3 predictions clustered around 1 GS at (16, 16).

    Under OR policy (d_max large): tp=1, ignored=2.
    """
    gt   = [_make_obj(16, 16)]
    pred = [_make_obj(16, 16), _make_obj(17, 17), _make_obj(15, 15)]
    return pred, gt


@pytest.fixture()
def no_overlap_objects() -> Tuple[List[Dict], List[Dict]]:
    """GS at (5, 5), pred at (27, 27) — completely separated."""
    gt   = [_make_obj(5, 5)]
    pred = [_make_obj(27, 27)]
    return pred, gt


# ---------------------------------------------------------------------------
# Probability map fixtures (for postprocess tests)
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_peak_prob_map() -> np.ndarray:
    """64×64 probability map with two sharp peaks at (10,10) and (50,50)."""
    pm = np.zeros((64, 64), dtype=np.float32)
    pm[10, 10] = 0.95
    pm[50, 50] = 0.85
    return pm


@pytest.fixture()
def empty_prob_map() -> np.ndarray:
    return np.zeros((64, 64), dtype=np.float32)


# ---------------------------------------------------------------------------
# Temporary directory fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_png_dir(tmp_path: Path) -> Path:
    """A temporary directory for PNG I/O tests."""
    d = tmp_path / "pngs"
    d.mkdir()
    return d
