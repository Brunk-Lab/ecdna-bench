"""
tests/test_run_consistency.py
==============================
Assert that the benchmark runner (_eval_one) and the evaluation stack
(match_objects / resolve_matching_from_pairwise) produce identical
object-level metrics on a synthetic fixture.

This test guards against regressions if either side is modified
independently after A9.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from ecdna_bench.evaluation.matching import match_objects, precompute_pairwise, resolve_matching_from_pairwise
from ecdna_bench.evaluation.metrics import object_metrics_from_counts


# ---------------------------------------------------------------------------
# Shared fixture helpers
# ---------------------------------------------------------------------------

def _make_mask(h: int, w: int, blobs: list[tuple[int, int, int]]) -> np.ndarray:
    """uint8 mask with filled squares of side sz at (r, c)."""
    m = np.zeros((h, w), dtype=np.uint8)
    for r, c, sz in blobs:
        m[r:r + sz, c:c + sz] = 255
    return m


def _write_mask(path: Path, mask: np.ndarray) -> None:
    cv2.imwrite(str(path), mask)


# Frozen eval params matching EvalConfig defaults
_D_MAX   = 20.0
_IOU_MIN = 0.1
_ALPHA   = 0.5
_MIN_AREA = 3


# ---------------------------------------------------------------------------
# Core consistency test
# ---------------------------------------------------------------------------

@pytest.fixture()
def two_image_fixture(tmp_path: Path):
    """
    Two images:
      img_a: 2 GT blobs, 2 identical pred blobs (perfect match).
      img_b: 1 GT blob,  0 pred blobs (all FN).
    Returns list of (uid, gt_path, pred_path) tuples.
    """
    H, W = 64, 64
    gt_dir   = tmp_path / "gt"
    pred_dir = tmp_path / "pred"
    gt_dir.mkdir(); pred_dir.mkdir()

    images = [
        ("img_a", [(5, 5, 7), (40, 40, 7)], [(5, 5, 7), (40, 40, 7)]),
        ("img_b", [(30, 30, 7)],             []),
    ]
    fixtures = []
    for uid, gt_blobs, pred_blobs in images:
        gt_p   = gt_dir   / f"{uid}.png"
        pred_p = pred_dir / f"{uid}.png"
        _write_mask(gt_p,   _make_mask(H, W, gt_blobs))
        _write_mask(pred_p, _make_mask(H, W, pred_blobs))
        fixtures.append((uid, gt_p, pred_p))
    return fixtures


def _eval_one_via_runner(uid, gt_path, pred_path, mode):
    """Call _eval_one (the worker function) and return the row for `mode`."""
    from ecdna_bench.benchmark.run import _eval_one
    args = (
        uid, "test", "CL1",
        str(gt_path), str(pred_path),
        "TestModel",
        _D_MAX, _IOU_MIN, _ALPHA, _MIN_AREA, 0.5,
    )
    return _eval_one(args)[mode]


def _eval_one_via_stack(gt_path, pred_path, mode):
    """Call the evaluation stack directly and return metrics dict."""
    from ecdna_bench.benchmark.run import _extract_objects, _load_gray

    gt   = _load_gray(gt_path)
    pred = _load_gray(pred_path)
    if pred is None:
        pred = np.zeros_like(gt)

    gt_objs   = _extract_objects(gt,   _MIN_AREA)
    pred_objs = _extract_objects(pred, _MIN_AREA)

    pw = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=_D_MAX)
    result = resolve_matching_from_pairwise(
        pw, d_max=_D_MAX, min_iou=_IOU_MIN, alpha=_ALPHA, policy=mode.upper()
    )
    m = object_metrics_from_counts(
        tp=result.tp, fp=result.fp, fn=result.fn, ignored=result.ignored
    )
    return {"tp": result.tp, "fp": result.fp, "fn": result.fn,
            "ignored": result.ignored, **m}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode", ["or", "and"])
def test_runner_matches_eval_stack_object_metrics(two_image_fixture, mode):
    """
    For every (image, mode) combination, _eval_one and the evaluation
    stack must produce identical tp/fp/fn/ignored and
    precision/recall/f1.
    """
    for uid, gt_path, pred_path in two_image_fixture:
        row   = _eval_one_via_runner(uid, gt_path, pred_path, mode)
        stack = _eval_one_via_stack(gt_path, pred_path, mode)

        assert row["obj_tp"]      == stack["tp"],      f"uid={uid} mode={mode}: tp mismatch"
        assert row["obj_fp"]      == stack["fp"],      f"uid={uid} mode={mode}: fp mismatch"
        assert row["obj_fn"]      == stack["fn"],      f"uid={uid} mode={mode}: fn mismatch"
        assert row["obj_ignored"] == stack["ignored"], f"uid={uid} mode={mode}: ignored mismatch"

        assert abs(row["obj_precision"] - stack["precision"]) < 1e-9, \
            f"uid={uid} mode={mode}: precision mismatch"
        assert abs(row["obj_recall"]    - stack["recall"])    < 1e-9, \
            f"uid={uid} mode={mode}: recall mismatch"
        assert abs(row["obj_f1"]        - stack["f1"])        < 1e-9, \
            f"uid={uid} mode={mode}: f1 mismatch"


@pytest.mark.parametrize("mode", ["or", "and"])
def test_perfect_prediction_gives_f1_one(two_image_fixture, mode):
    """img_a has identical GT and pred — F1 must be 1.0."""
    uid, gt_path, pred_path = two_image_fixture[0]  # img_a
    row = _eval_one_via_runner(uid, gt_path, pred_path, mode)
    assert abs(row["obj_f1"] - 1.0) < 1e-9, f"mode={mode}: expected f1=1.0 for perfect prediction"


@pytest.mark.parametrize("mode", ["or", "and"])
def test_empty_prediction_gives_f1_zero(two_image_fixture, mode):
    """img_b has 1 GT and 0 pred — F1 must be 0.0, fn=1."""
    uid, gt_path, pred_path = two_image_fixture[1]  # img_b
    row = _eval_one_via_runner(uid, gt_path, pred_path, mode)
    assert row["obj_f1"] == 0.0,  f"mode={mode}: expected f1=0.0 for empty prediction"
    assert row["obj_fn"] == 1,    f"mode={mode}: expected fn=1"
    assert row["obj_tp"] == 0,    f"mode={mode}: expected tp=0"


def test_no_local_matching_functions_in_run():
    """
    Guard: _precompute_pairwise and _match_for_mode must not exist in
    ecdna_bench.benchmark.run after A9.
    """
    import ecdna_bench.benchmark.run as run_mod
    assert not hasattr(run_mod, "_precompute_pairwise"), \
        "_precompute_pairwise still present in benchmark.run — A9 incomplete"
    assert not hasattr(run_mod, "_match_for_mode"), \
        "_match_for_mode still present in benchmark.run — A9 incomplete"