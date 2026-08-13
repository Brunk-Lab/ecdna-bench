"""
tests/test_sensitivity_smoke.py
=================================
Smoke test for ecdna_bench.cli.sensitivity.

Exercises the fixed precompute-once / sweep-cheap loop on a tiny
synthetic fixture. Verifies:
  1. No ImportError or TypeError (all three A2 import/kwarg/unpack bugs).
  2. Output CSV has the right shape and column set.
  3. The canonical (d_max=20, iou_min=0.1, OR) row produces plausible
     numbers for a perfect-prediction image pair.
  4. The multi-model loop (--model all) produces one combined CSV with
     rows for every model in the registry.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_mask(path: Path, h: int, w: int, blobs: list[tuple[int, int, int]]) -> None:
    """Write a uint8 PNG with filled squares of side `sz` at (r, c)."""
    m = np.zeros((h, w), dtype=np.uint8)
    for r, c, sz in blobs:
        m[r:r + sz, c:c + sz] = 255
    cv2.imwrite(str(path), m)


# ---------------------------------------------------------------------------
# Single-model fixture (legacy behaviour)
# ---------------------------------------------------------------------------

@pytest.fixture()
def sensitivity_fixture(tmp_path: Path):
    """
    Two synthetic images with one prediction directory:
      img_a: GT has 2 blobs; pred is identical (perfect prediction).
      img_b: GT has 1 blob;  pred is empty (all FN).

    Returns (consistency_csv_path, mask_dir, n_images).
    """
    H, W = 64, 64

    gt_dir   = tmp_path / "gt"
    mask_dir = tmp_path / "pred"
    gt_dir.mkdir(); mask_dir.mkdir()

    # img_a — perfect prediction
    _write_mask(gt_dir   / "img_a.png", H, W, [(5, 5, 7), (40, 40, 7)])
    _write_mask(mask_dir / "img_a.png", H, W, [(5, 5, 7), (40, 40, 7)])

    # img_b — empty prediction
    _write_mask(gt_dir   / "img_b.png", H, W, [(30, 30, 7)])
    _write_mask(mask_dir / "img_b.png", H, W, [])

    consistency_csv = tmp_path / "consistency.csv"
    pd.DataFrame({
        "unique_id":             ["img_a", "img_b"],
        "split":                 ["test",  "test"],
        "cell_line":             ["CL1",   "CL1"],
        "count_mask_consistent": [True,    True],
        "gt_fullpath":           [str(gt_dir / "img_a.png"),
                                  str(gt_dir / "img_b.png")],
    }).to_csv(consistency_csv, index=False)

    return consistency_csv, mask_dir, 2


# ---------------------------------------------------------------------------
# Multi-model fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def multi_model_fixture(tmp_path: Path):
    """
    Two synthetic images with three independent prediction directories
    (one per fake "model"). Same two images as ``sensitivity_fixture`` but
    each model gets its own pred dir so we can verify the multi-model loop
    produces one row per (model, image, params) cell.

    Returns (consistency_csv_path, dict[model_key -> mask_dir], n_images).
    """
    H, W = 64, 64

    gt_dir = tmp_path / "gt"
    gt_dir.mkdir()

    # One GT, two images.
    _write_mask(gt_dir / "img_a.png", H, W, [(5, 5, 7), (40, 40, 7)])
    _write_mask(gt_dir / "img_b.png", H, W, [(30, 30, 7)])

    # Three "model" prediction dirs with deliberately different behaviour
    # so the aggregated CSV must distinguish them.
    mask_dirs: dict[str, Path] = {}

    # model_perfect: identical to GT on both images
    md = tmp_path / "pred_perfect"; md.mkdir()
    _write_mask(md / "img_a.png", H, W, [(5, 5, 7), (40, 40, 7)])
    _write_mask(md / "img_b.png", H, W, [(30, 30, 7)])
    mask_dirs["model_perfect"] = md

    # model_partial: img_a has only one of two blobs; img_b empty
    md = tmp_path / "pred_partial"; md.mkdir()
    _write_mask(md / "img_a.png", H, W, [(5, 5, 7)])
    _write_mask(md / "img_b.png", H, W, [])
    mask_dirs["model_partial"] = md

    # model_empty: both predictions empty
    md = tmp_path / "pred_empty"; md.mkdir()
    _write_mask(md / "img_a.png", H, W, [])
    _write_mask(md / "img_b.png", H, W, [])
    mask_dirs["model_empty"] = md

    consistency_csv = tmp_path / "consistency.csv"
    pd.DataFrame({
        "unique_id":             ["img_a", "img_b"],
        "split":                 ["test",  "test"],
        "cell_line":             ["CL1",   "CL1"],
        "count_mask_consistent": [True,    True],
        "gt_fullpath":           [str(gt_dir / "img_a.png"),
                                  str(gt_dir / "img_b.png")],
    }).to_csv(consistency_csv, index=False)

    return consistency_csv, mask_dirs, 2


# ---------------------------------------------------------------------------
# Original tests (unchanged behaviour) — single model path
# ---------------------------------------------------------------------------

def test_sensitivity_imports_and_runs(sensitivity_fixture, tmp_path):
    """
    Guards the A2 import/kwarg/unpack bugs against the *current* public
    matching API (evaluation.matching), not the removed private helpers.
    """
    consistency_csv, mask_dir, n_images = sensitivity_fixture

    from ecdna_bench.benchmark.run import _extract_objects, _load_gray
    from ecdna_bench.evaluation.matching import (
        precompute_pairwise,
        resolve_matching_from_pairwise,
    )

    gt = _load_gray(consistency_csv.parent / "gt" / "img_a.png")
    assert gt is not None

    gt_objs   = _extract_objects(gt, min_area=3)
    pred_objs = _extract_objects(gt, min_area=3)  # identical → perfect match

    pw  = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=100.0)
    res = resolve_matching_from_pairwise(
        pw, d_max=20.0, min_iou=0.1, alpha=0.5, policy="OR",
    )

    assert res.tp == len(gt_objs), "perfect prediction must yield tp == n_gt"
    assert res.fp == 0
    assert res.fn == 0


def test_sensitivity_perfect_prediction_values(sensitivity_fixture, tmp_path):
    """
    img_a (perfect prediction, 2 GT): tp=2, fp=0, fn=0.
    img_b (empty prediction, 1 GT):   tp=0, fp=0, fn=1.
    Aggregated: tp=2, fp=0, fn=1.
    """
    consistency_csv, mask_dir, _ = sensitivity_fixture

    from ecdna_bench.benchmark.run import _extract_objects, _load_gray
    from ecdna_bench.evaluation.matching import (
        precompute_pairwise,
        resolve_matching_from_pairwise,
    )

    df = pd.read_csv(consistency_csv)
    tp_total = fp_total = fn_total = 0

    for _, row in df.iterrows():
        uid       = str(row["unique_id"])
        gt_path   = Path(str(row["gt_fullpath"]))
        pred_path = mask_dir / f"{uid}.png"

        gt   = _load_gray(gt_path)
        pred = _load_gray(pred_path)
        if pred is None:
            pred = np.zeros_like(gt)

        gt_objs   = _extract_objects(gt,   min_area=3)
        pred_objs = _extract_objects(pred, min_area=3)

        pw  = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=100.0)
        res = resolve_matching_from_pairwise(
            pw, d_max=20.0, min_iou=0.1, alpha=0.5, policy="OR",
        )
        tp_total += res.tp
        fp_total += res.fp
        fn_total += res.fn

    assert tp_total == 2, f"expected tp=2, got {tp_total}"
    assert fp_total == 0, f"expected fp=0, got {fp_total}"
    assert fn_total == 1, f"expected fn=1 (img_b empty pred), got {fn_total}"

def test_sensitivity_grid_constants():
    """Canonical grid values match PROJECT_RULES.md §2 exactly."""
    import ecdna_bench.cli.sensitivity as sens_mod

    assert sens_mod._D_MAX_GRID   == [5, 10, 15, 20, 25, 30, 40, 50, 75, 100]
    assert sens_mod._IOU_MIN_GRID == [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    assert set(sens_mod._POLICIES) == {"or", "and"}


# ---------------------------------------------------------------------------
# New tests — multi-model path
# ---------------------------------------------------------------------------

def test_sweep_matching_grid_carries_model_extras():
    """
    The DataFrame returned by sweep_matching_grid must propagate every key
    in `extra` — including `model` — so that we can groupby model later.
    """
    from ecdna_bench.evaluation.matching import precompute_pairwise
    from ecdna_bench.evaluation.sensitivity import sweep_matching_grid
    from ecdna_bench.benchmark.run import _extract_objects

    H, W = 64, 64
    gt_mask   = np.zeros((H, W), dtype=np.uint8)
    pred_mask = np.zeros((H, W), dtype=np.uint8)
    gt_mask[5:12, 5:12]  = 255
    pred_mask[5:12, 5:12] = 255

    gt_objs   = _extract_objects(gt_mask,   min_area=3)
    pred_objs = _extract_objects(pred_mask, min_area=3)

    pw = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=100.0)

    items = [
        ("img_a", pw, {"model": "ModelOne", "split": "test"}),
        ("img_b", pw, {"model": "ModelTwo", "split": "test"}),
    ]

    df = sweep_matching_grid(
        items,
        d_max_grid=[20],
        min_iou_grid=[0.1],
        policies=["OR"],
        alpha=0.5,
    )

    assert "model" in df.columns, "model column must be carried through"
    assert set(df["model"]) == {"ModelOne", "ModelTwo"}
    # 2 images × 1 d_max × 1 iou_min × 1 policy = 2 rows
    assert len(df) == 2


def test_sensitivity_multi_model_combined_csv(multi_model_fixture, tmp_path, monkeypatch):
    """
    End-to-end multi-model sweep: build the items list manually for three
    fake models on the same images, run sweep_matching_grid once, and
    verify the aggregate CSV has one row per model × policy × d_max × iou_min
    with sensible values.

    This mirrors what the CLI does in the --model all branch.
    """
    consistency_csv, mask_dirs, n_images = multi_model_fixture

    import ecdna_bench.cli.sensitivity as sens_mod
    monkeypatch.setattr(sens_mod, "_D_MAX_GRID",   [20])
    monkeypatch.setattr(sens_mod, "_IOU_MIN_GRID", [0.1])
    monkeypatch.setattr(sens_mod, "_POLICIES",     ["or"])

    from ecdna_bench.benchmark.run import _extract_objects
    from ecdna_bench.evaluation.matching import precompute_pairwise
    from ecdna_bench.evaluation.sensitivity import sweep_matching_grid

    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]

    max_precompute_dist = float(max(sens_mod._D_MAX_GRID))
    all_items: list[tuple] = []

    for model_key, mask_dir in mask_dirs.items():
        for _, row in df.iterrows():
            uid       = str(row["unique_id"])
            gt_path   = Path(str(row["gt_fullpath"]))
            pred_path = mask_dir / f"{uid}.png"

            gt   = cv2.imread(str(gt_path),   cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(str(pred_path), cv2.IMREAD_GRAYSCALE)
            if pred is None:
                pred = np.zeros_like(gt)

            gt_objs   = _extract_objects(gt,   min_area=3)
            pred_objs = _extract_objects(pred, min_area=3)

            pw = precompute_pairwise(
                pred_objs, gt_objs, max_precompute_dist=max_precompute_dist,
            )
            all_items.append((
                uid,
                pw,
                {
                    "model": model_key,
                    "split": "test",
                    "cell_line": str(row.get("cell_line", "unknown")),
                    "uid": uid,
                },
            ))

    tidy_df = sweep_matching_grid(
        all_items,
        d_max_grid=sens_mod._D_MAX_GRID,
        min_iou_grid=sens_mod._IOU_MIN_GRID,
        policies=[p.upper() for p in sens_mod._POLICIES],
        alpha=0.5,
    )

    assert "model" in tidy_df.columns
    assert set(tidy_df["model"]) == set(mask_dirs.keys()), (
        "every model_key must show up in the per-image sweep"
    )

    # 3 models × 2 images × 1 × 1 × 1 = 6 rows
    expected_tidy_rows = len(mask_dirs) * n_images
    assert len(tidy_df) == expected_tidy_rows, (
        f"expected {expected_tidy_rows} per-image rows, got {len(tidy_df)}"
    )

    # ----------- aggregate exactly like the CLI does -----------
    tidy_df["policy"] = tidy_df["policy"].astype(str).str.lower()
    group_cols = ["model", "split", "policy", "d_max", "min_iou"]
    sweep_df = (
        tidy_df.groupby(group_cols, as_index=False)[
            ["tp", "fp", "fn", "ignored"]
        ]
        .sum()
        .rename(columns={"min_iou": "iou_min"})
    )
    tp = sweep_df["tp"].astype(float)
    fp = sweep_df["fp"].astype(float)
    fn = sweep_df["fn"].astype(float)
    sweep_df["precision"] = np.where((tp + fp) > 0, tp / (tp + fp), 0.0)
    sweep_df["recall"]    = np.where((tp + fn) > 0, tp / (tp + fn), 0.0)
    sweep_df["f1"]        = np.where(
        (2 * tp + fp + fn) > 0, (2 * tp) / (2 * tp + fp + fn), 0.0
    )

    out_dir = tmp_path / "frozen_results" / "sensitivity"
    out_dir.mkdir(parents=True)
    out_csv = out_dir / "sensitivity_sweep.csv"
    sweep_df.to_csv(out_csv, index=False)

    assert out_csv.exists()
    result = pd.read_csv(out_csv)

    # 3 models × 1 split × 1 policy × 1 d_max × 1 iou_min = 3 rows
    assert len(result) == len(mask_dirs)
    assert set(result["model"]) == set(mask_dirs.keys())

    required_cols = {"model", "split", "policy", "d_max", "iou_min",
                     "tp", "fp", "fn", "ignored", "precision", "recall", "f1"}
    assert required_cols.issubset(result.columns)

    # Spot-check the per-model values:
    # model_perfect: 3 GT total, 3 TP, 0 FP, 0 FN → f1 == 1.0
    # model_empty:   3 GT total, 0 TP, 0 FP, 3 FN → f1 == 0.0
    perfect = result.loc[result["model"] == "model_perfect"].iloc[0]
    empty   = result.loc[result["model"] == "model_empty"  ].iloc[0]
    assert perfect["tp"] == 3 and perfect["fp"] == 0 and perfect["fn"] == 0
    assert pytest.approx(perfect["f1"]) == 1.0
    assert empty["tp"]   == 0 and empty["fp"]   == 0 and empty["fn"]   == 3
    assert pytest.approx(empty["f1"])   == 0.0


def test_cli_accepts_model_all_argument():
    """
    The argparse layer must accept --model all without complaint and the
    main() must resolve it to the full registry. Verified by parsing args
    in isolation; we don't run the full pipeline here.
    """
    import sys
    import ecdna_bench.cli.sensitivity as sens_mod

    argv_backup = sys.argv[:]
    try:
        sys.argv = ["sensitivity", "--model", "all", "--split", "test"]
        args = sens_mod.parse_args()
        assert args.model == "all"
        assert args.split == "test"
    finally:
        sys.argv = argv_backup


def test_registry_contains_the_expected_models():
    """
    The registry holds seven entries: the six models reported in the paper,
    plus `classical_before_opt`, which exists only for the before/after
    Bayesian-optimisation comparison and is not a benchmarked model.

    Asserting the key set (rather than a bare count) also guards the locked
    display names used in figures, tables and captions.
    """
    from ecdna_bench.benchmark.registry import MODEL_REGISTRY, MODEL_ORDER

    expected_keys = {
        "classical",              # Classic (after opt)   — benchmarked
        "classical_before_opt",   # Classic (before opt)  — comparison only
        "label_engine",           # benchmarked
        "ecseg",                  # benchmarked
        "mia",                    # benchmarked
        "eccount_mask",           # benchmarked
        "eccount_peaks",          # benchmarked
    }
    assert set(MODEL_REGISTRY) == expected_keys, (
        f"MODEL_REGISTRY drifted. Expected {sorted(expected_keys)}, "
        f"found {sorted(MODEL_REGISTRY)}"
    )

    # The six models reported in the paper, with locked spellings.
    paper_six = {
        "Classic (after opt)",
        "Label Engine",
        "ecSeg",
        "MIA",
        "ecCount (threshold mask)",
        "ecCount (peaks)",
    }
    display_names = {spec.name for spec in MODEL_REGISTRY.values()}
    assert paper_six <= display_names, (
        f"Locked display names missing: {sorted(paper_six - display_names)}"
    )

    # MODEL_ORDER drives figure and table ordering — it must cover the registry.
    assert set(MODEL_ORDER) == display_names