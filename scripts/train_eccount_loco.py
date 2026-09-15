#!/usr/bin/env python
"""
train_eccount_loco.py — leave-one-cell-line-out training for ecCount.

Trains ecCount on three cell lines and leaves the fourth entirely unseen, then
writes everything the downstream inference and scoring steps need. Four runs,
one per held-out line, plus an optional size-matched control.

Every hyperparameter is read from ``configs/default.yaml`` by the same code
path the released CLI uses, so the only difference between these runs and the
paper run is which images are in the training set.

Usage
-----
    # Verify the splits before spending any GPU time. Do this first.
    python scripts/train_eccount_loco.py --hold-out all --dry-run

    # One run.
    python scripts/train_eccount_loco.py --hold-out NCI-H2170

    # Size-matched control for the NCI-H2170 case (see WHY below).
    python scripts/train_eccount_loco.py --hold-out NCI-H2170 --size-matched-control

WHY THE SIZE-MATCHED CONTROL EXISTS
-----------------------------------
NCI-H2170 is 888 of the 1,145 benchmark images. Holding it out cuts the
training set from 800 images to 179 — a 4.5-fold reduction. A drop in
performance on held-out NCI-H2170 therefore has two candidate causes that
cannot be separated from the four leave-one-out runs alone:

    (1) the model has never seen this cell line, or
    (2) the model was trained on a fifth as much data.

The control trains on 179 images drawn from all four cell lines and is
evaluated on the same held-out images as the matching leave-one-out run. The
difference between the two is attributable to cell-line shift alone.

Without this control the NCI-H2170 row is not interpretable, and it is the row
a reviewer will look at first because it is the largest cell line.

OUTPUTS (per run, under paths.eccount_loco_dir / <run_id>/)
-----------------------------------------------------------
    best_model.pt          the trained checkpoint
    train_history.csv      per-epoch losses, written by the trainer
    split_composition.csv  exact image and object counts per line and split
    eval_metadata.csv      the held-out rows only; drives inference + scoring
    run_config.yaml        self-contained config for the downstream CLIs
    run_manifest.json      provenance: seed, sizes, guards passed, timestamps

A NOTE ON run_config.yaml
-------------------------
``ecdna_bench.cli._common.load_config`` merges ``paths.local.yaml`` from the
directory *of the config file it is given*, and the local file WINS. A per-run
config placed in ``configs/`` would therefore have every path silently
overwritten by the global paths.local.yaml. That is why run_config.yaml is
written into the run directory instead, and why this script asserts that no
paths.local.yaml exists beside it.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger("loco")

# ---------------------------------------------------------------------------
# Locked expectations. These are guards, not parameters. If one fires, stop and
# find out why — do not relax it.
# ---------------------------------------------------------------------------
EXPECTED_N_IMAGES = 1145
EXPECTED_GT_TOTAL = 228_039
EXPECTED_CELL_LINES = ("COLO320DM", "NCI-H2170", "NCI-H716", "SNU16")
EXPECTED_SPLIT_SIZES = {"train": 800, "val": 170, "test": 175}

# Per-cell-line composition, computed from the frozen consistency CSV.
# The dry run must reproduce this table exactly.
EXPECTED_COMPOSITION = {
    #  cell line     train  val  test   all   GS objects
    "COLO320DM":  (   44,    9,   11,    64,    2_943),
    "NCI-H2170":  (  621,  133,  134,   888,  176_883),
    "NCI-H716":   (   49,   10,   11,    70,   13_340),
    "SNU16":      (   86,   18,   19,   123,   34_873),
}


def slugify(cell_line: str) -> str:
    """Filesystem-safe run identifier. 'NCI-H2170' -> 'nci_h2170'."""
    return cell_line.lower().replace("-", "_").replace(" ", "_")


# ---------------------------------------------------------------------------
# Split construction — pure, testable, no torch import
# ---------------------------------------------------------------------------

def load_consistent_metadata(consistency_csv: Path) -> pd.DataFrame:
    """Read the consistency CSV and apply the one filter the paper applies."""
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)].reset_index(drop=True)
    return df


def check_invariants(df: pd.DataFrame, strict: bool = True) -> List[str]:
    """Verify the metadata matches the locked benchmark. Returns failures."""
    failures: List[str] = []

    if len(df) != EXPECTED_N_IMAGES:
        failures.append(
            f"image count is {len(df)}, expected {EXPECTED_N_IMAGES}"
        )

    gt_total = int(df["ecDNA_gt"].sum())
    if gt_total != EXPECTED_GT_TOTAL:
        failures.append(
            f"gold-standard object total is {gt_total:,}, "
            f"expected {EXPECTED_GT_TOTAL:,}"
        )

    lines = tuple(sorted(df["cell_line"].unique()))
    if lines != tuple(sorted(EXPECTED_CELL_LINES)):
        failures.append(f"cell lines are {lines}, expected {EXPECTED_CELL_LINES}")

    for split, n_expected in EXPECTED_SPLIT_SIZES.items():
        n = int((df["split"] == split).sum())
        if n != n_expected:
            failures.append(
                f"split '{split}' has {n} images, expected {n_expected}"
            )

    for line, (tr, va, te, tot, gt) in EXPECTED_COMPOSITION.items():
        sub = df[df["cell_line"] == line]
        got = (
            int((sub["split"] == "train").sum()),
            int((sub["split"] == "val").sum()),
            int((sub["split"] == "test").sum()),
            int(len(sub)),
            int(sub["ecDNA_gt"].sum()),
        )
        if got != (tr, va, te, tot, gt):
            failures.append(
                f"{line}: composition is {got}, expected {(tr, va, te, tot, gt)}"
            )

    if failures and strict:
        for f in failures:
            logger.error("INVARIANT FAILED: %s", f)
        logger.error(
            "The metadata does not match the locked benchmark. Nothing was "
            "trained. Resolve the discrepancy before re-running — do not pass "
            "--no-strict to work around it."
        )
        sys.exit(2)

    return failures


def build_loco_splits(
    df: pd.DataFrame,
    hold_out: str,
    size_matched_control: bool = False,
    control_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """Build (train, val, eval, info) frames for one leave-one-out run.

    Standard run
        train = 'train' rows of the three included lines
        val   = 'val'   rows of the three included lines
        eval  = every row of the held-out line

    Size-matched control
        train = a stratified sample from the 'train' rows of ALL FOUR lines,
                sized to match the standard run's training set
        val   = 'val' rows of ALL FOUR lines, sized to match likewise
        eval  = the 'test' rows of the held-out line only, because the control
                has seen some of that line's train and val images

    The control's eval set is a strict subset of the standard run's eval set,
    so the two are compared on the held-out line's test rows in both cases.
    """
    if hold_out not in set(df["cell_line"]):
        raise ValueError(
            f"cell line {hold_out!r} not present. "
            f"Available: {sorted(df['cell_line'].unique())}"
        )

    included = df[df["cell_line"] != hold_out]
    held = df[df["cell_line"] == hold_out]

    # Sizes are always defined by the standard leave-one-out run, so the
    # control matches it exactly.
    n_train = int((included["split"] == "train").sum())
    n_val = int((included["split"] == "val").sum())

    if not size_matched_control:
        train_df = included[included["split"] == "train"].reset_index(drop=True)
        val_df = included[included["split"] == "val"].reset_index(drop=True)
        eval_df = held.reset_index(drop=True)
        eval_scope = "all rows of the held-out line"
    else:
        train_df = _stratified_sample(
            df[df["split"] == "train"], n_train, control_seed
        )
        val_df = _stratified_sample(
            df[df["split"] == "val"], n_val, control_seed + 1
        )
        eval_df = held[held["split"] == "test"].reset_index(drop=True)
        eval_scope = "test rows of the held-out line only"

    # Guards on the constructed splits.
    if not size_matched_control:
        leaked = set(train_df["cell_line"]) | set(val_df["cell_line"])
        assert hold_out not in leaked, (
            f"{hold_out} leaked into training data — split construction is broken"
        )
    overlap = set(train_df["unique_id"]) & set(val_df["unique_id"])
    assert not overlap, f"{len(overlap)} images appear in both train and val"
    if not size_matched_control:
        ev_overlap = set(eval_df["unique_id"]) & (
            set(train_df["unique_id"]) | set(val_df["unique_id"])
        )
        assert not ev_overlap, (
            f"{len(ev_overlap)} evaluation images were trained on"
        )

    info = {
        "hold_out": hold_out,
        "size_matched_control": size_matched_control,
        "control_seed": control_seed if size_matched_control else None,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_eval": len(eval_df),
        "eval_scope": eval_scope,
        "train_gt_objects": int(train_df["ecDNA_gt"].sum()),
        "val_gt_objects": int(val_df["ecDNA_gt"].sum()),
        "eval_gt_objects": int(eval_df["ecDNA_gt"].sum()),
        "train_cell_lines": sorted(train_df["cell_line"].unique()),
    }
    return train_df, val_df, eval_df, info


def _stratified_sample(pool: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Sample *n* rows from *pool*, proportional to cell-line frequency.

    Largest-remainder allocation, so the quotas sum to exactly *n* and no line
    is silently dropped. Deterministic given *seed*.
    """
    if n > len(pool):
        raise ValueError(f"cannot sample {n} rows from a pool of {len(pool)}")

    counts = pool["cell_line"].value_counts()
    exact = counts / counts.sum() * n
    quota = exact.apply(int).to_dict()
    remainder = n - sum(quota.values())
    # Hand out the leftover seats to the largest fractional parts.
    order = (exact - exact.apply(int)).sort_values(ascending=False).index
    for line in list(order)[:remainder]:
        quota[line] += 1

    parts = [
        pool[pool["cell_line"] == line].sample(
            n=k, random_state=seed + i, replace=False
        )
        for i, (line, k) in enumerate(sorted(quota.items()))
        if k > 0
    ]
    out = pd.concat(parts).sort_values("unique_id").reset_index(drop=True)
    assert len(out) == n, f"stratified sample produced {len(out)} rows, wanted {n}"
    return out


def composition_table(
    train_df: pd.DataFrame, val_df: pd.DataFrame, eval_df: pd.DataFrame
) -> pd.DataFrame:
    """Per-cell-line image and object counts for the three constructed sets."""
    rows = []
    for role, frame in (("train", train_df), ("val", val_df), ("eval", eval_df)):
        if frame.empty:
            continue
        g = frame.groupby("cell_line")["ecDNA_gt"].agg(
            n_images="size", gt_objects="sum"
        )
        for line, r in g.iterrows():
            rows.append(
                {
                    "role": role,
                    "cell_line": line,
                    "n_images": int(r["n_images"]),
                    "gt_objects": int(r["gt_objects"]),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run-directory scaffolding
# ---------------------------------------------------------------------------

def write_run_config(
    base_cfg: dict,
    run_dir: Path,
    eval_metadata_csv: Path,
    tag: str = "",
) -> Path:
    """Write a self-contained config for run_eccount and benchmark.

    The merged base config is dumped in full, then the keys that must point at
    this run are overridden. Nothing is inherited at load time — see the module
    docstring for why that matters.

    *tag* names a second evaluation of the same checkpoint against a different
    image set, with its own mask and results directories so the two never mix.
    """
    import copy
    import yaml

    cfg = copy.deepcopy(base_cfg)
    paths = cfg.setdefault("paths", {})

    sfx = f"_{tag}" if tag else ""
    masks_root = run_dir / f"masks{sfx}"
    paths["consistency_csv"] = str(eval_metadata_csv)
    paths["eccount_checkpoint"] = str(run_dir / "best_model.pt")
    paths["eccount_threshold_masks"] = str(masks_root / "eccount_threshold")
    paths["eccount_peaks_masks"] = str(masks_root / "eccount_peaks")
    paths["frozen_results_dir"] = str(run_dir / f"results{sfx}")
    paths["harmonized_masks_dir"] = str(masks_root)

    # Never let a run write into the locked release tree.
    for key in (
        "eccount_threshold_masks",
        "eccount_peaks_masks",
        "frozen_results_dir",
        "harmonized_masks_dir",
    ):
        p = str(paths[key])
        if f"{'release'}/" in p.replace("\\", "/") and "eccount_loco" not in p:
            raise RuntimeError(
                f"refusing to run: paths.{key} points into the locked release "
                f"tree ({p})"
            )

    out = run_dir / f"run_config{sfx}.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=False)

    # load_config merges paths.local.yaml from the config's own directory and
    # the local file wins. There must not be one here.
    stray = run_dir / "paths.local.yaml"
    if stray.exists():
        raise RuntimeError(
            f"{stray} exists and would silently override every path in "
            f"run_config.yaml. Remove it."
        )
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one(
    base_cfg: dict,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    run_dir: Path,
    resume: Optional[str],
) -> dict:
    """Train one ecCount model.

    The config-reading block below is copied from
    ``ecdna_bench.cli.train_eccount`` verbatim so that every hyperparameter is
    identical to the paper run. Do not "tidy" it — the nesting levels and the
    amp default are both load-bearing.
    """
    from ecdna_bench.eccount.dataset import DatasetConfig, EcCountDataset
    from ecdna_bench.eccount.losses import LossConfig
    from ecdna_bench.eccount.model import ModelConfig
    from ecdna_bench.eccount.targets import SoftTargetConfig
    from ecdna_bench.eccount.train import TrainConfig, train_eccount

    eccount_cfg = base_cfg.get("eccount", {})
    train_sub = eccount_cfg.get("train", {})
    target_sub = eccount_cfg.get("targets", {})
    loss_sub = eccount_cfg.get("loss", {})
    aug_sub = eccount_cfg.get("augmentation", {})
    input_size = eccount_cfg.get("input_size", [1024, 1224])

    image_h, image_w = int(input_size[0]), int(input_size[1])
    sigma = float(target_sub.get("sigma", 1.0))

    loss_cfg = LossConfig(
        pos_weight=float(loss_sub.get("pos_weight", 20.0)),
        bce_weight=float(loss_sub.get("bce_weight", 1.0)),
        dice_weight=float(loss_sub.get("dice_weight", 1.0)),
        smooth=float(loss_sub.get("dice_smooth", loss_sub.get("smooth", 1e-6))),
    )

    train_cfg = TrainConfig(
        out_dir=run_dir,
        max_epochs=int(train_sub.get("epochs", 70)),
        batch_size=int(train_sub.get("batch_size", 2)),
        lr=float(train_sub.get("lr", 1e-4)),
        weight_decay=float(train_sub.get("weight_decay", 0.0)),
        image_h=image_h,
        image_w=image_w,
        sigma=sigma,
        device=str(train_sub.get("device", "cuda")),
        num_workers=int(train_sub.get("num_workers", train_sub.get("n_workers", 4))),
        seed=int(train_sub.get("seed", 42)),
        # amp defaults to True in TrainConfig; the paper is fp32. Read the YAML.
        use_amp=bool(train_sub.get("amp", False)),
        scheduler=str(train_sub.get("scheduler", "plateau")),
        plateau_patience=int(train_sub.get("scheduler_patience", 5)),
        plateau_factor=float(train_sub.get("scheduler_factor", 0.5)),
        model_cfg=ModelConfig(),
        loss_cfg=loss_cfg,
        resume=resume if resume is not None else str(train_sub.get("resume", "auto")),
    )

    if train_cfg.use_amp:
        logger.warning(
            "use_amp is True. The paper checkpoint is fp32; these runs will "
            "not be comparable to it. Set eccount.train.amp: false."
        )

    target_cfg = SoftTargetConfig(sigma=sigma)
    ds_train_cfg = DatasetConfig(
        training=True,
        hflip_prob=float(aug_sub.get("hflip_p", 0.5)),
        vflip_prob=float(aug_sub.get("vflip_p", 0.5)),
        brightness_jitter_prob=float(aug_sub.get("brightness_p", 0.2)),
        brightness_jitter_range=tuple(aug_sub.get("brightness_range", (0.9, 1.1))),
    )
    ds_val_cfg = DatasetConfig(training=False)

    logger.info(
        "Training: %d train / %d val images | epochs=%d batch=%d lr=%.0e "
        "seed=%d amp=%s",
        len(train_df), len(val_df), train_cfg.max_epochs, train_cfg.batch_size,
        train_cfg.lr, train_cfg.seed, train_cfg.use_amp,
    )

    train_ds = EcCountDataset(train_df, ds_train_cfg, target_cfg)
    val_ds = EcCountDataset(val_df, ds_val_cfg, target_cfg)

    result = train_eccount(train_cfg, train_ds, val_ds)

    # The trainer names its checkpoint from its own convention; normalise it so
    # run_config.yaml can point at a fixed filename.
    best = Path(str(result.best_ckpt))
    target = run_dir / "best_model.pt"
    if best.exists() and best.resolve() != target.resolve():
        shutil.copy2(best, target)

    return {
        "best_epoch": int(result.best_epoch),
        "best_val_loss": float(result.best_val_loss),
        "best_ckpt": str(target),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Leave-one-cell-line-out training for ecCount.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--hold-out", required=True,
        help="Cell line to hold out, or 'all' with --dry-run to preview every run.",
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument(
        "--out-root", type=Path, default=None,
        help="Run root. Default: paths.eccount_loco_dir, else outputs/eccount_loco.",
    )
    p.add_argument(
        "--size-matched-control", action="store_true",
        help="Train the data-volume control instead of the leave-one-out run.",
    )
    p.add_argument("--control-seed", type=int, default=42)
    p.add_argument(
        "--dry-run", action="store_true",
        help="Print the split composition and exit. No GPU, no writes.",
    )
    p.add_argument("--resume", default=None, help="'auto', 'never', or a path.")
    p.add_argument(
        "--no-strict", action="store_true",
        help="Report invariant failures without exiting. For diagnosis only.",
    )
    p.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    from ecdna_bench.cli._common import get_path, load_config, require_file

    base_cfg = load_config(args.config)
    consistency_csv = get_path(base_cfg, "consistency_csv")
    require_file(consistency_csv, "consistency CSV")

    out_root = (
        args.out_root
        or get_path(base_cfg, "eccount_loco_dir", required=False)
        or Path("outputs/eccount_loco")
    )

    df = load_consistent_metadata(consistency_csv)
    logger.info("Loaded %d consistent images from %s", len(df), consistency_csv)
    check_invariants(df, strict=not args.no_strict)
    logger.info("All metadata invariants passed.")

    targets = (
        list(EXPECTED_CELL_LINES) if args.hold_out.lower() == "all"
        else [args.hold_out]
    )
    if len(targets) > 1 and not args.dry_run:
        logger.error("--hold-out all is only supported with --dry-run.")
        return 2

    for hold_out in targets:
        train_df, val_df, eval_df, info = build_loco_splits(
            df, hold_out, args.size_matched_control, args.control_seed
        )
        suffix = "_control" if args.size_matched_control else ""
        run_id = f"holdout_{slugify(hold_out)}{suffix}"
        comp = composition_table(train_df, val_df, eval_df)

        print()
        print("=" * 78)
        print(f"  {run_id}")
        print("=" * 78)
        print(f"  hold out            : {hold_out}")
        print(f"  training on         : {', '.join(info['train_cell_lines'])}")
        print(f"  train / val images  : {info['n_train']} / {info['n_val']}")
        print(f"  evaluate on         : {info['n_eval']} images "
              f"({info['eval_scope']})")
        print(f"  evaluation objects  : {info['eval_gt_objects']:,}")
        print()
        print(comp.to_string(index=False))
        print()

        if args.dry_run:
            continue

        run_dir = Path(out_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        comp.to_csv(run_dir / "split_composition.csv", index=False)
        eval_csv = run_dir / "eval_metadata.csv"
        eval_df.to_csv(eval_csv, index=False)
        run_cfg_path = write_run_config(base_cfg, run_dir, eval_csv)

        # Second evaluation set: the test-split images of the cell lines this
        # model DID train on. Scoring the same checkpoint on both sets gives a
        # within-model comparison, so the generalisation gap is not confounded
        # by also changing which model is being measured. These images are
        # unseen either way — they are test rows — so this is a fair reference,
        # not a training-set score.
        seen_df = df[
            (df["cell_line"] != hold_out) & (df["split"] == "test")
        ].reset_index(drop=True)
        leak = set(seen_df["unique_id"]) & (
            set(train_df["unique_id"]) | set(val_df["unique_id"])
        )
        assert not leak, f"{len(leak)} reference images were trained on"
        seen_csv = run_dir / "eval_metadata_seen_lines.csv"
        seen_df.to_csv(seen_csv, index=False)
        seen_cfg_path = write_run_config(
            base_cfg, run_dir, seen_csv, tag="seen_lines")
        logger.info(
            "Reference set: %d test images from %s",
            len(seen_df), ", ".join(sorted(seen_df["cell_line"].unique())),
        )
        logger.info("Run directory: %s", run_dir)

        started = datetime.now(timezone.utc).isoformat()
        train_result = train_one(base_cfg, train_df, val_df, run_dir, args.resume)
        finished = datetime.now(timezone.utc).isoformat()

        manifest = {
            "run_id": run_id,
            **info,
            **train_result,
            "n_reference_images": len(seen_df),
            "reference_gt_objects": int(seen_df["ecDNA_gt"].sum()),
            "started_utc": started,
            "finished_utc": finished,
            "consistency_csv": str(consistency_csv),
            "base_config": str(args.config),
            "run_config": str(run_cfg_path),
            "run_config_seen_lines": str(seen_cfg_path),
            "invariants": "passed",
        }
        with open(run_dir / "run_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)

        logger.info(
            "Done: best epoch %d, best val loss %.4f",
            train_result["best_epoch"], train_result["best_val_loss"],
        )
        print()
        print("Next, from the repository root — the held-out cell line:")
        print(f"  python -m ecdna_bench.cli.run_eccount "
              f"--config {run_cfg_path} --split all")
        print(f"  python -m ecdna_bench.cli.benchmark "
              f"--config {run_cfg_path} --models eccount_peaks eccount_mask "
              f"--skip-harmonize --output-dir {run_dir / 'results'}")
        print("and the reference set of cell lines it trained on:")
        print(f"  python -m ecdna_bench.cli.run_eccount "
              f"--config {seen_cfg_path} --split all")
        print(f"  python -m ecdna_bench.cli.benchmark "
              f"--config {seen_cfg_path} --models eccount_peaks eccount_mask "
              f"--skip-harmonize --output-dir {run_dir / 'results_seen_lines'}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())