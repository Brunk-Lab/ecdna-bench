"""
ecdna_bench.cli.run_eccount
==============================
Run ecCount inference → threshold mask PNGs + peaks mask PNGs.

Usage
-----
    python -m ecdna_bench.cli.run_eccount --config configs/default.yaml

Required path keys (set in configs/paths.local.yaml)
------------------------------------------------------
* ``consistency_csv``
* ``eccount_checkpoint``        — best_model.pt written by train_eccount

Optional path keys (have working defaults)
-------------------------------------------
* ``eccount_threshold_masks``   — default: release/harmonized_masks/eccount_threshold
* ``eccount_peaks_masks``       — default: release/harmonized_masks/eccount_peaks

Performance design
------------------
The model is loaded ONCE at startup, then each image is processed
sequentially on the GPU.  Using a ProcessPoolExecutor was removed because
each worker would rebuild the model and reload the checkpoint from disk for
every single image — O(N × model_load_time) instead of O(N × infer_time).
On an A100, 1,145 images take ~20 min with the sequential design.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from ecdna_bench.cli._common import (
    get_path, load_config, require_file, resolve_path, setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ecCount inference on all benchmark images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--split",  choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--device", default=None,
                   help="Device override: 'cuda', 'cpu', or 'cuda:N'.")
    p.add_argument("--force",  action="store_true",
                   help="Re-run images whose output files already exist.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _write_mask(mask, out_path: Path) -> None:
    import cv2
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = str(out_path) + ".tmp.png"
    cv2.imwrite(tmp, mask)
    os.replace(tmp, str(out_path))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv = get_path(cfg, "consistency_csv")
    ckpt_path       = get_path(cfg, "eccount_checkpoint")
    threshold_dir   = get_path(cfg, "eccount_threshold_masks", required=False) or \
                      resolve_path("release/harmonized_masks/eccount_threshold")
    peaks_dir       = get_path(cfg, "eccount_peaks_masks", required=False) or \
                      resolve_path("release/harmonized_masks/eccount_peaks")

    require_file(consistency_csv, "consistency CSV")
    require_file(ckpt_path,       "ecCount checkpoint")
    threshold_dir.mkdir(parents=True, exist_ok=True)
    peaks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Read config — eccount.input_size is the correct YAML key.
    # (NOT eccount.image_h / eccount.image_w — those keys don't exist.)
    # ------------------------------------------------------------------
    eccount_cfg = cfg.get("eccount", {})
    input_size  = eccount_cfg.get("input_size", [1024, 1224])
    image_h, image_w = int(input_size[0]), int(input_size[1])

    # Post-processing config — all keys from eccount.postprocess.*
    pp_sub = eccount_cfg.get("postprocess", {})

    # Device: CLI flag > eccount.train.device > "cuda"
    device_str = (
        args.device
        or eccount_cfg.get("train", {}).get("device", "cuda")
    )

    # ------------------------------------------------------------------
    # Build and load model ONCE.
    # Do NOT move model construction inside any per-image loop.
    # ------------------------------------------------------------------
    import torch
    import cv2
    import numpy as np
    import pandas as pd

    from ecdna_bench.eccount.model       import ModelConfig, build_model
    from ecdna_bench.eccount.infer       import InferConfig, load_checkpoint, infer_one
    from ecdna_bench.eccount.postprocess import PostprocessConfig

    logger.info("Device: %s", device_str)
    logger.info("Checkpoint: %s", ckpt_path)

    device = torch.device(device_str)
    model  = build_model(ModelConfig())
    load_checkpoint(str(ckpt_path), model, device=str(device))
    model.eval().to(device)

    # Build InferConfig from YAML values
    pp_cfg = PostprocessConfig(
        smooth_sigma      = float(pp_sub.get("smooth_sigma",      0.5)),
        reapply_roi       = bool( pp_sub.get("reapply_roi",       True)),
        threshold_abs     = float(pp_sub.get("threshold_abs",     0.35)),
        peak_min_distance = int(  pp_sub.get("peak_min_distance", 2)),
        nms_min_distance  = int(  pp_sub.get("nms_min_distance",  2)),
        exclude_border    = int(  pp_sub.get("exclude_border",    0)),
        point_disk_radius = int(  pp_sub.get("point_disk_radius", 3)),
    )
    infer_cfg = InferConfig(
        postprocess           = pp_cfg,
        threshold_mask_cutoff = 0.5,
        peaks_disk_radius     = pp_cfg.point_disk_radius,
    )

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]
    if args.split != "all":
        df = df[df["split"] == args.split]

    logger.info("Running ecCount inference on %d images", len(df))
    logger.info("  threshold_dir: %s", threshold_dir)
    logger.info("  peaks_dir:     %s", peaks_dir)

    # ------------------------------------------------------------------
    # Sequential inference loop — GPU is shared within one process
    # ------------------------------------------------------------------
    ok = skip = err = 0

    for _, row in df.iterrows():
        uid       = str(row["unique_id"])
        rgb_path  = str(row["rgb_fullpath"])
        roi_path  = str(row.get("roi_fullpath", "")) if "roi_fullpath" in row else ""
        thr_out   = threshold_dir / f"{uid}.png"
        peaks_out = peaks_dir     / f"{uid}.png"

        if not args.force and thr_out.exists() and peaks_out.exists():
            skip += 1
            done = ok + skip + err
            if done % 100 == 0 or done == len(df):
                logger.info("Progress: %d/%d  ok=%d skip=%d err=%d",
                            done, len(df), ok, skip, err)
            continue

        try:
            # Load RGB — record original resolution before downsampling
            rgb = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if rgb is None:
                raise FileNotFoundError(f"Cannot read RGB: {rgb_path}")
            orig_h, orig_w = rgb.shape[:2]
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (image_w, image_h), interpolation=cv2.INTER_AREA)

            # Load ROI mask
            roi_np = None
            if roi_path and Path(roi_path).is_file():
                roi_raw = cv2.imread(roi_path, cv2.IMREAD_GRAYSCALE)
                if roi_raw is not None:
                    roi_np = (cv2.resize(roi_raw, (image_w, image_h),
                                         interpolation=cv2.INTER_NEAREST) > 0
                              ).astype(np.uint8)

            # Apply ROI mask before the forward pass
            if roi_np is not None:
                rgb = rgb * roi_np[:, :, None]

            # Build input tensor: (H, W, 3) → (1, 3, H, W) on device
            img_t = torch.from_numpy(
                rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
            ).unsqueeze(0).to(device)

            result = infer_one(img_t, roi_np, model, infer_cfg)

            # Upsample threshold mask to original resolution (INTER_NEAREST
            # preserves binary values exactly).
            # For peaks mask, scale COORDINATES then re-render disks at
            # original resolution — naive upsampling doubles disk radius.
            if (orig_h, orig_w) != (image_h, image_w):
                thr_mask = cv2.resize(result.threshold_mask,
                                      (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
                scale_x = orig_w / image_w
                scale_y = orig_h / image_h
                scaled_peaks = [
                    (int(round(x * scale_x)), int(round(y * scale_y)), s)
                    for x, y, s in result.peaks
                ]
                from ecdna_bench.eccount.postprocess import peaks_to_mask
                peaks_mask = peaks_to_mask(
                    scaled_peaks,
                    shape=(orig_h, orig_w),
                    disk_radius=infer_cfg.peaks_disk_radius,
                )
            else:
                thr_mask   = result.threshold_mask
                peaks_mask = result.peaks_mask

            _write_mask(thr_mask,   thr_out)
            _write_mask(peaks_mask, peaks_out)
            ok += 1

        except Exception as exc:
            err += 1
            logger.debug("%s: error — %s", uid, exc)

        done = ok + skip + err
        if done % 100 == 0 or done == len(df):
            logger.info("Progress: %d/%d  ok=%d skip=%d err=%d",
                        done, len(df), ok, skip, err)

    logger.info("run_eccount complete — ok=%d  skip=%d  err=%d", ok, skip, err)
    if err > 0:
        logger.warning(
            "%d images failed. Re-run with --log-level DEBUG to see per-image errors.",
            err,
        )


if __name__ == "__main__":
    main()