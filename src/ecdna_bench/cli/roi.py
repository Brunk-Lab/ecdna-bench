"""
ecdna_bench.cli.roi
===================
Predict or train the region-of-interest (ROI) model.

Predict (released weights; GPU if available, CPU works):

    python -m ecdna_bench.cli.roi predict --rgb <rgb folder> --dapi <dapi folder> \
        --checkpoint roi_model_best_checkpoint.pth --output <folder>

Train (needs ``albumentations==2.0.8``; the released model used these values):

    python -m ecdna_bench.cli.roi train --rgb <rgb> --dapi <dapi> --roi-mask <roi_mask> \
        --splits <folder with train_ids.csv and val_ids.csv> --output <folder> --name <run>

Images are matched by unique identifier: ``<folder>/<unique_id>.tif|.tiff|.png``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from ecdna_bench.roi import infer, io as roi_io


def _size(values):
    return (int(values[0]), int(values[1]))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Region-of-interest model: predict or train.")
    sub = p.add_subparsers(dest="command", required=True)
    f = argparse.ArgumentDefaultsHelpFormatter

    pr = sub.add_parser("predict", help="write one 0/255 PNG mask per image set", formatter_class=f)
    pr.add_argument("--rgb", type=Path, default=None, help="folder with RGB images")
    pr.add_argument("--dapi", type=Path, default=None, help="folder with DAPI images")
    pr.add_argument("--checkpoint", type=Path, required=True, help="roi_model_best_checkpoint.pth")
    pr.add_argument("--output", type=Path, required=True)
    pr.add_argument("--ids", type=Path, default=None,
                    help="CSV whose first column lists unique identifiers (header row skipped); "
                         "default: every image in --rgb")
    pr.add_argument("--device", default=None, help="cpu, cuda or cuda:N (default: cuda if available)")
    pr.add_argument("--threshold", type=float, default=infer.THRESHOLD)
    pr.add_argument("--train-size", nargs=2, type=int, default=list(infer.TRAIN_SIZE))
    pr.add_argument("--save-size", nargs=2, type=int, default=list(infer.SAVE_SIZE))
    pr.add_argument("--overwrite", action="store_true")

    tr = sub.add_parser("train", help="train the ROI model", formatter_class=f)
    tr.add_argument("--rgb", type=Path, default=None)
    tr.add_argument("--dapi", type=Path, default=None)
    tr.add_argument("--roi-mask", type=Path, required=True)
    tr.add_argument("--splits", type=Path, required=True,
                    help="folder with train_ids.csv and val_ids.csv")
    tr.add_argument("--output", type=Path, required=True)
    tr.add_argument("--name", default="ROI_training")
    tr.add_argument("--resize", nargs=2, type=int, default=[1024, 1224])
    tr.add_argument("--batch-size", type=int, default=2)
    tr.add_argument("--accum-steps", type=int, default=4)
    tr.add_argument("--num-epochs", type=int, default=200)
    tr.add_argument("--lr", type=float, default=1e-4)
    tr.add_argument("--decay", type=float, default=1e-4)
    tr.add_argument("--pos-weight", type=float, default=1.0)
    tr.add_argument("--device", default=None)
    tr.add_argument("--num-cpus", type=int, default=5)
    tr.add_argument("--max-images", type=int, default=None, help="smoke tests only")
    return p.parse_args(argv)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    if args.rgb is None and args.dapi is None:
        sys.exit("give --rgb and/or --dapi")
    if args.command == "predict":
        ids = roi_io.read_ids(args.ids) if args.ids else None
        n = infer.predict_folder(args.rgb, args.dapi, args.checkpoint, args.output, ids=ids,
                                 device=args.device, threshold=args.threshold,
                                 train_size=_size(args.train_size), save_size=_size(args.save_size),
                                 overwrite=args.overwrite)
        print(f"masks written: {n} -> {args.output}")
        return 0
    from ecdna_bench.roi.train import train_roi
    run = train_roi(args.rgb, args.dapi, args.roi_mask, args.splits, args.output, name=args.name,
                    img_size=_size(args.resize), batch_size=args.batch_size,
                    accum_steps=args.accum_steps, num_epochs=args.num_epochs, lr=args.lr,
                    weight_decay=args.decay, pos_weight=args.pos_weight, device=args.device,
                    num_cpus=args.num_cpus, max_images=args.max_images)
    print(f"run folder: {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
