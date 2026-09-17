#!/usr/bin/env python3
"""
scripts/prepare_local_run.py
============================
Prepare a self-contained run of the ecdna-bench command-line tools on any
machine, without the UNC Longleaf paths.

The released benchmark table (``release/manifests/
dl_master_metadata_stage1_step3_consistency.csv``) records absolute paths on
the machine that produced the paper.  The command-line tools read those
paths directly, so on another computer they must be rewritten.  This script
does that and writes everything a run needs into one folder:

    <out-dir>/
        manifest_local.csv   the benchmark table with local file paths
        predictions/<model>/ prediction masks as binary PNG (0/255), one folder
                             per method, named by the benchmark model key
        run_config.yaml      a complete configuration (configs/default.yaml
                             plus local paths); pass it with --config
        NEXT_STEPS.txt       the exact commands to run next

``run_config.yaml`` is complete on purpose: the tools merge a
``paths.local.yaml`` found next to the config file over it, so this script
refuses to write into a folder that contains one.

Sub-commands
------------
bia     Point the benchmark at a local copy of BioImage Archive S-BIAD4097
        (downloaded with scripts/fetch_bia_subset.py).

            python scripts/prepare_local_run.py bia --bia-root ~/ecdna_data \\
                --out-dir runs/bia_test --split test

images  Run ecCount on your own images (no gold standard needed).

            python scripts/prepare_local_run.py images --images my_images/ \\
                --out-dir runs/my_images --checkpoint models/eccount_best.pt

count   Count objects (8-connected components, area >= 3 px) in a folder of
        binary masks, for example the ecCount peaks output.

            python scripts/prepare_local_run.py count \\
                --masks runs/my_images/eccount/eccount_peaks --out counts.csv

Run from the repository root.  Requires the ecdna-bench environment
(numpy, pandas, opencv, PyYAML).
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
IMAGE_EXT = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
NATIVE_SHAPE = (2048, 2448)  # rows, columns

# Archive prediction folder -> benchmark display name (the deposited record).
ARCHIVE_METHODS: Dict[str, str] = {
    "eccount_peaks": "ecCount (peaks)",
    "eccount_threshold": "ecCount (threshold mask)",
    "label_engine": "Label Engine",
    "mia": "MIA",
    "classical_optimised": "Classic (after opt)",
    "ecseg": "ecSeg",
    "classical_default": "Classic (before opt)",
}

# Used only if the installed package cannot be imported.
FALLBACK_REGISTRY: Dict[str, Tuple[str, str]] = {
    # display name: (model key, config path key)
    "Classic (after opt)": ("classical", "classical_masks"),
    "Classic (before opt)": ("classical_before_opt", "classical_default_masks"),
    "Label Engine": ("label_engine", "label_engine_masks"),
    "ecSeg": ("ecseg", "ecseg_masks"),
    "MIA": ("mia", "mia_masks"),
    "ecCount (threshold mask)": ("eccount_mask", "eccount_threshold_masks"),
    "ecCount (peaks)": ("eccount_peaks", "eccount_peaks_masks"),
}
ECCOUNT_NAMES = ("ecCount (peaks)", "ecCount (threshold mask)")
SUFFIXES = ("_pred_roi", "_predroi", "_predicted_roi", "_predicted", "_pred", "_roi", "_mask")


def _die(msg: str, code: int = 2) -> None:
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def load_registry() -> Tuple[Dict[str, Tuple[str, str]], str]:
    try:
        from ecdna_bench.benchmark.registry import MODEL_REGISTRY  # type: ignore
    except Exception as exc:  # package not importable
        return dict(FALLBACK_REGISTRY), f"built-in table (package import failed: {exc})"
    reg = {spec.name: (key, spec.mask_dir_key) for key, spec in MODEL_REGISTRY.items()}
    return reg, "ecdna_bench.benchmark.registry"


# ----------------------------------------------------------------------------
# File indexing and image I/O
# ----------------------------------------------------------------------------

def index_folder(folder: Path, recursive: bool = False) -> Dict[str, Path]:
    """Map file stem -> path for image-like files in `folder`."""
    out: Dict[str, Path] = {}
    if not folder.is_dir():
        return out
    it = folder.rglob("*") if recursive else folder.iterdir()
    for p in sorted(it):
        if p.is_file() and p.suffix.lower() in IMAGE_EXT and not p.name.endswith(".part"):
            out.setdefault(p.stem, p)
    return out


def find(index: Dict[str, Path], uid: str) -> Optional[Path]:
    if uid in index:
        return index[uid]
    for suf in SUFFIXES:
        if uid + suf in index:
            return index[uid + suf]
    lower = uid.lower()
    for stem, p in index.items():
        if stem.lower() == lower:
            return p
    hits = [p for stem, p in index.items() if stem.startswith(uid + "_")]
    return hits[0] if len(hits) == 1 else None


def read_gray(path: Path):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        try:
            import tifffile  # type: ignore
            img = tifffile.imread(str(path))
        except Exception:
            try:
                from PIL import Image
                img = np.array(Image.open(path))
            except Exception as exc:
                raise IOError(f"cannot read {path}: {exc}")
    img = np.asarray(img)
    if img.ndim == 3:
        img = img.max(axis=2)
    return img


def write_binary_png(mask, dst: Path) -> None:
    import cv2
    import numpy as np

    dst.parent.mkdir(parents=True, exist_ok=True)
    out = (np.asarray(mask) > 0).astype(np.uint8) * 255
    tmp = dst.with_name(dst.stem + ".tmp.png")
    if not cv2.imwrite(str(tmp), out):
        raise IOError(f"could not write {tmp}")
    with open(tmp, "rb") as fh:
        if fh.read(8) != PNG_MAGIC:
            tmp.unlink()
            raise IOError(f"{tmp} is not a PNG file after writing")
    os.replace(tmp, dst)


# Archive folders that hold raw tool output rather than binary masks, with the
# benchmark's harmonization rule (threshold, minimum component area). Label
# Engine is stored as RGB; ecdna_bench.baselines.label_engine converts it to
# grayscale, keeps values above 0.5 and drops 8-connected components smaller
# than 3 px. The same rule here gives the masks behind the published results.
RAW_TOOL_OUTPUT: Dict[str, Tuple[float, int]] = {"label_engine": (0.5, 3)}


def harmonize_raw_mask(path: Path, threshold: float, min_area: int):
    import cv2
    import numpy as np

    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"cannot read {path}")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
    fg = (img.astype(np.float32) > threshold).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[lab]


def check_config_dir(out_dir: Path) -> None:
    stray = out_dir / "paths.local.yaml"
    if stray.exists():
        _die(f"{stray} exists. The tools would merge it over run_config.yaml. Remove it "
             "or choose another --out-dir.")


def write_run_config(base_config: Path, out_dir: Path, paths: Dict[str, str],
                     device: Optional[str], n_workers: int = 1) -> Path:
    import yaml

    if not base_config.is_file():
        _die(f"base config not found: {base_config} (run from the repository root)")
    with open(base_config, encoding="utf-8") as fh:
        base = yaml.safe_load(fh) or {}
    cfg = copy.deepcopy(base)
    cfg.setdefault("paths", {}).update(paths)
    if device:
        cfg.setdefault("eccount", {}).setdefault("train", {})["device"] = device
    # scoring memory: about 10 GB per worker; the CLI default (8) needs ~80 GB
    cfg.setdefault("benchmark", {})["n_workers"] = n_workers
    check_config_dir(out_dir)
    dest = out_dir / "run_config.yaml"
    with open(dest, "w", encoding="utf-8") as fh:
        fh.write("# Generated by scripts/prepare_local_run.py. Complete configuration:\n"
                 f"# {base_config} plus the local paths below. Do not place a\n"
                 "# paths.local.yaml in this folder.\n")
        yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
    return dest


# ----------------------------------------------------------------------------
# bia
# ----------------------------------------------------------------------------

def cmd_bia(args) -> int:
    import pandas as pd

    root = args.bia_root.expanduser().resolve()
    if (root / "Files").is_dir() and not (root / "images").is_dir():
        root = root / "Files"
    images = root / "images"
    if not images.is_dir():
        _die(f"{images} not found. --bia-root must contain images/ (the archive's Files folder).")
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_config_dir(out)

    manifest = Path(args.manifest)
    if not manifest.is_file():
        _die(f"benchmark table not found: {manifest}")
    df = pd.read_csv(manifest, low_memory=False)
    for col in ("unique_id", "split", "count_mask_consistent"):
        if col not in df.columns:
            _die(f"{manifest} has no '{col}' column")
    df["unique_id"] = df["unique_id"].astype(str)
    n0 = len(df)
    df = df[df["count_mask_consistent"].fillna(False).astype(bool)]
    if args.split != "all":
        df = df[df["split"].astype(str) == args.split]
    if args.cell_line:
        df = df[df["cell_line"].isin(args.cell_line)]
    print(f"benchmark table : {manifest} ({n0} rows; {len(df)} selected)")

    idx = {
        "rgb": index_folder(images / "rgb"),
        "dapi": index_folder(images / "dapi"),
        "gt": index_folder(images / "gt_image"),
        "roi_manual": index_folder(images / "roi_mask"),
        "roi_predicted": index_folder(root / "predicted_roi", recursive=True),
    }
    for k, v in idx.items():
        print(f"  found {len(v):>5} files for {k}")

    roi_key = {"manual": "roi_manual", "predicted": "roi_predicted"}.get(args.roi)
    rows, dropped = [], []
    for rec in df.to_dict(orient="records"):
        uid = rec["unique_id"]
        rgb, dapi, gt = find(idx["rgb"], uid), find(idx["dapi"], uid), find(idx["gt"], uid)
        roi = find(idx[roi_key], uid) if roi_key else None
        missing = []
        if gt is None:
            missing.append("gold-standard mask")
        if args.own_eccount and rgb is None:
            missing.append("RGB image")
        if roi_key and roi is None:
            missing.append(f"{args.roi} ROI mask")
        if missing:
            dropped.append((uid, ", ".join(missing)))
            continue
        rec["rgb_fullpath"] = str(rgb) if rgb else ""
        rec["dapi_fullpath"] = str(dapi) if dapi else ""
        rec["gt_fullpath"] = str(gt)
        rec["roi_fullpath"] = str(roi) if roi else ""
        rows.append(rec)
    if dropped:
        print(f"  {len(dropped)} row(s) skipped because files are not in the download, e.g.:")
        for uid, why in dropped[:5]:
            print(f"    {uid}: missing {why}")
        if args.strict:
            _die("--strict: every selected row must have its files", 1)
    if not rows:
        _die("no usable rows; check --bia-root, --split and what was downloaded")
    local = pd.DataFrame(rows)

    # Gold-standard masks must be 0/255 for the pixel-level metrics.
    shapes: Dict[str, Tuple[int, int]] = {}
    rescaled = 0
    gs_dir = out / "gold_standard_rescaled"
    for i, rec in local.iterrows():
        g = read_gray(Path(rec["gt_fullpath"]))
        shapes[rec["unique_id"]] = g.shape[:2]
        if g.max() not in (0, 255):
            write_binary_png(g, gs_dir / f"{rec['unique_id']}.png")
            local.at[i, "gt_fullpath"] = str(gs_dir / f"{rec['unique_id']}.png")
            rescaled += 1
    if rescaled:
        print(f"  {rescaled} gold-standard mask(s) rescaled to 0/255 in {gs_dir}")
    odd = sorted({s for s in shapes.values() if s != NATIVE_SHAPE})
    if odd:
        print(f"  note: gold-standard shapes other than {NATIVE_SHAPE}: {odd[:3]}")

    manifest_out = out / "manifest_local.csv"
    local.to_csv(manifest_out, index=False)

    registry, reg_source = load_registry()
    print(f"model registry  : {reg_source}")
    pred_root = out / "predictions"
    paths: Dict[str, str] = {
        "data_root": str(images),
        "metadata_csv": str(manifest_out),
        "consistency_csv": str(manifest_out),
        "frozen_results_dir": str(out / "results"),
        "harmonized_masks_dir": str(pred_root),
        "results_root": str(out / "results_root"),
        "logs_root": str(out / "logs"),
        "eccount_out_dir": str(out / "eccount_training"),
    }
    prepared: List[str] = []
    for folder, display in ARCHIVE_METHODS.items():
        if display not in registry:
            continue
        key, path_key = registry[display]
        dest = pred_root / key
        if args.own_eccount and display in ECCOUNT_NAMES:
            paths[path_key] = str(out / "own_eccount" / key)
        else:
            paths[path_key] = str(dest)
        src_idx = index_folder(root / "predictions" / folder)
        if not src_idx:
            continue
        ok = missing = bad_shape = 0
        for uid in local["unique_id"]:
            src = find(src_idx, uid)
            if src is None:
                missing += 1
                continue
            if folder in RAW_TOOL_OUTPUT:
                mask = harmonize_raw_mask(src, *RAW_TOOL_OUTPUT[folder])
            else:
                mask = read_gray(src)
            if mask.shape[:2] != shapes[uid]:
                bad_shape += 1
                continue
            write_binary_png(mask, dest / f"{uid}.png")
            ok += 1
        prepared.append(f"{display:<26} {key:<22} {ok:>5} ok  {missing:>4} missing  "
                        f"{bad_shape:>3} shape mismatch")
    print("prediction masks:")
    for line in prepared or ["  none found under predictions/ (fine if you only run ecCount)"]:
        print(f"  {line}")

    if args.checkpoint:
        ck = args.checkpoint.expanduser().resolve()
        if not ck.is_file():
            _die(f"checkpoint not found: {ck}")
        paths["eccount_checkpoint"] = str(ck)
    cfg_path = write_run_config(Path(args.base_config), out, paths, args.device,
                                args.n_workers)

    bench_keys = [registry[d][0] for f, d in ARCHIVE_METHODS.items()
                  if d in registry and index_folder(root / "predictions" / f)
                  and not (args.own_eccount and d in ECCOUNT_NAMES)]
    if not bench_keys:
        bench_keys = [registry[d][0] for d in ARCHIVE_METHODS.values()
                      if d in registry and not (args.own_eccount and d in ECCOUNT_NAMES)]
    split_word = "all" if args.split == "all" else args.split
    lines = [
        "Commands (run from the repository root):",
        "",
        "# score the prepared predictions (writes <out-dir>/results/).",
        "# --skip-harmonize: the masks written here are already binary 0/255 PNGs;",
        "# harmonizing them again would treat them as raw tool output.",
        f"# memory: about 10 GB per scoring worker; --n-workers {args.n_workers} is also the",
        "# default in run_config.yaml. Raise it only if the computer has the memory.",
        f"python -m ecdna_bench.cli.benchmark --config {cfg_path} --models {' '.join(k for k in bench_keys)} "
        f"--skip-harmonize --output-dir {out / 'results'} --n-workers {args.n_workers}",
        f"python scripts/verify_headline_numbers.py --results {out / 'results'}",
    ]
    if args.own_eccount:
        eccount_keys = [registry[n][0] for n in ECCOUNT_NAMES if n in registry]
        lines += [
            "",
            "# run ecCount yourself, then score only its two outputs",
            f"python -m ecdna_bench.cli.run_eccount --config {cfg_path} --split {split_word}"
            + (f" --device {args.device}" if args.device else ""),
            f"python -m ecdna_bench.cli.benchmark --config {cfg_path} --models {' '.join(eccount_keys)} "
            f"--skip-harmonize --output-dir {out / 'results_own_eccount'} --n-workers {args.n_workers}",
            f"python scripts/verify_headline_numbers.py --results {out / 'results_own_eccount'}",
        ]
    (out / "NEXT_STEPS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"rows written    : {len(local)} -> {manifest_out}")
    print(f"run config      : {cfg_path}")
    print("\n" + "\n".join(lines))
    return 0


# ----------------------------------------------------------------------------
# images
# ----------------------------------------------------------------------------

def cmd_images(args) -> int:
    import pandas as pd

    src = args.images.expanduser().resolve()
    if not src.is_dir():
        _die(f"not a folder: {src}")
    out = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    check_config_dir(out)
    pattern = re.compile(args.exclude) if args.exclude else None
    imgs = {k: v for k, v in index_folder(src, recursive=args.recursive).items()
            if not (pattern and pattern.search(v.name))}
    if not imgs:
        _die(f"no images (.tif/.tiff/.png/.jpg) in {src}")
    rois = index_folder(args.roi_dir.expanduser().resolve()) if args.roi_dir else {}

    rows, sizes = [], {}
    for uid, p in sorted(imgs.items()):
        roi = find(rois, uid) if rois else None
        if rois and roi is None and args.require_roi:
            print(f"  skipped {uid}: no ROI mask")
            continue
        try:
            shape = read_gray(p).shape[:2]
        except IOError as exc:
            print(f"  skipped {uid}: {exc}")
            continue
        sizes[shape] = sizes.get(shape, 0) + 1
        rows.append({"unique_id": uid, "cell_line": args.cell_line or "unknown", "split": "new",
                     "rgb_fullpath": str(p), "roi_fullpath": str(roi) if roi else "",
                     "dapi_fullpath": "", "gt_fullpath": "", "count_mask_consistent": True,
                     "ecDNA_gt": ""})
    if not rows:
        _die("no usable images")
    manifest_out = out / "manifest_local.csv"
    pd.DataFrame(rows).to_csv(manifest_out, index=False)
    print(f"images          : {len(rows)} -> {manifest_out}")
    print(f"with ROI mask   : {sum(1 for r in rows if r['roi_fullpath'])}")
    for shape, n in sorted(sizes.items(), key=lambda kv: -kv[1]):
        flag = "" if shape == NATIVE_SHAPE else "  <- differs from the training images (2,048 x 2,448 px)"
        print(f"  {n:>5} image(s) of {shape[0]} x {shape[1]} px{flag}")

    registry, _ = load_registry()
    paths = {"consistency_csv": str(manifest_out), "metadata_csv": str(manifest_out),
             "logs_root": str(out / "logs"), "results_root": str(out / "results_root")}
    for name in ECCOUNT_NAMES:
        if name in registry:
            key, path_key = registry[name]
            paths[path_key] = str(out / "eccount" / ("eccount_peaks" if "peaks" in name
                                                     else "eccount_threshold"))
    if args.checkpoint:
        ck = args.checkpoint.expanduser().resolve()
        if not ck.is_file():
            _die(f"checkpoint not found: {ck}")
        paths["eccount_checkpoint"] = str(ck)
    cfg_path = write_run_config(Path(args.base_config), out, paths, args.device)
    lines = [
        "Commands (run from the repository root):",
        "",
        f"python -m ecdna_bench.cli.run_eccount --config {cfg_path} --split all"
        + (f" --device {args.device}" if args.device else ""),
        f"python scripts/prepare_local_run.py count --masks {out / 'eccount' / 'eccount_peaks'} "
        f"--out {out / 'ecdna_counts.csv'}",
    ]
    (out / "NEXT_STEPS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"run config      : {cfg_path}")
    print("\n" + "\n".join(lines))
    return 0


# ----------------------------------------------------------------------------
# count
# ----------------------------------------------------------------------------

def cmd_count(args) -> int:
    import csv

    import cv2

    folder = args.masks.expanduser().resolve()
    files = sorted(p for p in folder.glob("*.png") if not p.name.endswith(".tmp.png"))
    if not files:
        _die(f"no PNG masks in {folder}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["unique_id", "ecdna_count", "min_area_px", "connectivity"])
        for p in files:
            m = (read_gray(p) > 0).astype("uint8")
            n, _, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            count = int((stats[1:, cv2.CC_STAT_AREA] >= args.min_area).sum()) if n > 1 else 0
            w.writerow([p.stem, count, args.min_area, 8])
    print(f"counted {len(files)} mask(s) -> {args.out}")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("bia", help="use a local copy of BioImage Archive S-BIAD4097")
    b.add_argument("--bia-root", type=Path, required=True,
                   help="folder that contains images/ (and predictions/, predicted_roi/)")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--manifest", default="release/manifests/dl_master_metadata_stage1_step3_consistency.csv")
    b.add_argument("--base-config", default="configs/default.yaml")
    b.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    b.add_argument("--cell-line", action="append", default=[])
    b.add_argument("--roi", default="manual", choices=["manual", "predicted", "none"],
                   help="region of interest used by run_eccount (the paper uses manual)")
    b.add_argument("--own-eccount", action="store_true",
                   help="you will run ecCount yourself; its outputs go to own_eccount/")
    b.add_argument("--checkpoint", type=Path, default=None, help="ecCount weights (.pt)")
    b.add_argument("--device", default=None, help="cpu, cuda or cuda:N (written into the config)")
    b.add_argument("--strict", action="store_true",
                   help="fail if any selected row lacks its files")
    b.add_argument("--n-workers", type=int, default=1,
                   help="scoring workers for run_config.yaml and NEXT_STEPS.txt; "
                        "each needs about 10 GB of memory (default 1)")
    b.set_defaults(func=cmd_bia)

    i = sub.add_parser("images", help="run ecCount on your own images")
    i.add_argument("--images", type=Path, required=True)
    i.add_argument("--roi-dir", type=Path, default=None,
                   help="optional binary ROI masks named like the images")
    i.add_argument("--require-roi", action="store_true", help="skip images without an ROI mask")
    i.add_argument("--recursive", action="store_true")
    i.add_argument("--exclude", default=None, help="regex of file names to ignore")
    i.add_argument("--cell-line", default=None)
    i.add_argument("--out-dir", type=Path, required=True)
    i.add_argument("--base-config", default="configs/default.yaml")
    i.add_argument("--checkpoint", type=Path, default=None)
    i.add_argument("--device", default=None)
    i.set_defaults(func=cmd_images)

    c = sub.add_parser("count", help="count objects in binary masks")
    c.add_argument("--masks", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--min-area", type=int, default=3)
    c.set_defaults(func=cmd_count)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
