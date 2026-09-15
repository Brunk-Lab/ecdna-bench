#!/usr/bin/env python

"""
Build lightweight full-resource ecDNA count tables.

This script is independent from ecdna_bench.cli.build_metadata.

It does NOT use train/val/test split files.
It scans the full GS mask directory and coordinate directory, then writes:

1. full_counts_master.csv
   One row per UID with:
   unique_id, cell_line, gt_mask_relpath, coord_file_relpath,
   ecDNA_gt, coord_count_npy, count_delta, count_source

2. full_metadata.csv
   Minimal metadata-style table with:
   unique_id, cell_line, gt_mask_relpath, coord_file_relpath

Definitions
-----------
ecDNA_gt:
    Connected-component count from the rendered GS mask:
    mask > 0, 8-connectivity, min_area >= 3 px.

coord_count_npy:
    Annotation-point count from NPY/NPZ/CSV/TSV/TXT/JSON coordinate files.

count_delta:
    coord_count_npy - ecDNA_gt

Usage
-----
From repo root:

    python scripts/build_full_counts_resource.py \
        --config configs/default.yaml \
        --output-dir release/manifests \
        --force

The script automatically merges configs/paths.local.yaml over configs/default.yaml
when configs/default.yaml is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import yaml


ALLOWED_IMAGE_SUFFIXES = {
    ".png",
    ".tif",
    ".tiff",
    ".jpg",
    ".jpeg",
}

ALLOWED_COORD_SUFFIXES = {
    ".npy",
    ".npz",
    ".csv",
    ".tsv",
    ".txt",
    ".json",
}


# ==============================================================================
# Config helpers
# ==============================================================================


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Recursively merge override into base.

    Values in override win.
    """
    out = dict(base)

    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value

    return out


def read_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def load_config(config_path: Path) -> dict[str, Any]:
    """
    Load YAML config.

    If config_path is configs/default.yaml, automatically merge
    configs/paths.local.yaml on top, matching the normal project behavior.
    """
    config_path = config_path.expanduser().resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")

    cfg = read_yaml(config_path)

    if config_path.name == "default.yaml":
        local_path = config_path.parent / "paths.local.yaml"
        if local_path.exists():
            local_cfg = read_yaml(local_path)
            cfg = deep_update(cfg, local_cfg)
        else:
            print(
                f"Warning: {local_path} not found. "
                "Using default.yaml without local path overrides."
            )

    return cfg


def require_path(paths: dict[str, Any], key: str) -> Path:
    value = paths.get(key)

    if value is None or str(value).strip() == "":
        raise ValueError(
            f"Missing required config path: paths.{key}\n"
            "Use --config configs/paths.local.yaml, or make sure "
            "configs/paths.local.yaml exists and is merged over configs/default.yaml."
        )

    return Path(value).expanduser().resolve()


# ==============================================================================
# UID / file helpers
# ==============================================================================


def file_key(path: Path) -> str:
    """
    Case-insensitive file key used to match GS masks and coordinate files.

    Assumes the UID is the filename stem.
    """
    return path.stem.strip().lower()


def cell_line_from_uid(uid: str) -> str:
    """
    Infer canonical cell line name from UID.
    """
    key = uid.lower().replace("-", "").replace("_", "")

    if key.startswith("colo320dm"):
        return "COLO320DM"
    if key.startswith("snu16"):
        return "SNU16"
    if key.startswith("ncih716"):
        return "NCI-H716"
    if key.startswith("ncih2170"):
        return "NCI-H2170"
    if key.startswith("sum159pt"):
        return "SUM159PT"

    return "UNKNOWN"


def index_files(root: Path, allowed_suffixes: set[str]) -> dict[str, Path]:
    """
    Build key -> file path index from a directory tree.
    """
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    out: dict[str, Path] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        key = file_key(path)

        # Keep first file if duplicates exist.
        # Duplicate reporting is handled separately.
        out.setdefault(key, path)

    return out


def find_duplicate_keys(root: Path, allowed_suffixes: set[str]) -> dict[str, list[Path]]:
    """
    Return file keys that occur more than once.
    """
    seen: dict[str, list[Path]] = {}

    if not root.exists():
        return seen

    for path in root.rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in allowed_suffixes:
            continue

        key = file_key(path)
        seen.setdefault(key, []).append(path)

    return {key: paths for key, paths in seen.items() if len(paths) > 1}


def relpath_or_abs(path: Path | None, data_root: Path) -> str:
    """
    Return path relative to data_root when possible, otherwise absolute path.
    """
    if path is None:
        return ""

    try:
        return str(path.relative_to(data_root))
    except ValueError:
        return str(path)


def canonical_uid_for_key(
    key: str,
    gt_files: dict[str, Path],
    coord_files: dict[str, Path],
) -> str:
    """
    Use the GS filename stem as the canonical UID when available.
    Otherwise use the coordinate filename stem.
    """
    if key in gt_files:
        return gt_files[key].stem

    if key in coord_files:
        return coord_files[key].stem

    return key


# ==============================================================================
# Count helpers
# ==============================================================================


def count_mask_cc_min3(mask_path: Path) -> int | None:
    """
    Count connected components in a rendered GS mask.

    Definition:
        binary mask = image > 0
        connectivity = 8
        min foreground component area = 3 px
    """
    image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image is None:
        return None

    binary = (image > 0).astype(np.uint8)

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
        binary,
        connectivity=8,
    )

    if n_labels <= 1:
        return 0

    areas = stats[1:, cv2.CC_STAT_AREA]
    return int((areas >= 3).sum())


def count_from_coords_file(coord_path: Path) -> int | None:
    """
    Count annotation points from a coordinate file.

    Supported formats:
        .npy, .npz, .csv, .tsv, .txt, .json
    """
    suffix = coord_path.suffix.lower()

    try:
        if suffix == ".npy":
            arr = np.load(coord_path, allow_pickle=True)

            if arr.ndim == 0:
                return int(arr)

            return int(arr.shape[0])

        if suffix == ".npz":
            z = np.load(coord_path, allow_pickle=True)

            for key in (
                "coords",
                "coordinates",
                "points",
                "centroids",
                "xy",
                "arr_0",
            ):
                if key not in z:
                    continue

                arr = np.asarray(z[key])

                if arr.ndim == 0:
                    return int(arr)

                return int(arr.shape[0])

            return None

        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(coord_path, sep=sep)
            return int(len(df))

        if suffix == ".txt":
            lines = [
                line
                for line in coord_path.read_text().splitlines()
                if line.strip()
            ]
            return int(len(lines))

        if suffix == ".json":
            obj = json.loads(coord_path.read_text())

            if isinstance(obj, list):
                return int(len(obj))

            if isinstance(obj, dict):
                for key in (
                    "coords",
                    "coordinates",
                    "points",
                    "centroids",
                    "xy",
                ):
                    value = obj.get(key)
                    if isinstance(value, list):
                        return int(len(value))

            return None

    except Exception:
        return None

    return None


# ==============================================================================
# Main builder
# ==============================================================================


def build_full_resource_tables(
    *,
    data_root: Path,
    gt_mask_dir: Path,
    gt_coords_dir: Path,
    output_dir: Path,
    force: bool,
    row_universe: str,
) -> tuple[pd.DataFrame, pd.DataFrame, Path, Path, Path]:
    """
    Build full resource count and minimal metadata tables.

    row_universe:
        "union"        = GS masks OR coordinate files
        "intersection" = only UIDs with both GS mask AND coordinate file
        "gt"           = only UIDs with GS mask
        "coords"       = only UIDs with coordinate file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_out = output_dir / "full_metadata.csv"
    counts_out = output_dir / "full_counts_master.csv"
    summary_out = output_dir / "full_counts_master.summary.txt"

    if not force:
        existing = [
            path
            for path in (metadata_out, counts_out, summary_out)
            if path.exists()
        ]

        if existing:
            existing_txt = "\n".join(f"  {path}" for path in existing)
            raise FileExistsError(
                "Output file(s) already exist. Use --force to overwrite:\n"
                f"{existing_txt}"
            )

    gt_files = index_files(gt_mask_dir, ALLOWED_IMAGE_SUFFIXES)
    coord_files = index_files(gt_coords_dir, ALLOWED_COORD_SUFFIXES)

    duplicate_gt = find_duplicate_keys(gt_mask_dir, ALLOWED_IMAGE_SUFFIXES)
    duplicate_coords = find_duplicate_keys(gt_coords_dir, ALLOWED_COORD_SUFFIXES)

    gt_keys = set(gt_files)
    coord_keys = set(coord_files)

    if row_universe == "union":
        keys = sorted(gt_keys | coord_keys)
    elif row_universe == "intersection":
        keys = sorted(gt_keys & coord_keys)
    elif row_universe == "gt":
        keys = sorted(gt_keys)
    elif row_universe == "coords":
        keys = sorted(coord_keys)
    else:
        raise ValueError(f"Invalid row_universe: {row_universe}")

    rows: list[dict[str, Any]] = []

    for key in keys:
        uid = canonical_uid_for_key(key, gt_files, coord_files)

        gt_path = gt_files.get(key)
        coord_path = coord_files.get(key)

        ecdna_gt = count_mask_cc_min3(gt_path) if gt_path is not None else None
        coord_count_npy = (
            count_from_coords_file(coord_path)
            if coord_path is not None
            else None
        )

        if ecdna_gt is not None and coord_count_npy is not None:
            count_delta: int | float = int(coord_count_npy) - int(ecdna_gt)
        else:
            count_delta = np.nan

        rows.append(
            {
                "unique_id": uid,
                "cell_line": cell_line_from_uid(uid),
                "gt_mask_relpath": relpath_or_abs(gt_path, data_root),
                "coord_file_relpath": relpath_or_abs(coord_path, data_root),
                "ecDNA_gt": int(ecdna_gt) if ecdna_gt is not None else np.nan,
                "coord_count_npy": (
                    int(coord_count_npy)
                    if coord_count_npy is not None
                    else np.nan
                ),
                "count_delta": count_delta,
                "count_source": "gt_mask_cc_conn8_min3",
            }
        )

    counts_df = pd.DataFrame(rows)

    if len(counts_df) == 0:
        raise RuntimeError(
            "No rows were found. Check gt_mask_dir and gt_coords_dir:\n"
            f"  gt_mask_dir   = {gt_mask_dir}\n"
            f"  gt_coords_dir = {gt_coords_dir}"
        )

    metadata_df = counts_df[
        [
            "unique_id",
            "cell_line",
            "gt_mask_relpath",
            "coord_file_relpath",
        ]
    ].copy()

    counts_df.to_csv(counts_out, index=False)
    metadata_df.to_csv(metadata_out, index=False)

    summary = make_summary(
        counts_df=counts_df,
        data_root=data_root,
        gt_mask_dir=gt_mask_dir,
        gt_coords_dir=gt_coords_dir,
        row_universe=row_universe,
        duplicate_gt=duplicate_gt,
        duplicate_coords=duplicate_coords,
    )

    summary_out.write_text(summary + "\n")

    return metadata_df, counts_df, metadata_out, counts_out, summary_out


def make_summary(
    *,
    counts_df: pd.DataFrame,
    data_root: Path,
    gt_mask_dir: Path,
    gt_coords_dir: Path,
    row_universe: str,
    duplicate_gt: dict[str, list[Path]],
    duplicate_coords: dict[str, list[Path]],
) -> str:
    lines: list[str] = []

    n_rows = len(counts_df)
    has_gt = counts_df["gt_mask_relpath"].astype(str).str.strip().ne("")
    has_coords = counts_df["coord_file_relpath"].astype(str).str.strip().ne("")

    lines.append("FULL RESOURCE COUNT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"data_root                : {data_root}")
    lines.append(f"gt_mask_dir              : {gt_mask_dir}")
    lines.append(f"gt_coords_dir            : {gt_coords_dir}")
    lines.append(f"row_universe             : {row_universe}")
    lines.append("")
    lines.append(f"Rows total               : {n_rows}")
    lines.append(f"Unique IDs               : {counts_df['unique_id'].nunique()}")
    lines.append(f"Rows with GS mask         : {int(has_gt.sum())} / {n_rows}")
    lines.append(f"Rows with coord file      : {int(has_coords.sum())} / {n_rows}")
    lines.append(f"Rows missing GS mask      : {int((~has_gt).sum())} / {n_rows}")
    lines.append(f"Rows missing coord file   : {int((~has_coords).sum())} / {n_rows}")
    lines.append("")
    lines.append("CELL LINE COUNTS")
    lines.append("-" * 80)
    lines.append(counts_df["cell_line"].value_counts(dropna=False).to_string())
    lines.append("")
    lines.append("COUNT COMPLETENESS")
    lines.append("-" * 80)
    lines.append(
        f"ecDNA_gt non-null         : "
        f"{counts_df['ecDNA_gt'].notna().sum()} / {n_rows}"
    )
    lines.append(
        f"coord_count_npy non-null  : "
        f"{counts_df['coord_count_npy'].notna().sum()} / {n_rows}"
    )

    both = counts_df["ecDNA_gt"].notna() & counts_df["coord_count_npy"].notna()

    if both.any():
        diff = counts_df.loc[both, "count_delta"]

        lines.append("")
        lines.append("GS MASK COUNT VS COORD COUNT")
        lines.append("-" * 80)
        lines.append(f"Rows with both counts     : {int(both.sum())}")
        lines.append(f"Rows with different counts: {int((diff != 0).sum())}")
        lines.append(f"Median abs difference     : {float(diff.abs().median()):.3f}")
        lines.append(f"Mean abs difference       : {float(diff.abs().mean()):.3f}")
        lines.append(f"Max abs difference        : {float(diff.abs().max()):.3f}")

        greater = counts_df.loc[both, "coord_count_npy"] > counts_df.loc[both, "ecDNA_gt"]
        equal = counts_df.loc[both, "coord_count_npy"] == counts_df.loc[both, "ecDNA_gt"]
        lower = counts_df.loc[both, "coord_count_npy"] < counts_df.loc[both, "ecDNA_gt"]

        lines.append(f"coord_count_npy > ecDNA_gt: {int(greater.sum())}")
        lines.append(f"coord_count_npy = ecDNA_gt: {int(equal.sum())}")
        lines.append(f"coord_count_npy < ecDNA_gt: {int(lower.sum())}")

    lines.append("")
    lines.append("DUPLICATE FILE-STEM CHECKS")
    lines.append("-" * 80)
    lines.append(f"Duplicate GS stems         : {len(duplicate_gt)}")
    lines.append(f"Duplicate coord stems      : {len(duplicate_coords)}")

    if duplicate_gt:
        lines.append("")
        lines.append("Example duplicate GS stems")
        lines.append("-" * 80)
        for key, paths in list(duplicate_gt.items())[:10]:
            lines.append(f"{key}:")
            for path in paths:
                lines.append(f"  {path}")

    if duplicate_coords:
        lines.append("")
        lines.append("Example duplicate coord stems")
        lines.append("-" * 80)
        for key, paths in list(duplicate_coords.items())[:10]:
            lines.append(f"{key}:")
            for path in paths:
                lines.append(f"  {path}")

    return "\n".join(lines)


# ==============================================================================
# CLI
# ==============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build lightweight full-resource ecDNA count tables."
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help=(
            "Config YAML. If configs/default.yaml is used, "
            "configs/paths.local.yaml is automatically merged on top."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("release/manifests"),
        help="Output directory for full_metadata.csv and full_counts_master.csv.",
    )

    parser.add_argument(
        "--row-universe",
        choices=["union", "intersection", "gt", "coords"],
        default="union",
        help=(
            "Which UIDs to include. "
            "union = GS masks OR coord files; "
            "intersection = both; "
            "gt = GS masks only; "
            "coords = coord files only."
        ),
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing outputs.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})

    data_root = require_path(paths, "data_root")
    gt_mask_dir = require_path(paths, "gt_mask_dir")

    # Prefer explicit gt_coords_dir. If absent, assume sibling of gt_mask_dir.
    if paths.get("gt_coords_dir"):
        gt_coords_dir = Path(paths["gt_coords_dir"]).expanduser().resolve()
    else:
        gt_coords_dir = (gt_mask_dir.parent / "gt_coords").resolve()

    output_dir = args.output_dir.expanduser().resolve()

    (
        metadata_df,
        counts_df,
        metadata_out,
        counts_out,
        summary_out,
    ) = build_full_resource_tables(
        data_root=data_root,
        gt_mask_dir=gt_mask_dir,
        gt_coords_dir=gt_coords_dir,
        output_dir=output_dir,
        force=args.force,
        row_universe=args.row_universe,
    )

    print(f"Wrote metadata : {metadata_out}  ({len(metadata_df)} rows)")
    print(f"Wrote counts   : {counts_out}  ({len(counts_df)} rows)")
    print(f"Wrote summary  : {summary_out}")
    print("")
    print(summary_out.read_text())


if __name__ == "__main__":
    main()