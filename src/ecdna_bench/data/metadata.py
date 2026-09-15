"""
ecdna_bench.data.metadata — build and validate the master metadata table.

Most users never call this module: the released dataset ships with a
ready-to-use ``metadata.csv`` that already merges cell-line, split, and path
information.  This module exists for (a) rebuilding the table if the raw
inputs change, and (b) verifying consistency of what the release ships —
it is the canonical definition of what the master metadata table should look
like.

The master metadata table has one row per metaphase UID and the following
columns (at a minimum):

    unique_id                str     canonical UID
    cell_line                str     canonical cell-line name
    split                    str     'train' / 'val' / 'test' (or blank for
                                     full-resource-only images)
    rgb_relpath              str     path relative to ``data_root``
    dapi_relpath             str
    roi_mask_relpath         str     (1,145-subset only; blank otherwise)
    gt_mask_relpath          str     rendered gold-standard binary mask
    ecDNA_gt                 int     CANONICAL per-image ecDNA count:
                                     8-connectivity connected components of
                                     the rendered GS mask, min_area=3 px.
                                     This is what every model in the benchmark
                                     is evaluated against.
    coord_count_npy          int     annotation-point count from NPY/NPZ files.
                                     Released for transparency.  Always >= ecDNA_gt;
                                     differences arise from the diamond-merge
                                     effect (two close annotation points whose
                                     7×7 diamonds overlap become a single CC).
    split_from_id_csv        str     redundant 'split' computed from split files
    split_consistent         bool    True iff split == split_from_id_csv
    in_train_ids / in_val_ids / in_test_ids : bool

This module provides ``build_master_metadata`` (pure function) and the
helper validators ``validate_metadata``, ``validate_counts``, and
``validate_splits_disjoint``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ecdna_bench.utils.log_utils import get_logger

if TYPE_CHECKING:
    import pandas as pd


logger = get_logger(__name__)


# ==============================================================================
# Expected column sets
# ==============================================================================


EXPECTED_METADATA_COLS: frozenset[str] = frozenset(
    {
        "unique_id",
        "cell_line",
        "split",
        "rgb_relpath",
        "dapi_relpath",
        "roi_mask_relpath",
        "gt_mask_relpath",
    }
)

EXPECTED_COUNTS_COLS: frozenset[str] = frozenset(
    {
        "unique_id",
        "ecDNA_gt",
    }
)

EXPECTED_SPLIT_ID_COLS: frozenset[str] = frozenset({"unique_id"})


# Ordered list used when writing the master CSV; columns present in the
# merged DataFrame but not in this list are appended at the end.
PREFERRED_COLUMN_ORDER: tuple[str, ...] = (
    "unique_id",
    "cell_line",
    "split",
    "split_from_id_csv",
    "split_consistent",
    "in_train_ids",
    "in_val_ids",
    "in_test_ids",
    "n_split_flags",
    "rgb_relpath",
    "dapi_relpath",
    "roi_mask_relpath",
    "gt_mask_relpath",
    "mia_mask_relpath",
    # ecDNA_gt is the CANONICAL evaluation count: connected components of the
    # rendered 7×7-diamond GS mask under 8-connectivity with min_area=3 px.
    # This is what every model in the benchmark predicts and is compared against.
    # See PROJECT_RULES.md §2 and §7.
    "ecDNA_gt",
    # coord_count_npy is the annotation-point count from NPY/NPZ files.
    # Released for transparency but NOT used for model evaluation.
    # coord_count_npy >= ecDNA_gt always; differences arise when adjacent
    # annotation points are close enough that their 7×7 diamonds merge into
    # a single connected component (the "diamond-merge" effect).
    "coord_count_npy",
    "has_rgb_relpath",
    "has_dapi_relpath",
    "has_roi_mask_relpath",
    "has_gt_mask_relpath",
    "original_rgb_name",
    "original_dapi_name",
    "original_roi_name",
    "original_gt_name",
    "original_mia_name",
)


# ==============================================================================
# Validators
# ==============================================================================


class MetadataError(ValueError):
    """Raised when a metadata input fails validation."""


def _require_columns(df: "pd.DataFrame", required: frozenset[str], source: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise MetadataError(
            f"{source} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def validate_metadata(df: "pd.DataFrame", *, source: str = "metadata CSV") -> None:
    """
    Validate a raw metadata DataFrame.

    Raises
    ------
    MetadataError
        If required columns are missing or if ``unique_id`` contains duplicates.
    """
    _require_columns(df, EXPECTED_METADATA_COLS, source)
    if not df["unique_id"].astype(str).is_unique:
        dupes = (
            df.loc[df["unique_id"].astype(str).duplicated(), "unique_id"]
            .astype(str)
            .tolist()[:20]
        )
        raise MetadataError(
            f"{source} has duplicate unique_id values. Examples: {dupes}"
        )


def validate_counts(df: "pd.DataFrame", *, source: str = "counts CSV") -> None:
    """Validate a raw per-image counts DataFrame.

    Required columns: ``unique_id``, ``ecDNA_gt``.
    Optional column:  ``coord_count_npy`` (annotation-point count from NPY/NPZ).
    """
    _require_columns(df, EXPECTED_COUNTS_COLS, source)
    if not df["unique_id"].astype(str).is_unique:
        dupes = (
            df.loc[df["unique_id"].astype(str).duplicated(), "unique_id"]
            .astype(str)
            .tolist()[:20]
        )
        raise MetadataError(
            f"{source} has duplicate unique_id values. Examples: {dupes}"
        )


def validate_splits_disjoint(
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> None:
    """Raise if any pair of split ID lists share a UID."""
    tr, va, te = set(train_ids), set(val_ids), set(test_ids)
    overlap_tv = tr & va
    overlap_tt = tr & te
    overlap_vt = va & te
    if overlap_tv or overlap_tt or overlap_vt:
        raise MetadataError(
            "Split ID lists are not disjoint.\n"
            f"  train ∩ val  : {len(overlap_tv)}\n"
            f"  train ∩ test : {len(overlap_tt)}\n"
            f"  val ∩ test   : {len(overlap_vt)}"
        )


# ==============================================================================
# Builder
# ==============================================================================


def _split_from_flags(in_train: bool, in_val: bool, in_test: bool) -> str:
    if in_train:
        return "train"
    if in_val:
        return "val"
    if in_test:
        return "test"
    return "missing"


def _nonempty_string_flag(series: "pd.Series") -> "pd.Series":
    return series.notna() & series.astype(str).str.strip().ne("")


def build_master_metadata(
    metadata_df: "pd.DataFrame",
    counts_df: "pd.DataFrame",
    train_ids: list[str],
    val_ids: list[str],
    test_ids: list[str],
) -> "pd.DataFrame":
    """
    Merge metadata, counts, and split-ID lists into the master metadata table.

    Parameters
    ----------
    metadata_df
        DataFrame with modality paths and cell-line / split labels. Must
        carry the columns in ``EXPECTED_METADATA_COLS``.
    counts_df
        DataFrame with per-UID ecDNA counts. Must carry ``unique_id`` and
        ``ecDNA_gt``. Optionally carries ``coord_count_npy``, which is passed
        through to the master table when present.
    train_ids, val_ids, test_ids
        Lists of UIDs for each split. Must be pairwise disjoint.

    Returns
    -------
    pd.DataFrame
        One row per UID, with the columns documented at the top of this
        module. Column order follows ``PREFERRED_COLUMN_ORDER``.

    Raises
    ------
    MetadataError
        If any input is malformed or if splits overlap.
    """
    import pandas as pd  # noqa: F401 — imported lazily

    validate_metadata(metadata_df)
    validate_counts(counts_df)
    validate_splits_disjoint(train_ids, val_ids, test_ids)

    meta   = metadata_df.copy()
    counts = counts_df.copy()
    meta["unique_id"]   = meta["unique_id"].astype(str)
    counts["unique_id"] = counts["unique_id"].astype(str)

    # Always carry ecDNA_gt.  Also carry coord_count_npy when present.
    merge_cols = ["unique_id", "ecDNA_gt"]
    if "coord_count_npy" in counts.columns:
        merge_cols.append("coord_count_npy")

    df = meta.merge(counts[merge_cols], on="unique_id", how="left", validate="one_to_one")

    train_set = set(map(str, train_ids))
    val_set   = set(map(str, val_ids))
    test_set  = set(map(str, test_ids))

    df["in_train_ids"] = df["unique_id"].isin(train_set)
    df["in_val_ids"]   = df["unique_id"].isin(val_set)
    df["in_test_ids"]  = df["unique_id"].isin(test_set)
    df["n_split_flags"] = (
        df["in_train_ids"].astype(int)
        + df["in_val_ids"].astype(int)
        + df["in_test_ids"].astype(int)
    )
    df["split_from_id_csv"] = [
        _split_from_flags(t, v, te)
        for t, v, te in zip(df["in_train_ids"], df["in_val_ids"], df["in_test_ids"])
    ]
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["split_consistent"] = df["split"] == df["split_from_id_csv"]

    for col in ("rgb_relpath", "dapi_relpath", "roi_mask_relpath", "gt_mask_relpath"):
        df[f"has_{col}"] = _nonempty_string_flag(df[col])

    remaining = [c for c in df.columns if c not in PREFERRED_COLUMN_ORDER]
    ordered   = [c for c in PREFERRED_COLUMN_ORDER if c in df.columns] + remaining
    return df[ordered]


def summarize_master_metadata(df: "pd.DataFrame") -> str:
    """
    Produce a human-readable summary string for a master metadata table.

    This is the text version that used to be written by
    ``stage1_step1_build_master_metadata.py``.
    """
    lines: list[str] = []
    total = len(df)

    lines.append("MASTER METADATA SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Rows                : {total}")
    lines.append(f"Unique IDs          : {df['unique_id'].nunique()}")
    lines.append("")

    lines.append("SPLIT COUNTS (metadata 'split' column)")
    lines.append("-" * 80)
    lines.append(df["split"].astype(str).value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("SPLIT COUNTS (from train/val/test ID CSVs)")
    lines.append("-" * 80)
    lines.append(df["split_from_id_csv"].astype(str).value_counts(dropna=False).to_string())
    lines.append("")

    lines.append("SPLIT CONSISTENCY")
    lines.append("-" * 80)
    lines.append(df["split_consistent"].value_counts(dropna=False).to_string())
    lines.append(f"Rows with n_split_flags != 1 : {(df['n_split_flags'] != 1).sum()}")
    lines.append("")

    if "cell_line" in df.columns:
        lines.append("CELL LINE COUNTS")
        lines.append("-" * 80)
        lines.append(df["cell_line"].astype(str).value_counts(dropna=False).to_string())
        lines.append("")

    lines.append("PATH-PRESENCE FLAGS")
    lines.append("-" * 80)
    for col in ("rgb_relpath", "dapi_relpath", "roi_mask_relpath", "gt_mask_relpath"):
        flag = f"has_{col}"
        if flag in df.columns:
            n = int(df[flag].astype(bool).sum())
            lines.append(f"{flag:<24}: {n} / {total}")
    if "ecDNA_gt" in df.columns:
        lines.append(f"ecDNA_gt non-null         : {df['ecDNA_gt'].notna().sum()} / {total}")
        lines.append(f"ecDNA_gt missing          : {df['ecDNA_gt'].isna().sum()} / {total}")
        lines.append(f"  (ecDNA_gt = GS mask CCs, 8-conn, min_area=3 — evaluation target)")
    if "coord_count_npy" in df.columns:
        npy_nn = df["coord_count_npy"].notna().sum()
        lines.append(f"coord_count_npy non-null  : {npy_nn} / {total}")
        if npy_nn > 0 and "ecDNA_gt" in df.columns:
            diff   = (df["coord_count_npy"] - df["ecDNA_gt"]).dropna()
            n_diff = int((diff != 0).sum())
            lines.append(
                f"  (coord_count_npy > ecDNA_gt in {n_diff} images — diamond-merge effect)"
            )
    lines.append("")

    lines.append("DUPLICATE CHECKS")
    lines.append("-" * 80)
    lines.append(f"Duplicate unique_id rows  : {df['unique_id'].duplicated().sum()}")

    return "\n".join(lines)


def resolve_path_columns(
    df: "pd.DataFrame",
    data_root: Path | str,
    *,
    relpath_to_fullpath: dict[str, str] | None = None,
) -> "pd.DataFrame":
    """
    Add ``*_fullpath`` columns computed from ``data_root`` and ``*_relpath`` columns.

    Useful when going from the shipped ``metadata.csv`` to the absolute paths
    that ``samples_from_metadata_df`` uses. Returns a copy; the input is not
    mutated.

    Parameters
    ----------
    df
        Metadata DataFrame carrying ``*_relpath`` columns.
    data_root
        Absolute directory against which relative paths are resolved.
    relpath_to_fullpath
        Optional mapping of input column → output column. Defaults to:
        ``rgb_relpath→rgb_fullpath``, ``dapi_relpath→dapi_fullpath``,
        ``roi_mask_relpath→roi_fullpath``, ``gt_mask_relpath→gt_fullpath``.
    """
    import pandas as pd  # noqa: F401

    data_root = Path(data_root)
    mapping = relpath_to_fullpath or {
        "rgb_relpath":     "rgb_fullpath",
        "dapi_relpath":    "dapi_fullpath",
        "roi_mask_relpath": "roi_fullpath",
        "gt_mask_relpath": "gt_fullpath",
    }
    out = df.copy()
    for rel_col, full_col in mapping.items():
        if rel_col not in out.columns:
            continue

        def _join(v: object) -> str:
            if v is None:
                return ""
            s = str(v).strip()
            if not s or s.lower() == "nan":
                return ""
            p = Path(s)
            return str(p if p.is_absolute() else (data_root / p).resolve())

        out[full_col] = out[rel_col].map(_join)
    return out


# ==============================================================================
# Filesystem builder — called by cli/build_metadata.py
# ==============================================================================

def build_metadata(
    *,
    rgb_dir,
    gt_mask_dir,
    split_dir,
    dapi_dir=None,
    roi_mask_dir=None,
    mia_mask_dir=None,
    metadata_out=None,
    counts_out=None,
):
    """
    Build ``metadata.csv`` and ``counts_master.csv`` from the local file tree.

    This is the filesystem-facing wrapper called by
    ``ecdna_bench.cli.build_metadata``.  It uses the official split files as
    the row universe, resolves modality paths by UID, and creates the counts
    table with the canonical count definition.

    Count columns in ``counts_master.csv``
    ---------------------------------------
    ecDNA_gt
        Canonical evaluation target: 8-connectivity connected components of
        the rendered GS mask with min_area=3 px.  Matches the evaluation
        framework in PROJECT_RULES.md §2.  This is what every model in the
        benchmark predicts and is evaluated against.
    coord_count_npy
        Annotation-point count read from the sibling NPY/NPZ coordinate file
        (``gt_coords/`` directory).  Released for transparency.  Always
        ``>= ecDNA_gt``; the difference is the diamond-merge effect.
    count_source
        Always ``"gt_mask_cc_conn8_min3"`` to document the canonical
        definition unambiguously.
    """
    from pathlib import Path
    import json

    import cv2
    import numpy as np
    import pandas as pd

    rgb_dir     = Path(rgb_dir).resolve()
    gt_mask_dir = Path(gt_mask_dir).resolve()
    split_dir   = Path(split_dir).resolve()

    dapi_dir     = Path(dapi_dir).resolve()     if dapi_dir     else None
    roi_mask_dir = Path(roi_mask_dir).resolve() if roi_mask_dir else None
    mia_mask_dir = Path(mia_mask_dir).resolve() if mia_mask_dir else None

    metadata_out = (
        Path(metadata_out).resolve() if metadata_out else Path("metadata.csv").resolve()
    )
    counts_out = (
        Path(counts_out).resolve() if counts_out else Path("counts_master.csv").resolve()
    )

    data_root    = rgb_dir.parent
    gt_coords_dir = gt_mask_dir.parent / "gt_coords"

    # ------------------------------------------------------------------
    # Read split files
    # ------------------------------------------------------------------

    def _read_ids(filename: str) -> list[str]:
        path = split_dir / filename
        df = pd.read_csv(path)
        if "unique_id" not in df.columns:
            df = df.rename(columns={df.columns[0]: "unique_id"})
        return df["unique_id"].astype(str).tolist()

    train_ids = _read_ids("train_ids.csv")
    val_ids   = _read_ids("val_ids.csv")
    test_ids  = _read_ids("test_ids.csv")

    split_by_uid: dict[str, str] = {}
    for uid in train_ids:
        split_by_uid[uid] = "train"
    for uid in val_ids:
        split_by_uid[uid] = "val"
    for uid in test_ids:
        split_by_uid[uid] = "test"

    uids = train_ids + val_ids + test_ids

    # ------------------------------------------------------------------
    # Build file indices  (stem.lower() → Path)
    # ------------------------------------------------------------------

    def _index_files(root: Path | None) -> dict[str, Path]:
        if root is None or not root.exists():
            return {}
        allowed = {
            ".tif", ".tiff", ".png", ".jpg", ".jpeg",
            ".csv", ".tsv", ".txt", ".json", ".npy", ".npz",
        }
        out: dict[str, Path] = {}
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in allowed:
                out.setdefault(p.stem.lower(), p)
        return out

    rgb_files   = _index_files(rgb_dir)
    dapi_files  = _index_files(dapi_dir)
    gt_files    = _index_files(gt_mask_dir)
    roi_files   = _index_files(roi_mask_dir)
    mia_files   = _index_files(mia_mask_dir)
    coord_files = _index_files(gt_coords_dir)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _relpath(p: Path | None) -> str:
        if p is None:
            return ""
        try:
            return str(p.relative_to(data_root))
        except ValueError:
            return str(p)

    def _cell_line_from_uid(uid: str) -> str:
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

    def _count_mask_cc_min3(p: Path) -> int | None:
        """Count 8-connectivity CCs with min_area=3 px in a GS mask.

        This is the canonical count definition (PROJECT_RULES.md §2 / §7).
        Every model in the benchmark predicts mask CCs, so the evaluation
        target must also be mask CCs computed with identical parameters.

        Returns
        -------
        int
            Number of connected components with area >= 3 px, or 0 if the
            mask is empty.
        None
            If the file cannot be read.
        """
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        binary = (img > 0).astype(np.uint8)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        if n_labels <= 1:
            return 0
        # stats[0] is background; stats[1:] are foreground components.
        areas = stats[1:, cv2.CC_STAT_AREA]
        return int((areas >= 3).sum())

    def _count_from_coords_file(p: Path) -> int | None:
        """Read annotation-point count from a coordinate file (NPY/NPZ/CSV/…).

        Returns the number of annotated points, or None on any failure.
        This value is stored as coord_count_npy — it is NOT the evaluation
        target, but is released for transparency and for the diamond-merge
        supplementary figure.
        """
        suffix = p.suffix.lower()
        try:
            if suffix in {".csv", ".tsv"}:
                sep = "\t" if suffix == ".tsv" else ","
                df = pd.read_csv(p, sep=sep)
                return int(len(df))

            if suffix == ".json":
                obj = json.loads(p.read_text())
                if isinstance(obj, list):
                    return int(len(obj))
                if isinstance(obj, dict):
                    for key in ("coords", "coordinates", "points", "centroids", "xy"):
                        if key in obj and isinstance(obj[key], list):
                            return int(len(obj[key]))
                return None

            if suffix == ".npz":
                z = np.load(p, allow_pickle=True)
                for key in ("coords", "coordinates", "points", "centroids", "xy", "arr_0"):
                    if key in z:
                        arr = np.asarray(z[key])
                        if arr.ndim == 0:
                            return int(arr)
                        return int(arr.shape[0])
                return None

            if suffix == ".npy":
                arr = np.load(p, allow_pickle=True)
                if arr.ndim == 0:
                    return int(arr)
                return int(arr.shape[0])

            if suffix == ".txt":
                lines = [x for x in p.read_text().splitlines() if x.strip()]
                return int(len(lines))

        except Exception:
            return None

        return None

    # ------------------------------------------------------------------
    # Main per-UID loop
    # ------------------------------------------------------------------

    metadata_rows: list[dict] = []
    counts_rows:   list[dict] = []
    missing_required: list[tuple[str, str]] = []

    for uid in uids:
        key = uid.lower()

        rgb    = rgb_files.get(key)
        dapi   = dapi_files.get(key)
        gt     = gt_files.get(key)
        roi    = roi_files.get(key)
        mia    = mia_files.get(key)
        coords = coord_files.get(key)

        if rgb is None:
            missing_required.append((uid, "rgb"))
        if gt is None:
            missing_required.append((uid, "gt_mask"))

        # ------------------------------------------------------------------
        # ecDNA_gt — CANONICAL COUNT (evaluation target)
        # Always derived from the rendered GS mask: 8-conn CCs, min_area=3 px.
        # This is what every model predicts, so we evaluate against this.
        # ------------------------------------------------------------------
        ecdna_gt: int | None = None
        if gt is not None:
            ecdna_gt = _count_mask_cc_min3(gt)

        # ------------------------------------------------------------------
        # coord_count_npy — annotation-point count (transparency only)
        # Read from NPY/NPZ coordinate files when available.
        # coord_count_npy >= ecDNA_gt always (diamond-merge can only reduce
        # the CC count, never inflate it).
        # ------------------------------------------------------------------
        coord_count_npy: int | None = None
        if coords is not None:
            coord_count_npy = _count_from_coords_file(coords)

        if ecdna_gt is None:
            missing_required.append((uid, "ecDNA_gt"))

        split     = split_by_uid[uid]
        cell_line = _cell_line_from_uid(uid)

        metadata_rows.append(
            {
                "unique_id":        uid,
                "cell_line":        cell_line,
                "split":            split,
                "rgb_relpath":      _relpath(rgb),
                "dapi_relpath":     _relpath(dapi),
                "roi_mask_relpath": _relpath(roi),
                "gt_mask_relpath":  _relpath(gt),
                "mia_mask_relpath": _relpath(mia),
                "rgb_fullpath":     str(rgb)  if rgb  else "",
                "dapi_fullpath":    str(dapi) if dapi else "",
                "roi_fullpath":     str(roi)  if roi  else "",
                "gt_fullpath":      str(gt)   if gt   else "",
                "original_rgb_name":  rgb.name  if rgb  else "",
                "original_dapi_name": dapi.name if dapi else "",
                "original_roi_name":  roi.name  if roi  else "",
                "original_gt_name":   gt.name   if gt   else "",
                "original_mia_name":  mia.name  if mia  else "",
            }
        )

        counts_rows.append(
            {
                "unique_id":       uid,
                "cell_line":       cell_line,
                "split":           split,
                # Canonical evaluation target: mask CCs (8-conn, min_area=3 px)
                "ecDNA_gt":        int(ecdna_gt)        if ecdna_gt        is not None else np.nan,
                # Annotation-point count: transparency / supplementary only
                "coord_count_npy": int(coord_count_npy) if coord_count_npy is not None else np.nan,
                "count_source":    "gt_mask_cc_conn8_min3",
            }
        )

    if missing_required:
        preview = "\n".join(
            f"  {uid}: missing {kind}" for uid, kind in missing_required[:30]
        )
        raise MetadataError(
            f"Cannot build metadata; found {len(missing_required)} missing required items.\n"
            f"First examples:\n{preview}"
        )

    # ------------------------------------------------------------------
    # Assemble and write outputs
    # ------------------------------------------------------------------

    metadata_df = pd.DataFrame(metadata_rows)
    counts_df   = pd.DataFrame(counts_rows)

    # Pass both ecDNA_gt and coord_count_npy into the master so the column
    # appears in metadata.csv under PREFERRED_COLUMN_ORDER.
    master_df = build_master_metadata(
        metadata_df = metadata_df,
        counts_df   = counts_df[["unique_id", "ecDNA_gt", "coord_count_npy"]],
        train_ids   = train_ids,
        val_ids     = val_ids,
        test_ids    = test_ids,
    )

    metadata_out.parent.mkdir(parents=True, exist_ok=True)
    counts_out.parent.mkdir(parents=True, exist_ok=True)

    master_df.to_csv(metadata_out, index=False)
    counts_df.to_csv(counts_out,   index=False)

    summary_path = metadata_out.with_suffix(".summary.txt")
    summary_path.write_text(summarize_master_metadata(master_df) + "\n")

    logger.info("metadata written  : %s  (%d rows)", metadata_out, len(master_df))
    logger.info("counts written    : %s  (%d rows)", counts_out,   len(counts_df))
    logger.info("summary written   : %s", summary_path)

    return master_df, counts_df


__all__ = [
    "MetadataError",
    "EXPECTED_METADATA_COLS",
    "EXPECTED_COUNTS_COLS",
    "EXPECTED_SPLIT_ID_COLS",
    "PREFERRED_COLUMN_ORDER",
    "validate_metadata",
    "validate_counts",
    "validate_splits_disjoint",
    "build_master_metadata",
    "summarize_master_metadata",
    "resolve_path_columns",
    "build_metadata",
]