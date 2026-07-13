"""
ecdna_bench.data.samples — the `ImageSample` dataclass and filter helpers.

A `Sample` is a lightweight, in-memory record that carries the resolved
paths to all modalities of one metaphase spread, plus its UID, cell-line
label, split membership, and ground-truth ecDNA count. Samples do NOT carry
image arrays — they are path-level handles that downstream code uses to
demand-load images only when needed.

This module is the interface between:

- the metadata CSV (which lives in `data/metadata.py`), and
- the inference / training loops in `classical/`, `eccount/`, `baselines/`.

Design notes
------------
- `Sample` is a regular (not frozen) dataclass because the `extra` field
  is a dict that callers sometimes mutate.
- Two builders are provided: `samples_from_metadata_df` (from an in-memory
  DataFrame that already contains resolved paths, i.e., the Stage 1 Step 2+
  audit CSV) and `samples_from_folders` (for the legacy path-discovery flow
  used when rebuilding the metadata table from raw directories).
- Filter helpers return a *new list* rather than mutating — samples are
  cheap; in-place filtering invites aliasing bugs when the same sample list
  feeds multiple pipelines.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    import pandas as pd


Split = Literal["train", "val", "test"]


# ==============================================================================
# Dataclass
# ==============================================================================


@dataclass
class Sample:
    """
    Path-level handle for one metaphase image set.

    Attributes
    ----------
    uid
        Unique identifier for this metaphase spread. Unique across the
        entire dataset.
    rgb_path
        Path to the RGB FISH image (2,048 × 2,448 TIFF at native resolution).
    dapi_path
        Path to the paired DAPI image.
    roi_mask_path
        Path to the manual ROI mask (only present for the 1,145-image
        benchmark subset).
    gt_mask_path
        Path to the rendered ground-truth binary mask (7 × 7 diamonds at
        each annotated centroid).
    cell_line
        Canonical cell-line name, e.g. ``"NCIH2170"``. See
        `ecdna_bench.data.ids.KNOWN_CELL_LINES`.
    split
        Split membership: ``"train"``, ``"val"``, ``"test"``, or ``None``
        for images that are in the full resource but not the benchmark
        subset.
    ecdna_gt
        Ground-truth ecDNA count (number of annotated centroids).
    extra
        Any additional per-sample metadata (e.g. the `count_mask_consistent`
        flag from QC, or `original_rgb_name` for dataset auditing).
    """

    uid: str
    rgb_path: Path | None = None
    dapi_path: Path | None = None
    roi_mask_path: Path | None = None
    gt_mask_path: Path | None = None
    cell_line: str | None = None
    split: Split | None = None
    ecdna_gt: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Normalize str paths to Path; leave None alone.
        for attr in ("rgb_path", "dapi_path", "roi_mask_path", "gt_mask_path"):
            v = getattr(self, attr)
            if v is not None and not isinstance(v, Path):
                setattr(self, attr, Path(v))

    @property
    def has_roi_mask(self) -> bool:
        return self.roi_mask_path is not None

    @property
    def has_gt_mask(self) -> bool:
        return self.gt_mask_path is not None

    def required_paths_exist(self, *, require_roi: bool = True) -> bool:
        """
        True iff every non-None resolved path actually exists on disk.
        Optionally require `roi_mask_path` to be set AND exist.
        """
        for attr in ("rgb_path", "dapi_path", "gt_mask_path"):
            p = getattr(self, attr)
            if p is None or not p.exists():
                return False
        if require_roi:
            if self.roi_mask_path is None or not self.roi_mask_path.exists():
                return False
        return True


# Backward-compatible alias so legacy calls that used `ImageSample` still work.
ImageSample = Sample


# ==============================================================================
# Builders
# ==============================================================================


# Column-name conventions of the Stage 1 master metadata CSV.
# If these change in the dataset release, change them here in one place and
# every downstream workflow follows.
DEFAULT_COLUMNS = {
    "uid": "unique_id",
    "cell_line": "cell_line",
    "split": "split",
    "ecdna_gt": "ecDNA_gt",
    "rgb_fullpath": "rgb_fullpath",
    "dapi_fullpath": "dapi_fullpath",
    "roi_fullpath": "roi_fullpath",
    "gt_fullpath": "gt_fullpath",
    # Fallback columns if *_fullpath is not present (older metadata shapes).
    "rgb_relpath": "rgb_relpath",
    "dapi_relpath": "dapi_relpath",
    "roi_relpath": "roi_mask_relpath",
    "gt_relpath": "gt_mask_relpath",
}

# Extra columns, if present, get stashed into `Sample.extra` for provenance.
# Adding a column name here is the preferred way to extend what QC / training
# loops can see without touching the Sample class itself.
EXTRA_COLUMNS: tuple[str, ...] = (
    "file_audit_pass",
    "count_mask_consistent",
    "original_rgb_name",
    "original_dapi_name",
    "original_roi_name",
    "original_gt_name",
    "split_consistent",
    "rgb_h",
    "rgb_w",
    "gt_nonzero_pixels",
    "roi_nonzero_pixels",
)


def _resolve(path_val: Any, data_root: Path | None) -> Path | None:
    """Resolve a path value from a CSV cell to an absolute Path or None."""
    if path_val is None:
        return None
    if isinstance(path_val, float):  # pandas NaN
        import math
        if math.isnan(path_val):
            return None
    s = str(path_val).strip()
    if not s or s.lower() == "nan":
        return None
    p = Path(s)
    if not p.is_absolute() and data_root is not None:
        p = (data_root / p).resolve()
    return p


def samples_from_metadata_df(
    df: "pd.DataFrame",
    *,
    data_root: Path | None = None,
    columns: dict[str, str] | None = None,
) -> list[Sample]:
    """
    Build a list of `Sample` objects from a metadata DataFrame.

    Expected shape
    --------------
    The DataFrame should carry at minimum a UID column. If it also carries
    `*_fullpath` columns (as produced by `data.metadata.build_master_metadata`
    and the file-audit CSV), those are used directly. Otherwise, `*_relpath`
    columns are resolved against `data_root`.

    Parameters
    ----------
    df
        Metadata DataFrame.
    data_root
        Absolute root directory for resolving relative paths. Required if
        the DataFrame only has `*_relpath` columns.
    columns
        Override the column-name mapping. Useful for datasets that diverge
        from the default schema — merge your overrides into a copy of
        `DEFAULT_COLUMNS` and pass it in.

    Returns
    -------
    list of `Sample`, in the order of rows in `df`.
    """
    cols = {**DEFAULT_COLUMNS, **(columns or {})}

    samples: list[Sample] = []
    for _, row in df.iterrows():
        uid = str(row[cols["uid"]])

        # Prefer `*_fullpath` columns when they exist.
        if cols["rgb_fullpath"] in df.columns:
            rgb = _resolve(row.get(cols["rgb_fullpath"]), None)
            dapi = _resolve(row.get(cols["dapi_fullpath"]), None)
            roi = _resolve(row.get(cols["roi_fullpath"]), None)
            gt = _resolve(row.get(cols["gt_fullpath"]), None)
        else:
            rgb = _resolve(row.get(cols["rgb_relpath"]), data_root)
            dapi = _resolve(row.get(cols["dapi_relpath"]), data_root)
            roi = _resolve(row.get(cols["roi_relpath"]), data_root)
            gt = _resolve(row.get(cols["gt_relpath"]), data_root)

        cell_line = row.get(cols["cell_line"])
        if cell_line is not None and isinstance(cell_line, float):
            import math
            if math.isnan(cell_line):
                cell_line = None
        cell_line = str(cell_line) if cell_line else None

        split_raw = row.get(cols["split"])
        split: Split | None = None
        if split_raw and str(split_raw).strip().lower() in {"train", "val", "test"}:
            split = str(split_raw).strip().lower()  # type: ignore[assignment]

        ecdna_gt_val = row.get(cols["ecdna_gt"])
        ecdna_gt: int | None
        try:
            ecdna_gt = int(ecdna_gt_val) if ecdna_gt_val is not None else None
        except (TypeError, ValueError):
            ecdna_gt = None

        extra: dict[str, Any] = {}
        for c in EXTRA_COLUMNS:
            if c in df.columns:
                extra[c] = row[c]

        samples.append(
            Sample(
                uid=uid,
                rgb_path=rgb,
                dapi_path=dapi,
                roi_mask_path=roi,
                gt_mask_path=gt,
                cell_line=cell_line,
                split=split,
                ecdna_gt=ecdna_gt,
                extra=extra,
            )
        )

    return samples


def samples_from_folders(
    uids: Iterable[str],
    *,
    rgb_dir: Path,
    dapi_dir: Path | None = None,
    roi_dir: Path | None = None,
    gt_dir: Path | None = None,
    dataset_type: Literal["counts_only", "localized_gt"] = "localized_gt",
    cell_line_inference: bool = True,
) -> list[Sample]:
    """
    Build `Sample` objects by discovering files in folders using UID matching.

    This is the legacy / rebuild flow: given a list of UIDs and a set of
    modality directories, use `data.ids` to locate the corresponding files.
    Prefer `samples_from_metadata_df` whenever the metadata CSV is available.
    """
    from ecdna_bench.data.ids import (
        find_corresponding_dapi_counts_extended,
        find_corresponding_file_localized,
        infer_cell_line,
    )

    samples: list[Sample] = []
    for uid in uids:
        if dataset_type == "localized_gt":
            rgb = find_corresponding_file_localized(uid, str(rgb_dir)) if rgb_dir else None
            dapi = find_corresponding_file_localized(uid, str(dapi_dir)) if dapi_dir else None
            roi = find_corresponding_file_localized(uid, str(roi_dir)) if roi_dir else None
            gt = find_corresponding_file_localized(uid, str(gt_dir)) if gt_dir else None
        else:
            # counts-only: RGB matched by whole-filename logic (UIDs already normalized)
            rgb = str(rgb_dir / f"{uid}.tif") if (rgb_dir and (rgb_dir / f"{uid}.tif").exists()) else None
            dapi = find_corresponding_dapi_counts_extended(uid, str(dapi_dir)) if dapi_dir else None
            roi = None
            gt = None

        cell_line = infer_cell_line(uid) if cell_line_inference else None

        samples.append(
            Sample(
                uid=uid,
                rgb_path=Path(rgb) if rgb else None,
                dapi_path=Path(dapi) if dapi else None,
                roi_mask_path=Path(roi) if roi else None,
                gt_mask_path=Path(gt) if gt else None,
                cell_line=cell_line,
            )
        )

    return samples


# ==============================================================================
# Filters
# ==============================================================================


def filter_by_split(samples: Iterable[Sample], split: Split) -> list[Sample]:
    """Return only samples with `sample.split == split`."""
    return [s for s in samples if s.split == split]


def filter_by_cell_line(
    samples: Iterable[Sample],
    cell_line: str | Iterable[str],
) -> list[Sample]:
    """
    Return samples whose cell line matches.

    `cell_line` may be a single canonical name or an iterable of canonical
    names. Matching is case-sensitive against the canonical spellings in
    `ecdna_bench.data.ids.KNOWN_CELL_LINES`.
    """
    if isinstance(cell_line, str):
        allowed = {cell_line}
    else:
        allowed = set(cell_line)
    return [s for s in samples if s.cell_line in allowed]


def filter_with_roi(samples: Iterable[Sample]) -> list[Sample]:
    """Return only samples that have a resolved ROI-mask path (benchmark subset)."""
    return [s for s in samples if s.has_roi_mask]


def filter_passing(
    samples: Iterable[Sample],
    *,
    require_count_mask_consistent: bool = True,
    require_file_audit_pass: bool = True,
) -> list[Sample]:
    """
    Keep samples that pass the QC flags stored in `Sample.extra`.

    Callers choose which flags to require; the defaults match the paper's
    training-data selection (both file audit and count-mask consistency).
    When a flag is missing from `extra` (older metadata shapes), it is
    treated as "pass" rather than "fail" — explicit flags should always
    be preferred.
    """
    out: list[Sample] = []
    for s in samples:
        if require_file_audit_pass and not bool(s.extra.get("file_audit_pass", True)):
            continue
        if require_count_mask_consistent and not bool(
            s.extra.get("count_mask_consistent", True)
        ):
            continue
        out.append(s)
    return out


def group_by_cell_line(samples: Iterable[Sample]) -> dict[str, list[Sample]]:
    """
    Group samples by cell-line label. Samples with cell_line=None are
    collected under the key `"UNKNOWN"` so callers can decide what to do.
    """
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        key = s.cell_line or "UNKNOWN"
        groups.setdefault(key, []).append(s)
    return groups


def iter_samples(samples: Iterable[Sample]) -> Iterator[Sample]:
    """
    Trivial pass-through generator. Provided as an explicit hook point so
    callers can later wrap sample iteration with progress bars or logging
    without touching call sites.
    """
    for s in samples:
        yield s


__all__ = [
    "Sample",
    "ImageSample",
    "Split",
    "DEFAULT_COLUMNS",
    "EXTRA_COLUMNS",
    "samples_from_metadata_df",
    "samples_from_folders",
    "filter_by_split",
    "filter_by_cell_line",
    "filter_with_roi",
    "filter_passing",
    "group_by_cell_line",
    "iter_samples",
]
