"""
ecdna_bench.data.qc — automated quality-control checks on the master metadata.

Three checks, each a pure function taking a DataFrame and returning a DataFrame:

1. ``run_file_audit``
     For every row, attempt to read the RGB / DAPI / ROI / GT files, record
     shapes, and verify the four modalities are spatially consistent. Adds
     columns ``rgb_exists``, ``*_read_ok``, ``shapes_match_all``,
     ``file_audit_pass``, etc. Supp Methods §3 / §5.

2. ``run_count_mask_consistency``
     Given the audited DataFrame, verify that ``ecDNA_gt == 0 iff gt_mask is
     empty``. Adds columns ``zero_count_empty_gt``, ``positive_count_empty_gt``,
     ``count_mask_consistent``. This is the strict filter used to select the
     ecCount training set (Supp §10.1). Supp Methods §5.

     Note: after a correct ``build_metadata`` run, ``ecDNA_gt`` is the GT-mask
     CC count (not the annotation-point count), so this check is about
     sign-agreement between the mask CC count and the mask's emptiness — which
     is always satisfied for well-formed benchmark images.

3. ``run_gt_morphology_audit``
     Decompose each GT mask into connected components, apply the canonical
     min_area=3 filter, and record area stats and diamond-merge discrepancy.
     Supp Methods §4 / §7. This is a pure reporting step — nothing downstream
     depends on its output — but it is the source for the Supplementary Figure
     on annotation-point / mask-CC discrepancy.

Every check is deterministic and side-effect-free on its inputs. Writing CSVs
is the caller's responsibility — typically handled by the CLI wrapper
``ecdna_bench.cli.run_qc``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ecdna_bench.utils.log_utils import get_logger

if TYPE_CHECKING:
    import pandas as pd


logger = get_logger(__name__)


# ==============================================================================
# File audit — replicates stage1_step2 behaviour
# ==============================================================================


# Columns the master metadata must provide before a file audit can run.
FILE_AUDIT_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "unique_id",
        "cell_line",
        "split",
        "rgb_relpath",
        "dapi_relpath",
        "roi_mask_relpath",
        "gt_mask_relpath",
        "ecDNA_gt",
    }
)


def _resolve_relpath(data_root: Path, relpath: object) -> Path | None:
    if relpath is None:
        return None
    s = "" if isinstance(relpath, float) and np.isnan(relpath) else str(relpath).strip()
    if not s or s.lower() == "nan":
        return None
    p = Path(s)
    return p if p.is_absolute() else data_root / p


def _read_rgb(path: Path) -> tuple[bool, np.ndarray | None, str]:
    import cv2
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            return False, None, "cv2.imread returned None"
        return True, img, ""
    except Exception as e:  # pragma: no cover
        return False, None, repr(e)


def _read_gray(path: Path) -> tuple[bool, np.ndarray | None, str]:
    import cv2
    try:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return False, None, "cv2.imread returned None"
        return True, img, ""
    except Exception as e:  # pragma: no cover
        return False, None, repr(e)


def _hw(arr: np.ndarray | None) -> tuple[int | None, int | None]:
    if arr is None:
        return None, None
    return int(arr.shape[0]), int(arr.shape[1])


def _shapes_match(*arrays: np.ndarray | None) -> bool:
    shapes = [a.shape[:2] for a in arrays if a is not None]
    if len(shapes) <= 1:
        return False
    first = shapes[0]
    return all(s == first for s in shapes[1:])


def _mask_rgb_with_roi(rgb: np.ndarray, roi: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"RGB must be HxWx3, got {rgb.shape}")
    if roi.ndim != 2:
        raise ValueError(f"ROI must be 2D, got {roi.shape}")
    if rgb.shape[:2] != roi.shape:
        raise ValueError(f"RGB/ROI shape mismatch: {rgb.shape[:2]} vs {roi.shape}")
    keep = roi > 0
    out  = np.zeros_like(rgb)
    out[keep] = rgb[keep]
    return out


def run_file_audit(
    df: "pd.DataFrame",
    data_root: Path | str,
    *,
    progress: bool = True,
) -> "pd.DataFrame":
    """
    Perform the Stage 1 Step 2 file audit on a master metadata DataFrame.

    For every row:
      - resolve the four modality paths against ``data_root``
      - verify existence and readability
      - record shapes and nonzero-pixel counts
      - verify spatial shape consistency across modalities
      - verify that masking RGB with ROI succeeds

    Returns a new DataFrame that is ``df`` plus audit columns. The input is
    not mutated.
    """
    _require_cols(df, FILE_AUDIT_REQUIRED_COLS, "master metadata DataFrame")
    data_root = Path(data_root)

    rows = df.to_dict(orient="records")
    if progress:
        try:
            from tqdm import tqdm
            rows_iter = tqdm(rows, total=len(rows), desc="File audit")
        except ImportError:
            rows_iter = rows
    else:
        rows_iter = rows

    audited: list[dict] = []
    for row in rows_iter:
        rec = dict(row)

        rgb_path  = _resolve_relpath(data_root, row.get("rgb_relpath"))
        dapi_path = _resolve_relpath(data_root, row.get("dapi_relpath"))
        roi_path  = _resolve_relpath(data_root, row.get("roi_mask_relpath"))
        gt_path   = _resolve_relpath(data_root, row.get("gt_mask_relpath"))

        rec["rgb_fullpath"]  = str(rgb_path)  if rgb_path  else ""
        rec["dapi_fullpath"] = str(dapi_path) if dapi_path else ""
        rec["roi_fullpath"]  = str(roi_path)  if roi_path  else ""
        rec["gt_fullpath"]   = str(gt_path)   if gt_path   else ""

        rec["rgb_exists"]  = bool(rgb_path  and rgb_path.exists())
        rec["dapi_exists"] = bool(dapi_path and dapi_path.exists())
        rec["roi_exists"]  = bool(roi_path  and roi_path.exists())
        rec["gt_exists"]   = bool(gt_path   and gt_path.exists())

        rgb_ok,  rgb,  rgb_err  = (False, None, "file missing")
        dapi_ok, dapi, dapi_err = (False, None, "file missing")
        roi_ok,  roi,  roi_err  = (False, None, "file missing")
        gt_ok,   gt,   gt_err   = (False, None, "file missing")

        if rec["rgb_exists"]:
            rgb_ok,  rgb,  rgb_err  = _read_rgb(rgb_path)   # type: ignore[arg-type]
        if rec["dapi_exists"]:
            dapi_ok, dapi, dapi_err = _read_gray(dapi_path) # type: ignore[arg-type]
        if rec["roi_exists"]:
            roi_ok,  roi,  roi_err  = _read_gray(roi_path)  # type: ignore[arg-type]
        if rec["gt_exists"]:
            gt_ok,   gt,   gt_err   = _read_gray(gt_path)   # type: ignore[arg-type]

        rec["rgb_read_ok"]    = rgb_ok
        rec["dapi_read_ok"]   = dapi_ok
        rec["roi_read_ok"]    = roi_ok
        rec["gt_read_ok"]     = gt_ok
        rec["rgb_read_error"]  = rgb_err
        rec["dapi_read_error"] = dapi_err
        rec["roi_read_error"]  = roi_err
        rec["gt_read_error"]   = gt_err

        rec["rgb_h"],  rec["rgb_w"]  = _hw(rgb)
        rec["dapi_h"], rec["dapi_w"] = _hw(dapi)
        rec["roi_h"],  rec["roi_w"]  = _hw(roi)
        rec["gt_h"],   rec["gt_w"]   = _hw(gt)

        rec["shapes_match_all"] = bool(
            rgb_ok and dapi_ok and roi_ok and gt_ok
            and _shapes_match(rgb, dapi, roi, gt)
        )

        rec["roi_nonzero_pixels"] = int(np.count_nonzero(roi)) if roi is not None else None
        rec["gt_nonzero_pixels"]  = int(np.count_nonzero(gt))  if gt  is not None else None
        rec["roi_nonempty"] = bool(rec["roi_nonzero_pixels"] and rec["roi_nonzero_pixels"] > 0)
        rec["gt_nonempty"]  = bool(rec["gt_nonzero_pixels"]  and rec["gt_nonzero_pixels"]  > 0)

        masked_rgb_ok  = False
        masked_rgb_err = ""
        masked_nonzero = None
        if (
            rgb_ok and roi_ok
            and rgb is not None and roi is not None
            and rgb.shape[:2] == roi.shape
        ):
            try:
                masked         = _mask_rgb_with_roi(rgb, roi)
                masked_rgb_ok  = True
                masked_nonzero = int(np.count_nonzero(masked))
            except Exception as e:  # pragma: no cover
                masked_rgb_err = repr(e)
        else:
            masked_rgb_err = "RGB/ROI unavailable or shape mismatch"

        rec["masked_rgb_ok"]             = masked_rgb_ok
        rec["masked_rgb_error"]          = masked_rgb_err
        rec["masked_rgb_nonzero_pixels"] = masked_nonzero

        rec["file_audit_pass"] = all(
            [
                rec["rgb_exists"],
                rec["dapi_exists"],
                rec["roi_exists"],
                rec["gt_exists"],
                rec["rgb_read_ok"],
                rec["dapi_read_ok"],
                rec["roi_read_ok"],
                rec["gt_read_ok"],
                rec["shapes_match_all"],
                rec["roi_nonempty"],
                rec["masked_rgb_ok"],
            ]
        )

        audited.append(rec)

    import pandas as pd
    return pd.DataFrame(audited)


# ==============================================================================
# Count-mask consistency — replicates stage1_step3
# ==============================================================================


COUNT_MASK_REQUIRED_COLS: frozenset[str] = frozenset(
    {
        "unique_id",
        "cell_line",
        "split",
        "ecDNA_gt",
        "gt_nonzero_pixels",
        "gt_nonempty",
        "file_audit_pass",
    }
)


def run_count_mask_consistency(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Verify that ``ecDNA_gt == 0 iff gt_mask is empty``.

    Consumes the output of ``run_file_audit``. Adds columns:
      - zero_count, positive_count
      - zero_count_empty_gt, zero_count_nonempty_gt
      - positive_count_empty_gt, positive_count_nonempty_gt
      - count_mask_consistent

    The ecCount training set is filtered by ``count_mask_consistent=True``
    (Supp §10.1). Callers should retain inconsistent rows in the full dataset
    release but exclude them from training.

    After a correct ``build_metadata`` run, ``ecDNA_gt`` is the GT-mask CC
    count (8-conn, min_area=3 px), so ``count_mask_consistent`` is True for
    all well-formed images: any image with a non-empty GT mask has at least
    one CC and therefore ``ecDNA_gt >= 1``.
    """
    import pandas as pd
    _require_cols(df, COUNT_MASK_REQUIRED_COLS, "audit DataFrame")

    out = df.copy()
    out["ecDNA_gt"] = pd.to_numeric(out["ecDNA_gt"], errors="coerce")
    out["gt_nonzero_pixels"] = (
        pd.to_numeric(out["gt_nonzero_pixels"], errors="coerce").fillna(0).astype(int)
    )
    out["gt_nonempty"]    = out["gt_nonempty"].fillna(False).astype(bool)
    out["file_audit_pass"] = out["file_audit_pass"].fillna(False).astype(bool)

    out["zero_count"]    = out["ecDNA_gt"] == 0
    out["positive_count"] = out["ecDNA_gt"] > 0

    out["zero_count_empty_gt"]       = out["zero_count"]    & (~out["gt_nonempty"])
    out["zero_count_nonempty_gt"]    = out["zero_count"]    & out["gt_nonempty"]
    out["positive_count_empty_gt"]   = out["positive_count"] & (~out["gt_nonempty"])
    out["positive_count_nonempty_gt"] = out["positive_count"] & out["gt_nonempty"]

    out["count_mask_consistent"] = (
        out["zero_count_empty_gt"] | out["positive_count_nonempty_gt"]
    )
    return out


# ==============================================================================
# GT morphology audit — replicates stage2_step1
# ==============================================================================


def run_gt_morphology_audit(
    df: "pd.DataFrame",
    *,
    connectivity: int = 8,
    min_area: int = 3,
    progress: bool = True,
) -> "pd.DataFrame":
    """
    For every GT mask, count connected components and record area stats.

    Consumes a DataFrame with at least ``gt_fullpath`` and ``ecDNA_gt``
    columns (the file-audit output satisfies this).

    Adds columns:
      gt_n_components
          Number of 8-connectivity CCs with area >= ``min_area`` (default 3 px).
          Matches the canonical evaluation definition (PROJECT_RULES.md §2).
          After a correct ``build_metadata`` run, this equals ``ecDNA_gt`` for
          every passing image.
      gt_min_area, gt_max_area, gt_mean_area
          Component area statistics computed over ALL components (no min filter),
          so the distribution of tiny CCs is visible in the supplementary.
      gt_components_match_ecDNA_gt
          True iff ``gt_n_components == ecDNA_gt``. After a correct
          ``build_metadata`` run this is always True and serves as a
          consistency guard. Images where it is False indicate a rebuild with
          different parameters.
      coord_minus_gt_cc_min3
          ``coord_count_npy - ecDNA_gt`` (when both columns present). Positive
          values quantify the diamond-merge effect. This is the primary source
          for the Supplementary Figure on annotation-point / mask-CC discrepancy
          (Supp Methods §4 / §7 of PROJECT_RULES.md).
      coord_gt_cc_match
          True iff ``coord_count_npy == ecDNA_gt``.
    """
    import cv2
    import pandas as pd

    if "gt_fullpath" not in df.columns:
        raise ValueError("Expected 'gt_fullpath' column; run run_file_audit first.")

    rows = df.to_dict(orient="records")
    if progress:
        try:
            from tqdm import tqdm
            rows_iter = tqdm(rows, total=len(rows), desc="GT morphology audit")
        except ImportError:
            rows_iter = rows
    else:
        rows_iter = rows

    out: list[dict] = []
    for row in rows_iter:
        rec = dict(row)
        path = row.get("gt_fullpath")
        n_comp     = 0
        min_a_val  = None
        max_a_val  = None
        mean_a_val = None

        if path and Path(path).exists():
            gt = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if gt is not None:
                binary = (gt > 0).astype(np.uint8)
                n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
                    binary, connectivity=connectivity
                )
                # index 0 is background; 1: are foreground components
                if n_labels > 1:
                    areas = stats[1:, cv2.CC_STAT_AREA]
                    # Area stats over ALL components (no min filter) — for distribution plots.
                    min_a_val  = int(areas.min())
                    max_a_val  = int(areas.max())
                    mean_a_val = float(areas.mean())
                    # Canonical count: only components meeting the min_area threshold.
                    # This matches the evaluation framework and ecDNA_gt definition.
                    n_comp = int((areas >= min_area).sum())

        rec["gt_n_components"] = n_comp
        rec["gt_min_area"]     = min_a_val
        rec["gt_max_area"]     = max_a_val
        rec["gt_mean_area"]    = mean_a_val

        # Consistency guard: after a correct build_metadata run,
        # gt_n_components == ecDNA_gt by construction (both use the same
        # mask-CC definition). If this is False, something diverged.
        ecdna_gt_val = row.get("ecDNA_gt")
        if ecdna_gt_val is not None:
            try:
                rec["gt_components_match_ecDNA_gt"] = bool(n_comp == int(ecdna_gt_val))
            except (TypeError, ValueError):
                rec["gt_components_match_ecDNA_gt"] = False
        else:
            rec["gt_components_match_ecDNA_gt"] = False

        # Diamond-merge quantification: coord_count_npy - ecDNA_gt.
        # Source for the Supplementary Figure on annotation uncertainty.
        coord_val = row.get("coord_count_npy")
        if coord_val is not None and ecdna_gt_val is not None:
            try:
                diff = int(coord_val) - int(ecdna_gt_val)
                rec["coord_minus_gt_cc_min3"] = diff
                rec["coord_gt_cc_match"]      = (diff == 0)
            except (TypeError, ValueError):
                rec["coord_minus_gt_cc_min3"] = None
                rec["coord_gt_cc_match"]      = False
        else:
            rec["coord_minus_gt_cc_min3"] = None
            rec["coord_gt_cc_match"]      = False

        out.append(rec)

    return pd.DataFrame(out)


# ==============================================================================
# Convenience wrapper
# ==============================================================================


def run_all_qc(
    master_df: "pd.DataFrame",
    data_root: Path | str,
    *,
    include_gt_morphology: bool = False,
    progress: bool = True,
) -> "pd.DataFrame":
    """
    Convenience: run the file audit + count/mask consistency in sequence.

    Returns a single DataFrame with every column from every stage. If
    ``include_gt_morphology=True``, also runs the (slower) per-image
    connected-component audit and diamond-merge statistics.

    For provenance logging:
      - file audit pass rate : ``sum(file_audit_pass) / N``
      - count/mask pass rate : ``sum(count_mask_consistent) / N``
      - GT component mismatch: ``sum(~gt_components_match_ecDNA_gt)`` (if run)
    """
    df1 = run_file_audit(master_df, data_root, progress=progress)
    df2 = run_count_mask_consistency(df1)
    if include_gt_morphology:
        df2 = run_gt_morphology_audit(df2, progress=progress)

    n = len(df2)
    logger.info(
        "QC complete: file_audit_pass=%d/%d  count_mask_consistent=%d/%d",
        int(df2["file_audit_pass"].sum()),
        n,
        int(df2["count_mask_consistent"].sum()),
        n,
    )
    return df2


# ==============================================================================
# Summaries
# ==============================================================================


def summarize_file_audit(df: "pd.DataFrame") -> str:
    """Produce the text summary that stage1_step2 used to write."""
    total = len(df)
    lines: list[str] = ["FILE AUDIT SUMMARY", "=" * 80, ""]
    for col in (
        "rgb_exists", "dapi_exists", "roi_exists", "gt_exists",
        "rgb_read_ok", "dapi_read_ok", "roi_read_ok", "gt_read_ok",
        "shapes_match_all", "masked_rgb_ok",
        "roi_nonempty", "gt_nonempty", "file_audit_pass",
    ):
        if col in df.columns:
            n = int(df[col].fillna(False).astype(bool).sum())
            lines.append(f"{col:<20}: {n} / {total}")
    return "\n".join(lines)


def summarize_count_mask(df: "pd.DataFrame") -> str:
    """Produce the text summary for count/mask consistency results."""
    total = len(df)
    lines: list[str] = ["COUNT-vs-MASK CONSISTENCY SUMMARY", "=" * 80, ""]
    for col in (
        "zero_count_empty_gt",
        "zero_count_nonempty_gt",
        "positive_count_empty_gt",
        "positive_count_nonempty_gt",
        "count_mask_consistent",
    ):
        if col in df.columns:
            n = int(df[col].astype(bool).sum())
            lines.append(f"{col:<30}: {n} / {total}")

    # Report diamond-merge statistics when the morphology audit was run.
    if "coord_gt_cc_match" in df.columns:
        lines.append("")
        lines.append("DIAMOND-MERGE STATISTICS (coord_count_npy vs ecDNA_gt)")
        lines.append("-" * 80)
        n_match = int(df["coord_gt_cc_match"].astype(bool).sum())
        lines.append(f"coord_count_npy == ecDNA_gt : {n_match} / {total}")
        lines.append(f"coord_count_npy >  ecDNA_gt : {total - n_match} / {total}")
        if "coord_minus_gt_cc_min3" in df.columns:
            import pandas as _pd
            diffs = _pd.to_numeric(df["coord_minus_gt_cc_min3"], errors="coerce").dropna()
            if len(diffs) > 0:
                lines.append(f"  median diff (coord - CC): {diffs.median():.1f}")
                lines.append(f"  mean diff   (coord - CC): {diffs.mean():.1f}")
                lines.append(f"  max diff    (coord - CC): {diffs.max():.0f}")
    return "\n".join(lines)


# ==============================================================================
# Internal helper
# ==============================================================================


def _require_cols(df: "pd.DataFrame", required: frozenset[str], source: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


# ==============================================================================
# CLI compatibility wrapper
# ==============================================================================

def run_qc(
    *,
    metadata_csv,
    counts_csv,
    data_root,
    output_csv,
    include_gt_morphology: bool = False,
    progress: bool = True,
) -> "pd.DataFrame":
    """
    Filesystem-facing wrapper expected by ``ecdna_bench.cli.run_qc``.

    Reads ``metadata.csv`` and ``counts_master.csv``, verifies that
    ``ecDNA_gt`` agrees between them, runs the file audit and count-mask
    consistency checks, then writes the combined QC CSV to ``output_csv``.

    ``counts_master.csv`` must carry an ``ecDNA_gt`` column whose values are
    GT-mask connected-component counts (8-connectivity, min_area=3 px) —
    i.e. the output of ``build_metadata`` in this package.  An optional
    ``coord_count_npy`` column (annotation-point counts from NPY/NPZ files)
    is forwarded into the QC output for the morphology-audit step.
    """
    import pandas as pd

    metadata_csv = Path(metadata_csv)
    counts_csv   = Path(counts_csv)
    output_csv   = Path(output_csv)

    if data_root is None:
        raise ValueError("data_root is required for QC because relative paths must be resolved.")
    data_root = Path(data_root)

    master_df = pd.read_csv(metadata_csv)
    counts_df = pd.read_csv(counts_csv)

    _require_cols(counts_df, frozenset({"unique_id", "ecDNA_gt"}), "counts CSV")

    master_df["unique_id"] = master_df["unique_id"].astype(str)
    counts_df["unique_id"] = counts_df["unique_id"].astype(str)

    # Columns to bring across from counts_df (beyond what is already in master_df).
    extra_count_cols = ["ecDNA_gt"]
    if "coord_count_npy" in counts_df.columns:
        extra_count_cols.append("coord_count_npy")

    if "ecDNA_gt" not in master_df.columns:
        # metadata.csv doesn't yet carry ecDNA_gt — merge it in.
        master_df = master_df.merge(
            counts_df[["unique_id"] + extra_count_cols],
            on="unique_id",
            how="left",
            validate="one_to_one",
        )
    else:
        # Verify that metadata.csv and counts_master.csv agree on ecDNA_gt.
        # Both must use the same canonical count (GT mask CCs, 8-conn, min_area=3).
        check = master_df[["unique_id", "ecDNA_gt"]].merge(
            counts_df[["unique_id", "ecDNA_gt"]].rename(
                columns={"ecDNA_gt": "ecDNA_gt_counts"}
            ),
            on="unique_id",
            how="left",
            validate="one_to_one",
        )
        left  = pd.to_numeric(check["ecDNA_gt"],        errors="coerce")
        right = pd.to_numeric(check["ecDNA_gt_counts"], errors="coerce")
        mismatch = (
            left.notna() & right.notna()
            & (left.astype(float) != right.astype(float))
        )
        if mismatch.any():
            examples = check.loc[
                mismatch, ["unique_id", "ecDNA_gt", "ecDNA_gt_counts"]
            ].head(10)
            raise ValueError(
                "metadata.csv and counts_master.csv disagree on ecDNA_gt. "
                "Both must use the canonical count (GT mask CCs, 8-conn, min_area=3 px). "
                "Re-run build_metadata to regenerate both files from scratch.\n"
                f"First mismatches:\n{examples.to_string(index=False)}"
            )
        # If coord_count_npy is in counts but not yet in master, merge it across.
        if "coord_count_npy" in counts_df.columns and "coord_count_npy" not in master_df.columns:
            master_df = master_df.merge(
                counts_df[["unique_id", "coord_count_npy"]],
                on="unique_id",
                how="left",
                validate="one_to_one",
            )

    result_df = run_all_qc(
        master_df,
        data_root,
        include_gt_morphology=include_gt_morphology,
        progress=progress,
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    output_csv.with_suffix(".file_audit_summary.txt").write_text(
        summarize_file_audit(result_df) + "\n"
    )
    output_csv.with_suffix(".count_mask_summary.txt").write_text(
        summarize_count_mask(result_df) + "\n"
    )

    return result_df


__all__ = [
    "FILE_AUDIT_REQUIRED_COLS",
    "COUNT_MASK_REQUIRED_COLS",
    "run_file_audit",
    "run_count_mask_consistency",
    "run_gt_morphology_audit",
    "run_all_qc",
    "run_qc",
    "summarize_file_audit",
    "summarize_count_mask",
]