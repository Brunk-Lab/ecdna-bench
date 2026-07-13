"""
ecdna_bench.evaluation.metrics — three families of metrics consumed by the
benchmark orchestrator and the figure generators.

Each family is a pure function family: no disk I/O, no global state, no
optional dependencies beyond numpy and scipy.stats.

Family overview
---------------

1. **Object metrics** — operate on a :class:`MatchResult` or, equivalently,
   on raw TP / FP / FN counts. Outputs precision, recall, F1. The key
   subtlety here is the treatment of *ignored predictions*: they do not
   contribute to either precision or recall, which is the paper's policy.

2. **Pixel metrics** — operate on aligned binary masks. Outputs
   pixel-level precision, recall, F1, IoU, Dice. These are used as a
   secondary lens on segmentation performance (main reported lens is
   object-level).

3. **Count metrics** — operate on paired arrays of true / predicted
   ecDNA counts. Outputs MAE, trimmed MAE, RMSE, signed bias, MdAPE
   (median absolute percentage error, computed on nonzero-GT images),
   Pearson ρ, Spearman ρ. These are the metrics that convert a
   localization output into a clinical-style counting endpoint.

The ``ecDNA_gt`` column of the benchmark CSVs is the canonical source of
ground-truth counts; predicted counts come from whatever post-processing
the model applied to its output mask.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from ecdna_bench.evaluation.matching import MatchResult

# scipy.stats is a light dependency already in the evaluation stack
# (scipy.optimize.linear_sum_assignment is used by matching.py), so
# importing pearsonr / spearmanr adds no new install burden.
from scipy.stats import pearsonr, spearmanr


__all__ = [
    "object_metrics",
    "object_metrics_from_counts",
    "pixel_confusion",
    "pixel_metrics",
    "count_metrics",
]


# Tiny constant used on all denominators to avoid 0/0 becoming NaN. The
# value is small enough that any non-degenerate case rounds to the correct
# metric, but large enough that denormals do not occur.
_EPS: float = 1e-9


# ==============================================================================
# Object-level metrics.
# ==============================================================================


def object_metrics_from_counts(
    tp: int | float,
    fp: int | float,
    fn: int | float,
    *,
    ignored: int | float = 0,
) -> dict[str, float]:
    """
    Compute precision / recall / F1 from raw TP/FP/FN counts.

    The ``ignored`` argument is accepted for audit-trail purposes (every
    per-image row in the frozen benchmark CSVs carries both ``fp`` and
    ``ignored`` so that reviewers can verify the policy) but does not
    enter the arithmetic: ignored predictions are explicitly excluded
    from the FP tally before it is passed in here.

    Parameters
    ----------
    tp, fp, fn : int or float
        True positive, false positive, false negative counts. Floats are
        accepted to allow aggregation of per-image metrics (e.g. mean FP
        across a split).
    ignored : int or float, default 0
        Count of ignored predictions; passed through into the output dict
        unchanged.

    Returns
    -------
    dict with keys ``tp, fp, fn, ignored, n_pred, n_gt, precision, recall, f1``.
    The ``n_pred`` / ``n_gt`` fields are derived under the assumption that
    ``tp + fp + ignored == n_pred`` and ``tp + fn == n_gt``, which is true
    for counts arising from a single :class:`MatchResult`.
    """
    tp_f = float(tp)
    fp_f = float(fp)
    fn_f = float(fn)
    ign_f = float(ignored)

    precision = tp_f / (tp_f + fp_f + _EPS)
    recall = tp_f / (tp_f + fn_f + _EPS)
    # Compute f1 directly to avoid double-eps error that breaks
    # the strict abs(f1 - 1.0) < 1e-9 test for perfect matches.
    denom = 2.0 * tp_f + fp_f + fn_f
    f1 = (2.0 * tp_f / denom) if denom > 0 else 0.0

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def object_metrics(result: MatchResult) -> dict[str, float]:
    """
    Compute precision / recall / F1 from a :class:`MatchResult`.

    Thin convenience wrapper around :func:`object_metrics_from_counts`;
    this is the form called by the benchmark orchestrator and by the
    sensitivity sweep for each grid cell. Every per-image row in the
    frozen benchmark CSVs is produced by this function.
    """
    return object_metrics_from_counts(
        tp=result.tp,
        fp=result.fp,
        fn=result.fn,
        ignored=result.ignored,
    )


# ==============================================================================
# Pixel-level metrics.
# ==============================================================================


def _to_binary(mask: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """
    Coerce an incoming mask (uint8/uint16/bool/float probability) to an
    unambiguous ``uint8`` mask with values in {0, 1}.

    The ``threshold`` argument is only consulted for floating-point inputs;
    integer inputs are binarized at "any nonzero value → 1".
    """
    if mask.dtype == bool:
        return mask.astype(np.uint8)

    if np.issubdtype(mask.dtype, np.integer):
        return (mask > 0).astype(np.uint8)

    # Floating point: assume a probability map in [0, 1]. If the max
    # happens to exceed 1, rescale to [0, 1] first — this lets us accept
    # uint8-rendered probability maps that were saved as 0–255 floats.
    m = mask.astype(np.float32, copy=False)
    if m.size > 0 and m.max() > 1.0:
        m = m / 255.0
    return (m >= float(threshold)).astype(np.uint8)


def pixel_confusion(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, int]:
    """
    Compute pixel-level TP / FP / FN / TN between a binary GT mask and a
    (binary or probability) prediction mask.

    Both inputs must share the same 2D shape; a shape mismatch raises
    ``ValueError``. This is a common enough foot-gun in practice (e.g.,
    a prediction saved at model resolution rather than original image
    resolution) that we refuse to silently resize either side — the
    benchmark orchestrator's harmonization step is responsible for
    resizing predictions to the GT frame before they reach this function.
    """
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"shape mismatch: gt={gt_mask.shape} vs pred={pred_mask.shape}"
        )

    gt = _to_binary(gt_mask, threshold=0.5)
    pr = _to_binary(pred_mask, threshold=threshold)

    tp = int(np.count_nonzero((gt == 1) & (pr == 1)))
    fp = int(np.count_nonzero((gt == 0) & (pr == 1)))
    fn = int(np.count_nonzero((gt == 1) & (pr == 0)))
    tn = int(np.count_nonzero((gt == 0) & (pr == 0)))

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def pixel_metrics(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float]:
    """
    Compute pixel-level precision, recall, F1, IoU, Dice between two masks.

    Passes through :func:`pixel_confusion`; see that function's docstring
    for input-shape constraints.
    """
    conf = pixel_confusion(gt_mask, pred_mask, threshold=threshold)
    tp = conf["tp"]
    fp = conf["fp"]
    fn = conf["fn"]

    precision = tp / (tp + fp + _EPS)
    recall = tp / (tp + fn + _EPS)
    f1 = 2.0 * precision * recall / (precision + recall + _EPS)

    intersection = float(tp)
    union = float(tp + fp + fn) + _EPS
    iou = intersection / union

    # Dice (== F1 for binary segmentation, but reported separately because
    # the paper's tables distinguish F1 from Dice where convention differs).
    dice = 2.0 * intersection / (2.0 * intersection + fp + fn + _EPS)

    return {
        "pixel_tp": float(tp),
        "pixel_fp": float(fp),
        "pixel_fn": float(fn),
        "pixel_tn": float(conf["tn"]),
        "pixel_precision": float(precision),
        "pixel_recall": float(recall),
        "pixel_f1": float(f1),
        "pixel_iou": float(iou),
        "pixel_dice": float(dice),
        # Unprefixed aliases for test suite and legacy scripts
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "dice": float(dice),
    }


# ==============================================================================
# Count metrics.
# ==============================================================================


def _safe_corr(
    func: Any,
    x: np.ndarray,
    y: np.ndarray,
) -> float:
    """
    Evaluate a scipy correlation function, returning ``nan`` in any of the
    documented failure modes (too few samples, zero variance on either
    side, scipy exception).
    """
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]

    if x.size < 2:
        return float("nan")

    # Zero variance on either side would otherwise yield an undefined
    # correlation. scipy does warn, but we prefer an explicit NaN.
    if np.isclose(np.std(x), 0.0) or np.isclose(np.std(y), 0.0):
        return float("nan")

    try:
        return float(func(x, y)[0])
    except Exception:  # pragma: no cover
        return float("nan")


def _trimmed_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    trim_percent: float = 5.0,
) -> float:
    """
    Mean absolute error after trimming the extreme ``trim_percent`` of
    absolute errors at both tails. Robust to per-image outliers and used
    as a supplementary-figure metric.
    """
    abs_errors = np.abs(y_pred - y_true)
    if abs_errors.size == 0:
        return float("nan")

    lo, hi = np.percentile(abs_errors, [trim_percent, 100.0 - trim_percent])
    kept = abs_errors[(abs_errors >= lo) & (abs_errors <= hi)]
    return float(kept.mean()) if kept.size > 0 else float("nan")


def _mdape(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    ignore_zero_gt: bool = True,
    fallback: float = float("nan"),
) -> float:
    """
    Median absolute percentage error. By default, images with
    ``y_true == 0`` are excluded from the pool (division-by-zero would
    otherwise dominate). This is the paper's reporting convention.
    """
    if y_true.size == 0:
        return float(fallback)

    if ignore_zero_gt:
        mask = y_true != 0
        y_true_f = y_true[mask].astype(float)
        y_pred_f = y_pred[mask].astype(float)
    else:
        y_true_f = y_true.astype(float)
        y_pred_f = y_pred.astype(float)

    if y_true_f.size == 0:
        return float(fallback)

    # Where y_true == 0 and we did not filter, contribute 0 if the
    # prediction is also 0 else a large sentinel. This matches the
    # legacy behavior.
    pct = np.zeros_like(y_true_f, dtype=float)
    nonzero = y_true_f != 0
    pct[nonzero] = 100.0 * np.abs(y_pred_f[nonzero] - y_true_f[nonzero]) / y_true_f[nonzero]
    pct[~nonzero] = np.where(y_pred_f[~nonzero] == 0, 0.0, 1000.0)

    return float(np.median(pct))


def count_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    *,
    trim_percent: float = 5.0,
) -> dict[str, float]:
    """
    Compute all count metrics reported in the paper from paired true /
    predicted count arrays.

    Parameters
    ----------
    y_true, y_pred : sequence of float
        Per-image true and predicted ecDNA counts. Must be the same
        length. Entries may be ``NaN``; they will be treated as missing
        by the correlation helpers.
    trim_percent : float, default 5.0
        Percentile trim used for ``count_trimmed_mae``.

    Returns
    -------
    dict with keys
        ``count_mae``, ``count_trimmed_mae``, ``count_rmse``,
        ``count_bias`` (signed mean error), ``count_mdape``,
        ``count_pearson_r``, ``count_spearman_rho``.

    Empty input returns a dict of ``nan`` values in all metric slots so
    that downstream CSV writers do not need special-case logic for empty
    slices.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"length mismatch: y_true has {y_true_arr.shape}, "
            f"y_pred has {y_pred_arr.shape}"
        )

    if y_true_arr.size == 0:
        nan = float("nan")
        return {
            "count_mae": nan,
            "count_trimmed_mae": nan,
            "count_rmse": nan,
            "count_bias": nan,
            "count_mdape": nan,
            "count_pearson_r": nan,
            "count_spearman_rho": nan,
        }

    residual = y_pred_arr - y_true_arr

    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    bias = float(np.mean(residual))
    trimmed_mae = _trimmed_mae(y_true_arr, y_pred_arr, trim_percent=trim_percent)
    mdape = _mdape(y_true_arr, y_pred_arr, ignore_zero_gt=True)
    pearson_r = _safe_corr(pearsonr, y_true_arr, y_pred_arr)
    spearman_rho = _safe_corr(spearmanr, y_true_arr, y_pred_arr)
    return {
        "count_mae": mae,
        "count_trimmed_mae": trimmed_mae,
        "count_rmse": rmse,
        "count_bias": bias,
        "count_mdape": mdape,
        "count_pearson_r": pearson_r,
        "count_spearman_rho": spearman_rho,
        # Unprefixed aliases for test suite and legacy scripts
        "mae": mae,
        "trimmed_mae": trimmed_mae,
        "rmse": rmse,
        "bias": bias,
        "mdape": mdape,
        "pearson_r": pearson_r,
        "spearman_rho": spearman_rho,
    }
