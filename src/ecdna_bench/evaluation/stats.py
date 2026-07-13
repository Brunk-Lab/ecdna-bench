"""
ecdna_bench.evaluation.stats — paired statistical tests used in the
benchmark-comparison panels.

The paper reports three statistical tools, all non-parametric:

  1. **Wilcoxon signed-rank** for head-to-head model comparisons on paired
     per-image metrics. Used in Figure 6 captions to flag statistically
     significant pairwise differences.
  2. **Friedman test** for joint significance across more than two models
     on the same set of paired per-image metrics. Produces the
     "models differ overall" signal that motivates the per-pair Wilcoxons.
  3. **Benjamini–Hochberg** FDR control to adjust the p-values from the
     pairwise Wilcoxons for multiple comparisons. We prefer BH (step-up
     procedure) to Bonferroni because the number of model pairs (15 for
     6 models) is modest but Bonferroni would still be unnecessarily
     conservative.

All three functions here are thin wrappers around ``scipy.stats`` (and, for
BH, a hand-rolled implementation because scipy's ``multipletests`` lives in
``statsmodels`` which we want to avoid pulling in as a dependency for this
stack). They return plain floats / tuples rather than scipy's typed Result
objects so that callers can write them straight to CSVs without unwrapping.

Design notes
------------
- These functions never mutate their inputs and never log.
- Ties: we rely on scipy's default tie handling for Wilcoxon, which is
  the "average" method (ranks tied observations by their mean rank). This
  is the most common convention and is what the rest of the bio-imaging
  literature uses.
- Zero differences: scipy's default behavior in recent versions is
  ``zero_method='wilcox'`` (discard zero differences). We set this
  explicitly for reproducibility across scipy versions.
- NaN handling: any pair with a NaN on either side is dropped before the
  test runs. The dropped-pair count is reported in the return tuple so
  callers can decide whether the remaining sample is large enough.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.stats import friedmanchisquare, wilcoxon


__all__ = [
    "wilcoxon_signed_rank",
    "friedman_test",
    "benjamini_hochberg",
]


# ==============================================================================
# Wilcoxon signed-rank — paired, two-model comparison.
# ==============================================================================


def wilcoxon_signed_rank(
    x: Sequence[float],
    y: Sequence[float],
    *,
    alternative: str = "two-sided",
) -> tuple[float, float, int]:
    """
    Wilcoxon signed-rank test on paired per-image metrics ``x`` vs ``y``.

    Pairs with NaN on either side are dropped silently; pairs with a zero
    difference are dropped by scipy under ``zero_method='wilcox'``. The
    effective sample size (pairs actually used by scipy) is returned as
    the third element of the result tuple.

    Parameters
    ----------
    x, y : sequences of float
        Paired observations, same length. ``x[i]`` and ``y[i]`` must come
        from the same image.
    alternative : {"two-sided", "greater", "less"}, default "two-sided"
        Null-hypothesis directionality. Passed through to scipy.

    Returns
    -------
    statistic : float
        Wilcoxon W statistic.
    pvalue : float
        Two-sided (or one-sided, per ``alternative``) p-value.
    n_used : int
        Number of pairs actually used by the test (after NaN and zero-diff
        filtering). If ``n_used < 5`` the test is underpowered and the
        p-value should be treated with caution; we do not raise in that
        case because some paper panels intentionally report "too few
        samples" cells in the grid.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    if x_arr.shape != y_arr.shape:
        raise ValueError(
            f"x and y must have the same shape; got {x_arr.shape} vs {y_arr.shape}"
        )

    mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    x_f = x_arr[mask]
    y_f = y_arr[mask]

    if x_f.size < 2:
        return float("nan"), float("nan"), int(x_f.size)

    diffs = x_f - y_f
    # If all nonzero differences have the same sign and there is variation
    # across pairs, scipy still returns a well-defined p-value; but if all
    # differences are exactly zero it errors out. Short-circuit that case.
    if np.allclose(diffs, 0.0):
        return 0.0, 1.0, int(x_f.size)

    stat, pval = wilcoxon(
        x_f,
        y_f,
        zero_method="wilcox",
        alternative=alternative,
    )

    # After scipy's zero filter, the effective n is the number of
    # non-zero pair differences.
    n_used = int(np.count_nonzero(diffs != 0))

    return float(stat), float(pval), n_used


# ==============================================================================
# Friedman — omnibus test across >= 3 models.
# ==============================================================================


def friedman_test(*arrays: Sequence[float]) -> tuple[float, float, int]:
    """
    Friedman chi-squared test for ``k`` paired repeated measurements
    (e.g., ``k`` models each evaluated on the same ``n`` images).

    Each argument is a 1-D sequence of per-image scores for one model;
    the i-th entry across all arrays must correspond to the same image.
    Any image with a NaN in any array is dropped from all arrays (complete
    observations only).

    Parameters
    ----------
    *arrays
        ``k`` sequences of float, all the same length ``n``. ``k >= 3``
        is required by the Friedman test.

    Returns
    -------
    statistic : float
        Friedman chi-squared statistic.
    pvalue : float
    n_used : int
        Number of images retained after NaN filtering.
    """
    if len(arrays) < 3:
        raise ValueError(
            f"Friedman test requires at least 3 groups, got {len(arrays)}"
        )

    stacked = np.asarray(arrays, dtype=float)
    if stacked.ndim != 2:
        raise ValueError("all input arrays must be 1-D and the same length")

    # Drop columns (images) with any NaN in any row (model).
    mask = np.all(np.isfinite(stacked), axis=0)
    stacked_clean = stacked[:, mask]
    n_used = int(stacked_clean.shape[1])

    if n_used < 2:
        return float("nan"), float("nan"), n_used

    stat, pval = friedmanchisquare(*stacked_clean)
    return float(stat), float(pval), n_used


# ==============================================================================
# Benjamini–Hochberg FDR correction.
# ==============================================================================


def benjamini_hochberg(
    pvalues: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Benjamini–Hochberg step-up procedure for FDR control.

    For a vector of raw p-values, returns
      (a) a boolean "rejected" array (True where the null is rejected at
          the given FDR level), and
      (b) a vector of BH-adjusted p-values (also known as q-values).

    The adjustment is the standard step-up:

        1. Sort p-values ascending: p_(1) <= p_(2) <= … <= p_(m).
        2. Adjusted p_(i) = min over k >= i of (m / k) * p_(k), then
           clipped to [0, 1].
        3. Reject all hypotheses with adjusted p <= alpha.

    NaN p-values are preserved as NaN in the adjusted output; they do
    not participate in the rejection decision, but they also do not
    reduce ``m`` (we use the total count in the denominator to be
    conservative — the assumption here is that NaN indicates "test
    could not be run" rather than "test was skipped for a reason").

    Parameters
    ----------
    pvalues : sequence of float, shape (m,)
    alpha : float, default 0.05
        Target FDR level.

    Returns
    -------
    rejected : np.ndarray of bool, shape (m,)
        True where the null is rejected. ``rejected[i]`` is False for
        any ``NaN`` input p-value.
    adjusted : np.ndarray of float, shape (m,)
        BH-adjusted p-values; NaN where the input was NaN.
    """
    raw = np.asarray(pvalues, dtype=float)
    if raw.ndim != 1:
        raise ValueError(f"pvalues must be 1-D, got shape {raw.shape}")

    m = raw.size
    adjusted = np.full(m, np.nan, dtype=float)

    finite_mask = np.isfinite(raw)
    finite_p = raw[finite_mask]
    k = finite_p.size

    if k == 0:
        return np.zeros(m, dtype=bool), adjusted

    # 1. sort ascending, keep original indices.
    order = np.argsort(finite_p, kind="mergesort")
    p_sorted = finite_p[order]

    # 2. raw adjusted = (m / rank) * p_sorted, where rank = 1..k.
    #    NOTE: we divide by the *total* number of tests (``m``), not by
    #    the number of non-NaN tests. This is the conservative choice;
    #    using ``k`` instead would make NaNs "free" which is usually not
    #    what you want.
    ranks = np.arange(1, k + 1, dtype=float)
    raw_adj = (m / ranks) * p_sorted

    # 3. enforce monotonicity (step-up): adjusted_(i) = min(raw_adj[i:k]).
    #    Computed as a reverse cumulative minimum.
    adj_sorted = np.minimum.accumulate(raw_adj[::-1])[::-1]
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)

    # 4. undo the sort to place q-values back in original order.
    inv = np.empty_like(order)
    inv[order] = np.arange(k)
    adj_unordered = adj_sorted[inv]

    # Place back into the full-size vector at the finite indices.
    adjusted_finite = np.full(k, np.nan, dtype=float)
    adjusted_finite[:] = adj_unordered
    adjusted[finite_mask] = adjusted_finite

    rejected = np.zeros(m, dtype=bool)
    rejected[finite_mask] = adjusted_finite <= float(alpha)

    return rejected, adjusted
