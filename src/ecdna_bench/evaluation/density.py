"""
ecdna_bench.evaluation.density — density-bin stratification.

The paper reports every headline metric both globally and split by five
count bins chosen to separate clinically-relevant regimes:

    [0, 10)   "0-9"      — essentially ecDNA-negative cells
    [10, 50)  "10-49"    — low burden
    [50, 150) "50-149"   — intermediate burden
    [150, 300) "150-299" — high burden
    [300, ∞)  "300+"     — very high burden (rarely sampled outside COLO320DM)

The bins are left-closed, right-open so that ``10`` falls in ``"10-49"``
and ``300`` falls in ``"300+"``. This convention matches the stage11 /
stage14b scripts and the figures.

This module exposes:

- :data:`DEFAULT_DENSITY_EDGES` — the paper's bin edges.
- :data:`DEFAULT_DENSITY_LABELS` — matching string labels.
- :func:`assign_density_bin` — single-image int → label.
- :func:`bin_counts` — vectorized version over a pandas Series / array.
- :func:`stratify_by_density` — aggregate a per-image metrics frame by
  bin, returning one row per bin (with a stable row ordering even when
  some bins are empty).

The edges and labels are configurable via ``config.evaluation.density_bins``
so that reviewers can experiment with alternative bin choices, but the
defaults here reproduce the published figures exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable, Sequence

import numpy as np

if TYPE_CHECKING:
    import pandas as pd


__all__ = [
    "DEFAULT_DENSITY_EDGES",
    "DEFAULT_DENSITY_LABELS",
    "assign_density_bin",
    "bin_counts",
    "stratify_by_density",
]


# ==============================================================================
# Default bin specification (paper §2 / Figure 6d/e / Supp §13).
# ==============================================================================

DEFAULT_DENSITY_EDGES: tuple[int, ...] = (0, 10, 50, 150, 300)
"""Lower bounds of each bin, in ascending order. The last bin extends to +∞."""

DEFAULT_DENSITY_LABELS: tuple[str, ...] = ("0-9", "10-49", "50-149", "150-299", "300+")
"""Human-readable labels, same length as :data:`DEFAULT_DENSITY_EDGES`."""


# ==============================================================================
# Helpers.
# ==============================================================================


def _check_edges_and_labels(
    edges: Sequence[int],
    labels: Sequence[str],
) -> None:
    """Validate that edges / labels match in length and are monotonic."""
    if len(edges) != len(labels):
        raise ValueError(
            f"edges and labels must have the same length; got "
            f"{len(edges)} edges and {len(labels)} labels"
        )
    if len(edges) == 0:
        raise ValueError("edges must not be empty")
    if any(edges[i + 1] <= edges[i] for i in range(len(edges) - 1)):
        raise ValueError(f"edges must be strictly increasing, got {edges!r}")


def assign_density_bin(
    gt_count: int | float,
    *,
    edges: Sequence[int] = DEFAULT_DENSITY_EDGES,
    labels: Sequence[str] = DEFAULT_DENSITY_LABELS,
) -> str:
    """
    Assign a per-image count bin label for an integer gold-standard count.

    Behavior matches the stage11 / stage14b implementations: bins are
    half-open on the right, the last bin is ``[edges[-1], ∞)``.

    Parameters
    ----------
    gt_count : int or float
        Number of annotated ecDNAs in the image. Non-negative.
    edges, labels : sequences, same length
        Bin specification. Defaults to the paper's convention.

    Returns
    -------
    str
        One of the entries in ``labels``.

    Raises
    ------
    ValueError
        If edges and labels are inconsistent, or if ``gt_count`` is
        negative (which would indicate an upstream bug, not a valid
        density).
    """
    _check_edges_and_labels(edges, labels)

    if gt_count < 0:
        raise ValueError(f"gt_count must be non-negative, got {gt_count}")

    # Walk the edges top-down so that the final "300+" catch-all falls
    # out naturally.
    for i in range(len(edges) - 1, -1, -1):
        if gt_count >= edges[i]:
            return labels[i]

    # Only reachable if gt_count < edges[0], which means it sits below
    # even the lowest bin; this should not happen with edges[0] == 0 and
    # gt_count ≥ 0, but we guard against it for defensive programming.
    return labels[0]


def bin_counts(
    gt_counts: Iterable[int | float],
    *,
    edges: Sequence[int] = DEFAULT_DENSITY_EDGES,
    labels: Sequence[str] = DEFAULT_DENSITY_LABELS,
) -> list[str]:
    """
    Vectorized equivalent of :func:`assign_density_bin`.

    Returns a list of labels in input order. The implementation is a
    simple Python loop because ``gt_counts`` is always under a few
    thousand entries in practice and a numpy ``digitize``-based version
    would complicate the boundary handling ("300" goes in "300+", not
    "150-299") without a measurable speed gain.
    """
    _check_edges_and_labels(edges, labels)
    return [
        assign_density_bin(int(c), edges=edges, labels=labels)
        for c in gt_counts
    ]


# ==============================================================================
# Stratified aggregation.
# ==============================================================================


def stratify_by_density(
    df: "pd.DataFrame",
    *,
    count_col: str = "ecDNA_gt",
    bin_col: str = "density_bin",
    labels: Sequence[str] = DEFAULT_DENSITY_LABELS,
    edges: Sequence[int] = DEFAULT_DENSITY_EDGES,
    overwrite: bool = False,
) -> "pd.DataFrame":
    """
    Attach a ``density_bin`` column to a per-image metrics DataFrame.

    The returned DataFrame is a copy with the new column appended; the
    input is not mutated. The ``density_bin`` column is a pandas
    ``Categorical`` with ``categories=labels`` in the natural low-to-high
    order, so downstream ``groupby(bin_col, observed=False)`` produces
    a stable row ordering in aggregates (and empty bins still appear in
    the output with zero counts).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``count_col`` (default ``ecDNA_gt``).
    count_col : str, default ``"ecDNA_gt"``
        Column holding the per-image GS count.
    bin_col : str, default ``"density_bin"``
        Name of the column to write.
    labels, edges : sequences
        Bin specification. Defaults to the paper's convention.
    overwrite : bool, default False
        If False and ``bin_col`` is already present, raise ``ValueError``
        rather than silently clobbering an existing column.

    Returns
    -------
    pandas.DataFrame
        Copy of the input with ``bin_col`` attached.
    """
    # Local import so that importing this module without pandas is cheap;
    # the evaluation-only install path may skip pandas.
    import pandas as pd

    if count_col not in df.columns:
        raise KeyError(
            f"count_col {count_col!r} not found in DataFrame; "
            f"available columns: {list(df.columns)}"
        )
    if bin_col in df.columns and not overwrite:
        raise ValueError(
            f"column {bin_col!r} already exists; pass overwrite=True to "
            f"replace it"
        )

    bins = bin_counts(df[count_col].to_numpy(), edges=edges, labels=labels)
    out = df.copy()
    out[bin_col] = pd.Categorical(
        bins, categories=list(labels), ordered=True
    )
    return out
