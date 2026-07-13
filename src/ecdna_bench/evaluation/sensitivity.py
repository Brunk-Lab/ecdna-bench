"""
ecdna_bench.evaluation.sensitivity — grid sweep of matching parameters.

Given a collection of images and, for each image, a precomputed
:class:`~ecdna_bench.evaluation.matching.PairwiseTensors`, this module
evaluates every combination of ``(d_max, min_iou, policy)`` on the grid
and returns a tidy DataFrame with one row per (model, image, d_max,
min_iou, policy). The benchmark orchestrator feeds this into Supplementary
Figure 6.

Why a dedicated module?
-----------------------
The sweep is dominated by the *per-image* cost of computing pairwise
geometry (one Python loop over prediction × GT pairs, with a small ROI
extraction for the IoU on each valid pair). The per-*parameter* cost is
just an elementwise mask of precomputed tensors plus a small Hungarian
solve, which is cheap. So the right shape for the code is:

    for image in images:
        pw = precompute_pairwise(…)          # once per image — expensive
        for params in grid:
            res = resolve_matching_from_pairwise(pw, …)   # cheap
            yield (image, params, res)

The original stage-script implementations did this correctly but the
logic was entangled with mask loading, figure plotting, and CSV writing
in the same file. This module isolates just the sweep.

Output schema
-------------
The returned DataFrame carries one row per ``(image_id, d_max, min_iou,
policy)`` cell with columns::

    image_id, d_max, min_iou, policy,
    tp, fp, fn, ignored, n_pred, n_gt,
    precision, recall, f1

If the caller also provides ``model_id``, ``cell_line``, ``split``, or
any other per-image metadata, those columns are copied through verbatim.
This makes it trivial to groupby for the figures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

from ecdna_bench.evaluation.matching import (
    MatchPolicy,
    PairwiseTensors,
    resolve_matching_from_pairwise,
)
from ecdna_bench.evaluation.metrics import object_metrics_from_counts

if TYPE_CHECKING:
    import pandas as pd


__all__ = [
    "SweepRow",
    "sweep_one_image",
    "sweep_matching_grid",
]


# ==============================================================================
# Row container.
# ==============================================================================


@dataclass
class SweepRow:
    """
    One output row of a matching sweep.

    The ``extra`` dict carries arbitrary per-image metadata (``model_id``,
    ``cell_line``, ``split``, …) supplied by the caller. It is expanded
    into columns of the final DataFrame, so any key you add here shows
    up as a column for free.
    """

    image_id: str
    d_max: float
    min_iou: float
    policy: str
    tp: int
    fp: int
    fn: int
    ignored: int
    n_pred: int
    n_gt: int
    precision: float
    recall: float
    f1: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Flatten into a single-level dict suitable for a DataFrame row."""
        row = {
            "image_id": self.image_id,
            "d_max": self.d_max,
            "min_iou": self.min_iou,
            "policy": self.policy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "ignored": self.ignored,
            "n_pred": self.n_pred,
            "n_gt": self.n_gt,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }
        # ``extra`` values win over core keys to allow the caller to
        # override (e.g., replacing ``image_id`` with a longer form) —
        # but we sort by merging so the core keys come first in the
        # dict, which is the order pandas will use for the DataFrame.
        for k, v in self.extra.items():
            if k not in row:
                row[k] = v
        return row


# ==============================================================================
# Per-image sweep.
# ==============================================================================


def sweep_one_image(
    image_id: str,
    pw: PairwiseTensors,
    *,
    d_max_grid: Sequence[float],
    min_iou_grid: Sequence[float],
    policies: Sequence[MatchPolicy] = ("OR", "AND"),
    alpha: float = 0.5,
    extra: Mapping[str, Any] | None = None,
) -> list[SweepRow]:
    """
    Sweep a single image's precomputed pairwise tensors over the matching
    parameter grid.

    Parameters
    ----------
    image_id : str
        Canonical image identifier. Usually the ``unique_id`` from the
        benchmark metadata.
    pw : PairwiseTensors
        Output of :func:`precompute_pairwise` for this image. Its
        ``max_precompute_dist`` must be at least ``max(d_max_grid)``;
        otherwise :func:`resolve_matching_from_pairwise` will raise.
    d_max_grid, min_iou_grid : sequences
        Parameter grids. The Cartesian product of
        ``d_max_grid × min_iou_grid × policies`` is swept.
    policies : sequence of {"OR", "AND"}, default ``("OR", "AND")``
        Validity policies to evaluate.
    alpha : float, default 0.5
        Cost-blend parameter; normally the paper's fixed 0.5.
    extra : mapping, optional
        Extra metadata attached to every row (copied into
        :attr:`SweepRow.extra`).

    Returns
    -------
    list[SweepRow]
        Length equals ``len(d_max_grid) × len(min_iou_grid) × len(policies)``.
    """
    extra_dict = dict(extra) if extra else {}
    rows: list[SweepRow] = []

    for d_max in d_max_grid:
        for min_iou in min_iou_grid:
            for policy in policies:
                result = resolve_matching_from_pairwise(
                    pw,
                    d_max=float(d_max),
                    min_iou=float(min_iou),
                    alpha=float(alpha),
                    policy=policy,
                )

                tp = int(result.tp)
                fp = int(result.fp)
                fn = int(result.fn)
                ignored = int(result.ignored)

                m = object_metrics_from_counts(
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    ignored=ignored,
                )

                rows.append(
                    SweepRow(
                        image_id=image_id,
                        d_max=float(d_max),
                        min_iou=float(min_iou),
                        policy=str(policy),
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        ignored=ignored,
                        n_pred=tp + fp + ignored,
                        n_gt=tp + fn,
                        precision=float(m["precision"]),
                        recall=float(m["recall"]),
                        f1=float(m["f1"]),
                        extra=dict(extra_dict),
                    )
                )

    return rows

# ==============================================================================
# Multi-image sweep → tidy DataFrame.
# ==============================================================================


def sweep_matching_grid(
    items: Iterable[tuple[str, PairwiseTensors, Mapping[str, Any]]],
    *,
    d_max_grid: Sequence[float],
    min_iou_grid: Sequence[float],
    policies: Sequence[MatchPolicy] = ("OR", "AND"),
    alpha: float = 0.5,
) -> "pd.DataFrame":
    """
    Run :func:`sweep_one_image` across many images and concatenate the
    results into one tidy DataFrame.

    Parameters
    ----------
    items : iterable of ``(image_id, pairwise_tensors, extra_metadata)``
        Each item triples an image identifier, its precomputed pairwise
        geometry, and a mapping of extra metadata (e.g.
        ``{"model_id": "ecCount", "cell_line": "NCI-H2170", "split": "test"}``)
        that is copied into every output row for that image.
    d_max_grid, min_iou_grid, policies, alpha
        See :func:`sweep_one_image`.

    Returns
    -------
    pandas.DataFrame
        One row per ``(image_id, d_max, min_iou, policy)``. Columns are
        the twelve canonical fields from :class:`SweepRow` plus every
        key present in any ``extra`` mapping. Missing extra keys on a
        given row become ``NaN``.
    """
    # Local import so non-pandas users of the eval package don't pay the
    # import cost when they only need :func:`sweep_one_image`.
    import pandas as pd

    all_rows: list[dict[str, Any]] = []
    for image_id, pw, extra in items:
        rows = sweep_one_image(
            image_id,
            pw,
            d_max_grid=d_max_grid,
            min_iou_grid=min_iou_grid,
            policies=policies,
            alpha=alpha,
            extra=extra,
        )
        all_rows.extend(r.to_dict() for r in rows)

    if not all_rows:
        # Return an empty DataFrame with the canonical columns so
        # downstream code does not have to special-case this.
        return pd.DataFrame(
            columns=[
                "image_id",
                "d_max",
                "min_iou",
                "policy",
                "tp",
                "fp",
                "fn",
                "ignored",
                "n_pred",
                "n_gt",
                "precision",
                "recall",
                "f1",
            ]
        )

    return pd.DataFrame(all_rows)
