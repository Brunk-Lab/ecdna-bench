#!/usr/bin/env python
"""
rescore_in_distribution_by_scope.py — object F1 by image scope, both sides.

    # in-distribution only (the released model, per cell line)
    python scripts/rescore_in_distribution_by_scope.py \
        --per-image  release/frozen_results/or_matching/per_image_metrics.csv \
        --metadata   release/manifests/dl_master_metadata_stage1_step3_consistency.csv \
        --loco-table outputs/eccount_loco/source_loco_paper_table_or_matching.csv \
        --out        outputs/eccount_loco/in_distribution_by_scope.csv

    # both sides — adds the leave-one-out runs, so every row of Fig. 6d can be
    # put on the same footing
    python scripts/rescore_in_distribution_by_scope.py \
        --per-image     release/frozen_results/or_matching/per_image_metrics.csv \
        --metadata      release/manifests/dl_master_metadata_stage1_step3_consistency.csv \
        --loco-run-root outputs/eccount_loco \
        --loco-table    outputs/eccount_loco/source_loco_paper_table_or_matching.csv \
        --out           outputs/eccount_loco/f1_by_scope.csv

WHAT PROBLEM THIS SOLVES
------------------------
Every row of Fig. 6d is scored on all images of its cell line. The image sets
therefore already match. What does NOT match is exposure: the six
in-distribution rows are scored partly on images their own model trained on,
while the bottom row is a model that never saw the line. The difference between
them — Fig. 6f's "cost of an unseen cell line" — is consequently inflated by
however much each model gained from memorising its training split.

There is exactly one image set on which no model in that panel has an
advantage: each cell line's held-out test split. Object F1 here is
micro-averaged — pooled TP, FP and FN, then F1 = 2TP / (2TP + FP + FN) — so
both sides can be re-pooled on that split from per-image metrics that already
exist. No GPU, no rescoring.

With --loco-run-root this script pools BOTH sides at BOTH scopes and prints
what Fig. 6d looks like either way, so the choice can be made by looking at the
numbers instead of by argument.

THE CATCH, STATED UP FRONT
--------------------------
The test splits are 11 (COLO320DM), 134 (NCI-H2170), 11 (NCI-H716) and 19
(SNU16) images. A like-for-like panel is scope-correct but rests on eleven
images for two of the four lines, and per-cell-line values on those splits swing
by up to 0.11. The script flags every cell whose n falls below --min-images so
that trade-off is visible rather than buried.

It computes F1 from pooled counts only. It never rescales, interpolates or
carries a number across scopes.

INPUTS
------
    --per-image      per-image metrics for the released model: a model column,
                     tp/fp/fn, and enough to identify each image's cell line and
                     split. Columns are identified by content, not by name.
    --metadata       optional: uid -> cell_line, split, if --per-image lacks them
    --loco-run-root  optional: the directory holding holdout_* run directories.
                     Each contributes its own per-image metrics and its held-out
                     cell line is read from its split_composition.csv.
    --loco-table     optional: source_loco_paper_table_*.csv, to compare the
                     published cost of novelty against the scope-clean one
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger("rescope")

CELL_LINES = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]
TEST_TOKENS = {"test", "held-out", "held_out", "heldout", "holdout"}
FIT_TOKENS = {"train", "training", "val", "valid", "validation"}
PEAKS = "ecCount (peaks)"

ALL, TEST = "all rows", "test rows"
IN_DIST, LOCO, CONTROL = ("in-distribution", "leave-one-out",
                          "size-matched control")


# --------------------------------------------------------------------------
# column identification — by content, never by assumed name
# --------------------------------------------------------------------------

def find_cell_line_column(df: pd.DataFrame) -> str | None:
    """By content first: the column holding at least three of the four lines.

    Falls back to an exact 'cell_line' name match, because a per-image file
    covering only one cell line — which is exactly what a leave-one-out run
    produces — cannot be identified by content.
    """
    for c in df.columns:
        vals = {str(v).strip() for v in df[c].dropna().unique()}
        if len(vals & set(CELL_LINES)) >= 3:
            return c
    for c in df.columns:
        if str(c).strip().lower() in ("cell_line", "cellline", "cell line"):
            vals = {str(v).strip() for v in df[c].dropna().unique()}
            if vals & set(CELL_LINES):
                return c
    return None


def find_split_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        vals = {str(v).strip().lower() for v in df[c].dropna().unique()}
        if vals & TEST_TOKENS and len(vals) <= 6:
            return c
    return None


def find_named(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in lowered:
            return lowered[n]
    return None


def normalise_model(name: object) -> str:
    """Map registry keys onto the display names the LOCO table uses.

    Copied from loco_paper_table.py so both scripts agree. Order matters: the
    threshold model's display name contains 'mask' AND 'thresh', and the peaks
    name contains neither, so 'peak' is tested first.
    """
    s = str(name).lower()
    if "peak" in s:
        return PEAKS
    if "mask" in s or "thresh" in s:
        return "ecCount (threshold mask)"
    return str(name)


def f1_from_counts(tp: float, fp: float, fn: float) -> float:
    denom = 2 * tp + fp + fn
    return float("nan") if denom == 0 else 2 * tp / denom


# --------------------------------------------------------------------------
# pooling
# --------------------------------------------------------------------------

def metric_columns(df: pd.DataFrame) -> tuple[str, str, str, str] | None:
    tp = find_named(df, ("obj_tp", "tp", "true_positives", "n_tp"))
    fp = find_named(df, ("obj_fp", "fp", "false_positives", "n_fp"))
    fn = find_named(df, ("obj_fn", "fn", "false_negatives", "n_fn"))
    model = find_named(df, ("model", "method", "model_name"))
    if None in (tp, fp, fn, model):
        return None
    return tp, fp, fn, model


def pool(df: pd.DataFrame, condition: str, source: str,
         line_override: str | None = None) -> tuple[list[dict], list[str]]:
    """Pool TP/FP/FN per model, per cell line, at both scopes."""
    notes: list[str] = []
    cols = metric_columns(df)
    if cols is None:
        return [], [f"{source}: needs tp/fp/fn and a model column; saw "
                    f"{list(df.columns)}."]
    tp_c, fp_c, fn_c, model_c = cols

    split_c = find_split_column(df)
    if split_c is None:
        return [], [f"{source}: no split column; cannot separate test rows."]

    if line_override is not None:
        df = df.copy()
        df["_line"] = line_override
    else:
        line_c = find_cell_line_column(df)
        if line_c is None:
            return [], [f"{source}: no cell-line column; saw "
                        f"{list(df.columns)}."]
        df = df.copy()
        df["_line"] = df[line_c].astype(str).str.strip()

    df["_split"] = df[split_c].astype(str).str.strip().str.lower()
    df["_model"] = df[model_c].map(normalise_model)
    df = df[df["_line"].isin(CELL_LINES)]

    rows = []
    for (model, line), sub in df.groupby(["_model", "_line"]):
        for scope, s in ((ALL, sub),
                         (TEST, sub[sub["_split"].isin(TEST_TOKENS)])):
            if not len(s):
                continue
            tp, fp, fn = s[tp_c].sum(), s[fp_c].sum(), s[fn_c].sum()
            rows.append(dict(model=model, cell_line=line, condition=condition,
                             scope=scope, n_images=len(s), tp=int(tp),
                             fp=int(fp), fn=int(fn),
                             object_f1=f1_from_counts(tp, fp, fn),
                             source=source))
    return rows, notes


def held_out_line(run_dir: Path) -> tuple[str | None, str]:
    """Which cell line a run scored, and in what role.

    A leave-one-out run evaluates exactly one line and trained on none of it.
    The size-matched control also evaluates one line, but trained on some of
    it — which is precisely why it is scored on that line's test rows only.
    """
    sc = run_dir / "split_composition.csv"
    if not sc.is_file():
        return None, ""
    df = pd.read_csv(sc)
    line_c = find_cell_line_column(df)
    role_c = find_named(df, ("role", "kind", "assignment", "status"))
    if line_c is None or role_c is None:
        return None, ""
    role = df[role_c].astype(str).str.strip().str.lower()
    ev = {str(v).strip() for v in df[line_c][role.isin({"eval"})]}
    fit = {str(v).strip() for v in df[line_c][role.isin(FIT_TOKENS)]}
    held = ev - fit
    if len(held) == 1:
        return held.pop(), LOCO
    if len(ev) == 1:
        return ev.pop(), CONTROL
    return None, ""


def read_loco_runs(root: Path, policy: str) -> tuple[list[dict], list[str]]:
    rows: list[dict] = []
    notes: list[str] = []
    if not root.is_dir():
        return rows, [f"{root} is not a directory; leave-one-out runs skipped."]

    for run_dir in sorted(d for d in root.iterdir() if d.is_dir()):
        pim = run_dir / "results" / policy / "per_image_metrics.csv"
        if not pim.is_file():
            hits = sorted(run_dir.glob(f"results/**/{policy}/per_image_metrics.csv"))
            hits = hits or sorted(run_dir.glob("results/**/per_image_metrics.csv"))
            if not hits:
                continue
            pim = hits[0]
            notes.append(f"{run_dir.name}: used {pim} (the {policy} path was "
                         f"not where expected).")

        line, condition = held_out_line(run_dir)
        if line is None:
            notes.append(f"{run_dir.name}: could not determine which cell line "
                         f"it scored; skipped.")
            continue
        r, n = pool(pd.read_csv(pim), condition, f"{condition} ({run_dir.name})",
                    line_override=line)
        rows += r
        notes += n
        logger.info("%s: %s on %s, %d rows pooled.", run_dir.name, condition,
                    line, len(r))
    return rows, notes


# --------------------------------------------------------------------------

def get(out: pd.DataFrame, model: str, line: str, condition: str,
        scope: str) -> tuple[float, int] | None:
    r = out[(out.model == model) & (out.cell_line == line)
            & (out.condition == condition) & (out.scope == scope)]
    return (float(r.object_f1.iloc[0]), int(r.n_images.iloc[0])) if len(r) else None


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-image", required=True, type=Path)
    p.add_argument("--metadata", type=Path, default=None)
    p.add_argument("--loco-run-root", type=Path, default=None)
    p.add_argument("--policy", default="or_matching")
    p.add_argument("--loco-table", type=Path, default=None)
    p.add_argument("--model", default=PEAKS)
    p.add_argument("--min-images", type=int, default=25,
                   help="Flag any scope whose image count falls below this.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not args.per_image.is_file():
        logger.error("Not found: %s", args.per_image)
        return 2

    df = pd.read_csv(args.per_image)
    if metric_columns(df) is None:
        logger.error("Need tp/fp/fn and a model column; saw %s.",
                     list(df.columns))
        return 3

    # join cell line / split from the manifest when the metrics lack them
    if find_cell_line_column(df) is None or find_split_column(df) is None:
        if args.metadata is None:
            logger.error("--per-image has no cell-line or split column. Pass "
                         "--metadata to join them.")
            return 3
        if not args.metadata.is_file():
            logger.error("Not found: %s", args.metadata)
            return 2
        meta = pd.read_csv(args.metadata)
        uids = ("uid", "unique_id", "image_id", "image_uid", "image", "id",
                "filename")
        uid_l, uid_r = find_named(df, uids), find_named(meta, uids)
        if uid_l is None or uid_r is None:
            logger.error("Cannot join metadata: no shared uid column "
                         "(per-image %s / metadata %s).",
                         list(df.columns), list(meta.columns))
            return 3
        keep = [uid_r] + [c for c in (find_cell_line_column(meta),
                                      find_split_column(meta)) if c]
        df = df.merge(meta[keep].drop_duplicates(uid_r), left_on=uid_l,
                      right_on=uid_r, how="left", suffixes=("", "_meta"))

    rows, notes = pool(df, IN_DIST, str(args.per_image))
    if not rows:
        for n in notes:
            logger.error(n)
        return 3

    if args.loco_run_root is not None:
        r, n = read_loco_runs(args.loco_run_root, args.policy)
        rows += r
        notes += n

    out = pd.DataFrame(rows).sort_values(
        ["model", "cell_line", "condition", "scope"]).reset_index(drop=True)
    out_path = args.out or args.per_image.with_name("f1_by_scope.csv")
    out.to_csv(out_path, index=False)

    w = 82
    print()
    print("=" * w)
    print("  OBJECT F1 BY SCOPE   (pooled TP/FP/FN, F1 = 2TP/(2TP+FP+FN))")
    print("=" * w)
    print(f"  written: {out_path}   ({len(out)} rows)\n")
    piv = out[out.condition == IN_DIST].pivot_table(
        index=["model", "cell_line"], columns="scope",
        values=["n_images", "object_f1"])
    print(piv.to_string())

    h = out[(out.cell_line == "NCI-H2170") & (out.scope == ALL)
            & (out.condition == IN_DIST) & (out.model == PEAKS)]
    if len(h):
        n = int(h.n_images.iloc[0])
        print()
        if n == 888:
            print(f"  CHECK: NCI-H2170 all-rows n = {n}, {PEAKS} pooled F1 = "
                  f"{h.object_f1.iloc[0]:.4f}.")
            print("  If that matches the per-cell-line heatmap CSV, this is the "
                  "same scoring run.")
        else:
            print(f"  WARNING: NCI-H2170 all-rows n = {n}, not 888. This "
                  f"per-image file does")
            print("  not cover the whole benchmark.")

    # ---- what Fig. 6d looks like under each scope ------------------------
    if args.loco_run_root is not None:
        m = args.model
        print()
        print("=" * w)
        print(f"  FIG. 6d UNDER EACH SCOPE — {m}")
        print("=" * w)
        print(f"  {'':<12}{'ALL ROWS (as published)':>34}"
              f"{'TEST ROWS (like-for-like)':>34}")
        print(f"  {'cell line':<12}{'in-dist':>9}{'unseen':>9}{'cost':>8}"
              f"{'n':>8}{'in-dist':>10}{'unseen':>9}{'cost':>8}{'n':>7}")
        print("  " + "-" * (w - 4))
        thin = []
        for line in CELL_LINES:
            cells, vals = [], {}
            for scope in (ALL, TEST):
                a = get(out, m, line, IN_DIST, scope)
                b = get(out, m, line, LOCO, scope)
                if a is None or b is None:
                    cells.append(f"{'—':>9}{'—':>9}{'—':>8}{'—':>8}")
                    continue
                vals[scope] = a[0] - b[0]
                cells.append(f"{a[0]:>9.3f}{b[0]:>9.3f}{a[0]-b[0]:>8.3f}"
                             f"{b[1]:>8d}")
                if scope == TEST and b[1] < args.min_images:
                    thin.append(f"{line} (n = {b[1]})")
            print(f"  {line:<12}{cells[0]}{cells[1][:1]}{cells[1][1:]}")
        print()
        if thin:
            print(f"  Fewer than {args.min_images} test images: "
                  f"{', '.join(thin)}.")
            print("  A like-for-like panel is scope-correct but rests on those "
                  "counts. Judge")
            print("  the swing between the two 'unseen' columns before "
                  "adopting it.")
        print("  Both sides of the TEST ROWS block exclude every image any of "
              "these models")
        print("  trained on. The ALL ROWS block does not, on the in-dist side.")

    # ---- published cost of novelty against the clean one -----------------
    if args.loco_table is not None and args.loco_table.is_file():
        tab = pd.read_csv(args.loco_table)
        clean = out[(out.scope == TEST) & (out.condition == IN_DIST)] \
            .set_index(["model", "cell_line"])
        print()
        print("=" * w)
        print("  COST OF NOVELTY — as published vs against a clean "
              "in-distribution value")
        print("=" * w)
        print(f"  {'model':<26}{'cell line':<12}{'published':>11}"
              f"{'clean in-dist':>15}{'change':>9}")
        print("  " + "-" * (w - 4))
        for _, r in tab[tab.run_kind.astype(str).str.strip() == LOCO].iterrows():
            key = (str(r["model"]).strip(), str(r["held_out_cell_line"]).strip())
            pub = r.get("cost_of_novelty")
            if key not in clean.index or pd.isna(pub):
                continue
            new = clean.loc[key, "object_f1"] - r["obj_f1"]
            print(f"  {key[0]:<26}{key[1]:<12}{pub:>11.3f}{new:>15.3f}"
                  f"{new - pub:>+9.3f}")
        print()
        print("  This column still pairs a test-rows in-distribution value with "
              "an all-rows")
        print("  leave-one-out one. The fully like-for-like figure is the TEST "
              "ROWS block above.")

    if notes:
        print()
        print("=" * w)
        print("  SKIPPED / INCOMPLETE")
        print("=" * w)
        for n in notes:
            print("  - " + n)

    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())