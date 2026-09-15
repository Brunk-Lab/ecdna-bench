#!/usr/bin/env python
"""
build_source_extended_data_fig8.py — one source-data file for Extended Data Fig. 8.

    python scripts/build_source_extended_data_fig8.py \
        --loco-table   outputs/eccount_loco/source_loco_paper_table_or_matching.csv \
        --run-root     outputs/eccount_loco \
        --benchmark    data/benchmark/dl_master_metadata_stage1_step3_consistency.csv \
        --in-distribution release/figures/notebook05/source_fig6_f1_by_cell_line_heatmap_long.csv \
        --out          outputs/eccount_loco/source_extended_data_fig8.csv

WHY THIS EXISTS
---------------
Extended Data Fig. 8 draws on three different artefacts: the scored LOCO metrics,
the per-run split composition, and the per-run training histories. Nature expects
one source-data file per figure. This concatenates them into a single tidy table
with a `panel` column, and — more importantly — a `scope` column that records
which image set every single value was computed over.

The scope column is not decoration. The NCI-H2170 leave-one-out model appears in
the figure twice with two different numbers (0.848 on all 888 images of the line,
0.833 on the 134 test images only), and the reason is scope. Any source file that
does not carry it invites exactly the confusion it is meant to prevent.

WHAT IT REFUSES TO DO
---------------------
It does not compute, interpolate or fill any metric. Every number written comes
from an input file. If an input is missing, its panel is skipped with a warning
and the rest still builds. If a column cannot be identified, the script says so
and leaves the field empty rather than guessing.

THE AUDIT
---------
After writing, it prints a scope audit and runs one specific check: for every
cost-of-novelty value it compares the full-precision difference against the
difference you would get from the rounded three-decimal values. Where those
disagree, the figure and the text can legitimately print different numbers for
the same quantity. That is the 0.095-versus-0.096 class of defect, and it is
silent unless something looks for it.

INPUTS
------
    --loco-table    source_loco_paper_table_{policy}.csv   (loco_paper_table.py)
    --run-root      directory holding one subdirectory per run, each with
                      split_composition.csv   role, cell_line, n_images, gt_objects
                      train_history.csv       per-epoch losses
    --benchmark     optional: benchmark metadata with cell line, split and a
                    per-image ground-truth count, for panel b
    --in-distribution  optional: the per-cell-line F1 CSV, read only to report
                    its scope in the audit

OUTPUT SCHEMA
-------------
    panel        a | b | c | d | e
    run          run directory name, or "" for benchmark-level rows
    cell_line    cell line the row describes, or ""
    series       role / split / train_loss / val_loss / in_distribution / unseen ...
    x            epoch number for panel c, otherwise empty
    metric       what `value` measures
    value        the number (or the role string for panel a)
    n_images     images behind the value, where meaningful
    scope        the image set the value was computed over
    source_file  the file the value came from
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("ed_fig8")

CELL_LINES = ["COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"]
PEAKS = "ecCount (peaks)"

# Locked in CLAUDE_INSTRUCTIONS.md §3. Used only as the reference line in panel
# d, recorded with its provenance, never mixed into a computed value.
RELEASED_MODEL_BIAS = 0.4

COLUMNS = ["panel", "run", "cell_line", "series", "x", "metric", "value",
           "n_images", "scope", "source_file"]


# --------------------------------------------------------------------------
# column identification — by content, never by assumed name
# --------------------------------------------------------------------------

def find_cell_line_column(df: pd.DataFrame) -> str | None:
    for c in df.columns:
        vals = {str(v).strip() for v in df[c].dropna().unique()}
        if len(vals & set(CELL_LINES)) >= 3:
            return c
    return None


def find_column(df: pd.DataFrame, names: tuple[str, ...]) -> str | None:
    lowered = {str(c).strip().lower(): c for c in df.columns}
    for n in names:
        if n in lowered:
            return lowered[n]
    return None


def find_loss_columns(df: pd.DataFrame) -> tuple[str | None, str | None, str | None]:
    """Return (epoch, train_loss, val_loss), detected rather than assumed."""
    epoch = find_column(df, ("epoch", "epochs", "step", "iteration"))
    train = find_column(df, ("train_loss", "training_loss", "loss_train",
                             "train", "loss"))
    val = find_column(df, ("val_loss", "valid_loss", "validation_loss",
                           "loss_val", "val"))
    return epoch, train, val


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def panel_a_and_c(run_root: Path) -> tuple[list[dict], list[dict],
                                           dict[str, pd.DataFrame], list[str]]:
    """Design matrix (a) and loss trajectories (c), from the run directories.

    Also returns the parsed split_composition frames, because panel b can be
    reconstructed from them without any external metadata file.
    """
    rows_a: list[dict] = []
    rows_c: list[dict] = []
    comps: dict[str, pd.DataFrame] = {}
    notes: list[str] = []

    # A directory is a run only if it carries at least one of the two files we
    # read. This keeps sibling directories such as figures/ or logs/ out of the
    # warning list instead of reporting them as broken runs.
    run_dirs = sorted(d for d in run_root.iterdir() if d.is_dir()
                      and ((d / "split_composition.csv").is_file()
                           or (d / "train_history.csv").is_file()))
    if not run_dirs:
        notes.append(f"No run subdirectories under {run_root} contain "
                     f"split_composition.csv or train_history.csv. "
                     f"Panels a and c skipped.")
        return rows_a, rows_c, comps, notes

    for run_dir in run_dirs:
        run = run_dir.name

        # ---- panel a: what each run held out -----------------------------
        sc = run_dir / "split_composition.csv"
        if sc.is_file():
            df = pd.read_csv(sc)
            line_col = find_cell_line_column(df)
            role_col = find_column(df, ("role", "kind", "assignment", "status"))
            n_col = find_column(df, ("n_images", "images", "n", "count"))
            obj_col = find_column(df, ("gt_objects", "objects", "n_objects"))

            if line_col is None:
                notes.append(f"{sc}: no cell-line column identified; run skipped "
                             f"in panel a.")
            elif role_col is None:
                notes.append(f"{sc}: no role column identified (saw "
                             f"{list(df.columns)}); run skipped in panel a, "
                             f"because train/val/eval counts cannot be told "
                             f"apart without it.")
            else:
                comps[run] = df
                for _, r in df.iterrows():
                    line = str(r[line_col]).strip()
                    if line not in CELL_LINES:
                        continue
                    # The role IS the series. A run carries a train row, a val
                    # row and an eval row for the same cell line; writing them
                    # all as "composition" makes 44 and 9 indistinguishable.
                    role = str(r[role_col]).strip()
                    base = dict(panel="a", run=run, cell_line=line, x="",
                                series=role, scope=f"run, {role} set",
                                source_file=str(sc))
                    if n_col is not None and pd.notna(r[n_col]):
                        rows_a.append({**base, "metric": "n_images",
                                       "value": r[n_col], "n_images": r[n_col]})
                    if obj_col is not None and pd.notna(r[obj_col]):
                        rows_a.append({**base, "metric": "gt_objects",
                                       "value": r[obj_col],
                                       "n_images": r[n_col] if n_col else ""})
        else:
            notes.append(f"{sc} not found; {run} contributes nothing to panel a.")

        # ---- panel c: training and validation loss ------------------------
        th = run_dir / "train_history.csv"
        if th.is_file():
            df = pd.read_csv(th)
            ep, tr, va = find_loss_columns(df)
            if ep is None or va is None:
                notes.append(f"{th}: could not identify epoch/val_loss columns "
                             f"(saw {list(df.columns)}); {run} skipped in panel c.")
            else:
                best_i = df[va].idxmin()
                for _, r in df.iterrows():
                    base = dict(panel="c", run=run, cell_line="", scope="run",
                                x=r[ep], source_file=str(th), n_images="")
                    if tr is not None and pd.notna(r[tr]):
                        rows_c.append({**base, "series": "train_loss",
                                       "metric": "loss", "value": r[tr]})
                    if pd.notna(r[va]):
                        rows_c.append({**base, "series": "val_loss",
                                       "metric": "loss", "value": r[va]})
                rows_c.append(dict(
                    panel="c", run=run, cell_line="", series="selected_checkpoint",
                    x=df.loc[best_i, ep], metric="val_loss_at_selected",
                    value=df.loc[best_i, va], n_images="", scope="run",
                    source_file=str(th)))
        else:
            notes.append(f"{th} not found; {run} contributes nothing to panel c.")

    return rows_a, rows_c, comps, notes


def panel_b(comps: dict[str, pd.DataFrame],
            benchmark_csv: Path | None) -> tuple[list[dict], list[str]]:
    """Benchmark composition by cell line and split.

    Preferred source is the runs themselves. In a leave-one-out run every image
    of the held-out line appears once with role 'eval', and that line's train
    and val counts appear in the other runs, so the four runs together
    reconstitute the whole benchmark. That needs no file outside the run tree
    and cannot drift from what was actually trained.

    A --benchmark CSV is used only as a fallback, and the consistency filter is
    applied when the column is present: the raw manifest holds more rows than
    the 1,145-image benchmark.
    """
    rows: list[dict] = []
    notes: list[str] = []

    # ---- preferred: reconstruct from the runs ---------------------------
    totals: dict[str, dict] = {}
    fitted: dict[str, dict[str, int]] = {l: {} for l in CELL_LINES}
    for run, df in comps.items():
        line_col = find_cell_line_column(df)
        role_col = find_column(df, ("role", "kind", "assignment", "status"))
        n_col = find_column(df, ("n_images", "images", "n", "count"))
        obj_col = find_column(df, ("gt_objects", "objects", "n_objects"))
        if None in (line_col, role_col, n_col):
            continue
        role = df[role_col].astype(str).str.strip().str.lower()
        line = df[line_col].astype(str).str.strip()
        ev = df[role.isin({"eval", "test", "held out", "held_out"})]
        fit = df[role.isin({"train", "val", "validation"})]
        ev_lines = {str(v).strip() for v in ev[line_col]}
        fit_lines = {str(v).strip() for v in fit[line_col]}
        # A true leave-one-out run: exactly one line evaluated, and that line
        # contributed nothing to training. The size-matched control fails this
        # and is correctly excluded.
        held = ev_lines - fit_lines
        if len(ev_lines) == 1 and len(held) == 1:
            l = held.pop()
            r = ev.iloc[0]
            totals[l] = {"n_images": int(r[n_col]),
                         "gt_objects": int(r[obj_col]) if obj_col else None,
                         "source": str(run)}
        for _, r in fit.iterrows():
            l = str(r[line_col]).strip()
            if l in CELL_LINES:
                key = "train" if role.loc[r.name].startswith("train") else "val"
                fitted[l][key] = int(r[n_col])

    if totals:
        for l in CELL_LINES:
            if l not in totals:
                notes.append(f"Panel b: no leave-one-out run evaluated {l}; "
                             f"its row is incomplete.")
                continue
            tot = totals[l]["n_images"]
            tr, va = fitted[l].get("train"), fitted[l].get("val")
            src = f"reconstructed from run split_composition.csv ({totals[l]['source']})"
            base = dict(panel="b", run="", cell_line=l, x="", scope="benchmark",
                        source_file=src)
            for series, val in (("train", tr), ("val", va),
                                ("test", None if tr is None or va is None
                                 else tot - tr - va), ("all", tot)):
                if val is None:
                    notes.append(f"Panel b: {l} {series} count unavailable.")
                    continue
                rows.append({**base, "series": series, "metric": "n_images",
                             "value": val, "n_images": val})
            if totals[l]["gt_objects"] is not None:
                obj = totals[l]["gt_objects"]
                rows.append({**base, "series": "all", "metric": "gt_objects",
                             "value": obj, "n_images": tot})
                rows.append({**base, "series": "all",
                             "metric": "gt_objects_per_image",
                             "value": round(obj / tot, 1), "n_images": tot})
        return rows, notes

    # ---- fallback: an external benchmark manifest ------------------------
    if benchmark_csv is None:
        notes.append("Panel b: could not be reconstructed from the runs and no "
                     "--benchmark given; skipped.")
        return rows, notes
    if not benchmark_csv.is_file():
        notes.append(f"{benchmark_csv} not found; panel b skipped.")
        return rows, notes

    df = pd.read_csv(benchmark_csv)
    # The release manifest carries rows the benchmark excludes.
    cons = find_column(df, ("count_mask_consistent", "consistent"))
    if cons is not None:
        before = len(df)
        df = df[df[cons].fillna(False).astype(bool)]
        notes.append(f"Panel b: applied the {cons} filter to "
                     f"{benchmark_csv.name} ({before} -> {len(df)} rows).")
    else:
        notes.append(f"Panel b: {benchmark_csv.name} has no consistency column; "
                     f"using all {len(df)} rows. Verify this is the benchmark "
                     f"subset and not the full manifest.")

    line_col = find_cell_line_column(df)
    split_col = find_column(df, ("split", "partition", "set", "fold"))
    count_col = find_column(df, ("ecdna_gt", "gt_count", "count", "n_objects",
                                 "ecdna_count", "gt_objects", "n_ecdna"))
    if line_col is None or split_col is None:
        notes.append(
            f"{benchmark_csv}: needs a cell-line column and a split column "
            f"(saw {list(df.columns)}). Panel b skipped rather than guessed.")
        return rows, notes

    for line, sub in df.groupby(df[line_col].astype(str).str.strip()):
        if line not in CELL_LINES:
            continue
        base = dict(panel="b", run="", cell_line=line, x="", scope="benchmark",
                    source_file=str(benchmark_csv))
        for split, s2 in sub.groupby(
                sub[split_col].astype(str).str.strip().str.lower()):
            rows.append({**base, "series": split, "metric": "n_images",
                         "value": len(s2), "n_images": len(s2)})
        rows.append({**base, "series": "all", "metric": "n_images",
                     "value": len(sub), "n_images": len(sub)})
        if count_col is not None:
            tot = pd.to_numeric(sub[count_col], errors="coerce").sum()
            rows.append({**base, "series": "all", "metric": "gt_objects",
                         "value": int(tot), "n_images": len(sub)})
            rows.append({**base, "series": "all",
                         "metric": "gt_objects_per_image",
                         "value": round(tot / len(sub), 1), "n_images": len(sub)})
        else:
            notes.append(f"{benchmark_csv}: no per-image count column found "
                         f"(saw {list(df.columns)}); panel b has image counts "
                         f"but no object counts.")
    return rows, notes


def panels_d_and_e(table_csv: Path, model: str) -> tuple[list[dict], pd.DataFrame, list[str]]:
    """Count bias (d) and cost of cell-line novelty (e), from the LOCO table."""
    rows: list[dict] = []
    notes: list[str] = []
    df = pd.read_csv(table_csv)

    need = {"held_out_cell_line", "run_kind", "model", "obj_f1"}
    missing = need - set(df.columns)
    if missing:
        notes.append(f"{table_csv} is missing {sorted(missing)}; panels d and e "
                     f"skipped.")
        return rows, pd.DataFrame(), notes

    m = df[(df["model"].astype(str).str.strip() == model)
           & (df["run_kind"].astype(str).str.strip() == "leave-one-out")].copy()
    if m.empty:
        notes.append(f"No 'leave-one-out' rows for {model!r} in {table_csv}; "
                     f"panels d and e skipped.")
        return rows, pd.DataFrame(), notes

    for _, r in m.iterrows():
        line = str(r["held_out_cell_line"]).strip()
        scope = str(r.get("scope", "")).strip()
        n = r.get("n_images", "")
        base = dict(run=f"hold_out_{line}", cell_line=line, x="",
                    n_images=n, scope=scope, source_file=str(table_csv))

        # panel d — signed count bias on the unseen line
        if "mean_signed_bias" in r and pd.notna(r["mean_signed_bias"]):
            rows.append({**base, "panel": "d", "series": "unseen",
                         "metric": "mean_signed_bias",
                         "value": r["mean_signed_bias"]})

        # panel e — in-distribution against unseen
        if pd.notna(r.get("in_distribution_obj_f1", np.nan)):
            rows.append({**base, "panel": "e", "series": "in_distribution",
                         "metric": "obj_f1", "value": r["in_distribution_obj_f1"],
                         "n_images": "", "scope": "SEE AUDIT — scope of the "
                                                  "in-distribution CSV"})
        rows.append({**base, "panel": "e", "series": "unseen",
                     "metric": "obj_f1", "value": r["obj_f1"]})
        if pd.notna(r.get("cost_of_novelty", np.nan)):
            rows.append({**base, "panel": "e", "series": "difference",
                         "metric": "cost_of_novelty",
                         "value": r["cost_of_novelty"]})

    # the released model's bias, drawn as the reference line in panel d
    rows.append(dict(
        panel="d", run="released_model", cell_line="", series="reference",
        x="", metric="mean_signed_bias", value=RELEASED_MODEL_BIAS,
        n_images=1145, scope="full benchmark, all 1,145 images",
        source_file="CLAUDE_INSTRUCTIONS.md §3 (locked)"))

    return rows, m, notes


# --------------------------------------------------------------------------
# audit
# --------------------------------------------------------------------------

def rounding_audit(m: pd.DataFrame) -> list[str]:
    """Flag values where the printed difference cannot be reproduced from the
    printed operands. This is the 0.095-versus-0.096 defect."""
    out: list[str] = []
    if m.empty or "cost_of_novelty" not in m.columns:
        return ["cost_of_novelty absent — rounding audit not run."]
    out.append(f"{'cell line':<12}{'in-dist':>9}{'unseen':>9}"
               f"{'true diff':>11}{'prints':>8}{'from rounded':>14}")
    out.append("-" * 63)
    for _, r in m.iterrows():
        ind, un = r.get("in_distribution_obj_f1"), r["obj_f1"]
        if pd.isna(ind):
            continue
        true = ind - un
        naive = round(ind, 3) - round(un, 3)
        flag = "" if round(true, 3) == round(naive, 3) else "   <-- MISMATCH"
        out.append(f"{str(r['held_out_cell_line']):<12}{ind:>9.4f}{un:>9.4f}"
                   f"{true:>11.7f}{round(true, 3):>8.3f}"
                   f"{round(naive, 3):>14.3f}{flag}")
    out.append("")
    out.append("A MISMATCH means the figure prints one number and a reader "
               "subtracting the")
    out.append("two displayed values gets another. Always plot the "
               "cost_of_novelty column,")
    out.append("never a difference computed from rounded labels.")
    return out


def scope_audit(m: pd.DataFrame, in_dist_csv: Path | None) -> list[str]:
    out: list[str] = []
    if not m.empty:
        out.append(f"{'cell line':<12}{'n_images':>10}  scope")
        out.append("-" * 63)
        for _, r in m.iterrows():
            out.append(f"{str(r['held_out_cell_line']):<12}"
                       f"{r.get('n_images', ''):>10}  {r.get('scope', '')}")
        out.append("")

    out.append("The in-distribution values are NOT from this table — they are "
               "joined in from")
    out.append("the per-cell-line F1 CSV, and that file's scope decides whether "
               "panel e")
    out.append("compares like with like.")
    if in_dist_csv is not None and in_dist_csv.is_file():
        df = pd.read_csv(in_dist_csv)
        n_col = find_column(df, ("n_images", "images", "n"))
        line_col = find_cell_line_column(df)
        out.append(f"  file    : {in_dist_csv}")
        out.append(f"  columns : {list(df.columns)}")
        if n_col and line_col:
            sub = df[df[line_col].astype(str).str.strip() == "NCI-H2170"]
            if len(sub):
                vals = sorted({int(v) for v in
                               pd.to_numeric(sub[n_col], errors="coerce").dropna()})
                out.append(f"  NCI-H2170 n_images = {vals}")
                if 888 in vals:
                    out.append("  -> ALL 888 images. The released model trained "
                               "on 621 of these,")
                    out.append("     so panel e's in-distribution bar includes "
                               "training data while")
                    out.append("     the unseen bar does not. Say so in the "
                               "caption, or rescore.")
                elif 134 in vals:
                    out.append("  -> the 134 held-out test images. Panel e is "
                               "scope-clean.")
        else:
            out.append("  No n_images column — open the file and check which "
                       "image set it")
            out.append("  covers before quoting panel e as a like-for-like "
                       "comparison.")
    else:
        out.append("  --in-distribution not given, so the scope could not be "
                   "reported.")
        out.append("  Check it by hand: if NCI-H2170 covers 888 images the "
                   "comparison is")
        out.append("  scope-mixed; if 134, it is clean.")
    return out


# --------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--loco-table", required=True, type=Path)
    p.add_argument("--run-root", required=True, type=Path)
    p.add_argument("--benchmark", type=Path, default=None)
    p.add_argument("--in-distribution", type=Path, default=None)
    p.add_argument("--model", default=PEAKS)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    if not args.loco_table.is_file():
        logger.error("LOCO table not found: %s", args.loco_table)
        return 2
    if not args.run_root.is_dir():
        logger.error("Run root not found: %s", args.run_root)
        return 2

    notes: list[str] = []
    rows_a, rows_c, comps, n1 = panel_a_and_c(args.run_root)
    rows_b, n2 = panel_b(comps, args.benchmark)
    rows_de, m, n3 = panels_d_and_e(args.loco_table, args.model)
    notes += n1 + n2 + n3

    rows = rows_a + rows_b + rows_c + rows_de
    if not rows:
        logger.error("Nothing was assembled. Check the inputs above.")
        return 3

    out = pd.DataFrame(rows).reindex(columns=COLUMNS)
    out = out.sort_values(["panel", "run", "cell_line", "series", "x"],
                          kind="stable").reset_index(drop=True)

    out_path = args.out or (args.run_root / "source_extended_data_fig8.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    w = 68
    print()
    print("=" * w)
    print("  SOURCE DATA — Extended Data Fig. 8")
    print("=" * w)
    print(f"  written : {out_path}")
    print(f"  rows    : {len(out)}")
    for panel, sub in out.groupby("panel"):
        print(f"    panel {panel} : {len(sub):>5} rows   "
              f"({', '.join(sorted(set(sub['metric'])))[:44]})")

    print()
    print("=" * w)
    print("  ROUNDING AUDIT")
    print("=" * w)
    for line in rounding_audit(m):
        print("  " + line)

    print()
    print("=" * w)
    print("  SCOPE AUDIT")
    print("=" * w)
    for line in scope_audit(m, args.in_distribution):
        print("  " + line)

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
