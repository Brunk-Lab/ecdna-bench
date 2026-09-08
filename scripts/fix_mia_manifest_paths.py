#!/usr/bin/env python
"""
fix_mia_manifest_paths.py — fill the two empty MIA columns in the release
manifest, and audit every other column for the same problem while it is open.

    # look first
    python scripts/fix_mia_manifest_paths.py \\
        --manifest release/manifests/dl_master_metadata_stage1_step3_consistency.csv \\
        --mia-root release/harmonized_masks/mia \\
        --original-mia-root data/baselines/mia_predictions

    # then write a NEW file (never the locked one)
    python scripts/fix_mia_manifest_paths.py ... \\
        --out outputs/manifests/dl_master_metadata_with_mia_paths.csv

WHY THIS EXISTS
---------------
`mia_mask_relpath` and `original_mia_name` are present in the manifest and empty
for every row. The manifest is how a reader navigates 27,080 deposited files, so
a reader cannot get from an image to its MIA prediction. Nobody noticed because
the neighbouring columns (`gt_mask_relpath`, `roi_mask_relpath`) are complete.

WHAT IT DOES
------------
Indexes the MIA mask files under --mia-root by image UID, joins them onto the
manifest, and writes the result. Matching is by exact UID stem first, then by
UID-contained-in-stem, and a UID that matches more than one file is reported and
left empty rather than guessed at.

It also prints a completeness audit of every column, because two columns being
silently empty is the kind of defect that comes in pairs.

WHAT IT REFUSES TO DO
---------------------
It does not write into the manifest in place. `release/` is a locked artifact
tree; the default is to write a new file elsewhere and let a human decide what
becomes canonical. `--in-place` exists but warns loudly and takes a backup.

It never invents a path. A row with no matching file keeps its empty value and
appears in the unmatched report.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

MASK_EXT = {".png", ".tif", ".tiff", ".npz", ".npy"}
MIA_COL = "mia_mask_relpath"
ORIG_COL = "original_mia_name"

# Columns whose emptiness is expected and not a defect.
EXPECTED_SPARSE = {"notes", "comment", "comments", "exclusion_reason"}


def find_uid_column(df: pd.DataFrame) -> str | None:
    """The UID column is the one whose values are unique, string-like and used
    as filename stems. Detect it rather than assume a name."""
    for name in ("uid", "image_uid", "image_id", "id", "unique_id", "sample_uid"):
        for c in df.columns:
            if str(c).strip().lower() == name:
                return c
    # fall back: first column that is unique across all rows and looks textual
    for c in df.columns:
        s = df[c]
        if s.is_unique and s.dtype == object and s.notna().all():
            if s.astype(str).str.len().median() > 8:
                return c
    return None


def index_masks(root: Path) -> tuple[dict[str, list[Path]], int]:
    """stem -> paths, for every mask-like file under root."""
    idx: dict[str, list[Path]] = defaultdict(list)
    n = 0
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in MASK_EXT:
            idx[p.stem].append(p)
            n += 1
    return idx, n


def match_uid(uid: str, idx: dict[str, list[Path]],
              stems: list[str]) -> tuple[Path | None, str]:
    """Exact stem, then unique containment. Ambiguity is reported, never
    resolved by picking the first."""
    if uid in idx:
        hits = idx[uid]
        return (hits[0], "exact") if len(hits) == 1 else (None, "ambiguous-exact")
    contained = [s for s in stems if uid in s]
    if len(contained) == 1:
        hits = idx[contained[0]]
        return (hits[0], "contained") if len(hits) == 1 else (None, "ambiguous")
    if len(contained) > 1:
        return None, "ambiguous"
    return None, "missing"


def completeness_audit(df: pd.DataFrame) -> list[str]:
    out, n = [], len(df)
    out.append(f"{'column':<44}{'filled':>8}{'empty':>8}  state")
    out.append("-" * 76)
    for c in df.columns:
        s = df[c]
        empty = int(s.isna().sum() + (s.astype(str).str.strip() == "").sum()
                    - (s.isna() & (s.astype(str).str.strip() == "")).sum())
        filled = n - empty
        if empty == n:
            state = "EMPTY - every row"
        elif empty == 0:
            state = "complete"
        elif str(c).strip().lower() in EXPECTED_SPARSE:
            state = "partial (expected)"
        else:
            state = f"PARTIAL - {empty} rows blank"
        out.append(f"{str(c):<44}{filled:>8}{empty:>8}  {state}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--mia-root", type=Path, default=None,
                    help="tree holding the harmonized MIA masks")
    ap.add_argument("--original-mia-root", type=Path, default=None,
                    help="tree holding the MIA files as the authors supplied "
                         "them; used to fill original_mia_name")
    ap.add_argument("--relative-to", type=Path, default=None,
                    help="write mask paths relative to this directory "
                         "(default: the manifest's parent's parent)")
    ap.add_argument("--out", type=Path, default=None,
                    help="write the filled manifest here")
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite --manifest after taking a .bak copy; "
                         "refuses without --i-know-this-is-locked if the path "
                         "is under release/")
    ap.add_argument("--i-know-this-is-locked", action="store_true")
    ap.add_argument("--unmatched-report", type=Path,
                    default=Path("mia_unmatched.csv"))
    args = ap.parse_args()

    if not args.manifest.is_file():
        print(f"error: manifest not found: {args.manifest}", file=sys.stderr)
        return 2

    df = pd.read_csv(args.manifest, dtype=str, keep_default_na=False)
    n = len(df)
    print(f"manifest: {args.manifest}  ({n} rows, {len(df.columns)} columns)\n")

    print("=" * 76)
    print("  COLUMN COMPLETENESS")
    print("=" * 76)
    for line in completeness_audit(df):
        print("  " + line)
    print()

    uid_col = find_uid_column(df)
    if uid_col is None:
        print("error: could not identify a UID column. Pass one explicitly by "
              "renaming it, or extend find_uid_column().", file=sys.stderr)
        return 2
    print(f"UID column: {uid_col!r}  (example: {df[uid_col].iloc[0]!r})\n")

    for col in (MIA_COL, ORIG_COL):
        if col not in df.columns:
            print(f"note: {col} is not in the manifest; it will be created.")
            df[col] = ""

    if args.mia_root is None:
        print("No --mia-root given, so nothing was filled. The audit above is "
              "the whole output.")
        return 0
    if not args.mia_root.is_dir():
        print(f"error: --mia-root not found: {args.mia_root}", file=sys.stderr)
        return 2

    rel_base = args.relative_to or args.manifest.parent.parent
    idx, n_files = index_masks(args.mia_root)
    stems = list(idx)
    print(f"indexed {n_files} mask files under {args.mia_root} "
          f"({len(stems)} distinct stems)")

    orig_idx, orig_stems, n_orig = {}, [], 0
    if args.original_mia_root and args.original_mia_root.is_dir():
        orig_idx, n_orig = index_masks(args.original_mia_root)
        orig_stems = list(orig_idx)
        print(f"indexed {n_orig} original files under {args.original_mia_root}")
    print()

    stats: dict[str, int] = defaultdict(int)
    unmatched: list[dict] = []
    mia_vals, orig_vals = [], []

    for _, row in df.iterrows():
        uid = str(row[uid_col]).strip()
        path, how = match_uid(uid, idx, stems)
        stats[how] += 1
        if path is None:
            mia_vals.append("")
            unmatched.append({uid_col: uid, "reason": how})
        else:
            try:
                mia_vals.append(str(path.resolve().relative_to(rel_base.resolve())))
            except ValueError:
                mia_vals.append(str(path))

        if orig_idx:
            opath, _ = match_uid(uid, orig_idx, orig_stems)
            orig_vals.append(opath.name if opath else "")
        else:
            orig_vals.append(str(row[ORIG_COL]))

    df[MIA_COL] = mia_vals
    df[ORIG_COL] = orig_vals

    filled = sum(1 for v in mia_vals if v)
    print("=" * 76)
    print("  MATCHING")
    print("=" * 76)
    for k in ("exact", "contained", "ambiguous", "ambiguous-exact", "missing"):
        if stats.get(k):
            print(f"  {k:<18}{stats[k]:>6}")
    print(f"  {'FILLED':<18}{filled:>6} of {n} rows "
          f"({100 * filled / n:.1f}%)")
    if orig_idx:
        print(f"  {'original names':<18}{sum(1 for v in orig_vals if v):>6}")
    print()

    if unmatched:
        pd.DataFrame(unmatched).to_csv(args.unmatched_report, index=False)
        print(f"  {len(unmatched)} rows unmatched -> {args.unmatched_report}")
        print("  Ambiguous rows are left empty on purpose. Resolve the "
              "duplicate stems rather than\n  letting the script pick one.")
        print()

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"written: {args.out}")
    elif args.in_place:
        locked = "release" in args.manifest.parts
        if locked and not args.i_know_this_is_locked:
            print("refusing: the manifest is under release/, which is a locked "
                  "artifact tree.\nWrite a new file with --out, or pass "
                  "--i-know-this-is-locked deliberately.", file=sys.stderr)
            return 2
        backup = args.manifest.with_suffix(args.manifest.suffix + ".bak")
        shutil.copy2(args.manifest, backup)
        df.to_csv(args.manifest, index=False)
        print(f"backup : {backup}")
        print(f"written: {args.manifest}  (IN PLACE, under a locked tree)")
    else:
        print("Dry run - nothing written. Add --out <path> to write the "
              "filled manifest.")

    return 0 if filled == n else 1


if __name__ == "__main__":
    raise SystemExit(main())
