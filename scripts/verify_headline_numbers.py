#!/usr/bin/env python3
"""
scripts/verify_headline_numbers.py
==================================
Compare a benchmark result folder with the published numbers.

Reads ``<results>/or_matching/summary_overall.csv`` (and
``summary_by_split.csv`` when present) and checks object-level F1, count
MAE and mean signed count bias for every model it finds, at the precision
used in the paper (F1 three decimals, MAE and bias one decimal).

The scope is taken from the number of images in the table:

* 1,145 images  — the full benchmark; the test-split rows of
  ``summary_by_split.csv`` are checked as well;
* 175 images    — the held-out test split only (for example after
  re-scoring the test images downloaded from the BioImage Archive);
* anything else — the table is printed without a comparison.

Usage
-----
    python scripts/verify_headline_numbers.py                      # release/frozen_results
    python scripts/verify_headline_numbers.py --results runs/bia_test/results

Exit status: 0 all compared values match, 1 at least one differs,
2 the results folder is missing or unreadable.
Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# (obj_f1, count_mae, count_bias) at paper precision.
# All-image values: locked list in CLAUDE_INSTRUCTIONS.md §3.
# Test-split values: F1 and MAE from the locked list; bias from
# release/frozen_results/or_matching/summary_by_split.csv.
EXPECTED_ALL: Dict[str, Tuple[float, float, float]] = {
    "ecCount (peaks)":          (0.942, 13.1, 0.4),
    "ecCount (threshold mask)": (0.917, 18.1, -11.4),
    "Label Engine":             (0.825, 36.0, -28.8),
    "MIA":                      (0.800, 47.0, -40.6),
    "Classic (after opt)":      (0.777, 44.9, -24.3),
    "ecSeg":                    (0.464, 121.5, -118.5),
}
EXPECTED_TEST: Dict[str, Tuple[float, float, float]] = {
    "ecCount (peaks)":          (0.939, 13.3, 1.9),
    "ecCount (threshold mask)": (0.916, 17.6, -9.1),
    "Label Engine":             (0.825, 34.4, -26.9),
    "MIA":                      (0.813, 40.4, -35.7),
    "Classic (after opt)":      (0.775, 40.9, -17.4),
    "ecSeg":                    (0.508, 110.3, -107.4),
}
N_ALL, N_TEST = 1145, 175


def _read(path: Path) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _fmt(x: float, nd: int, sign: bool = False) -> str:
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _compare(rows: List[Dict[str, str]], expected: Dict[str, Tuple[float, float, float]],
             title: str) -> Tuple[int, int]:
    print(f"\n{title}")
    header = f"  {'model':<26}{'n':>6}{'obj F1':>9}{'MAE':>8}{'bias':>9}   {'published':<24} check"
    print(header)
    print("  " + "-" * (len(header) - 2))
    checked = failed = 0
    for r in rows:
        model = r["model"]
        f1, mae, bias = float(r["obj_f1"]), float(r["count_mae"]), float(r["count_bias"])
        got = (round(f1, 3), round(mae, 1), round(bias, 1))
        exp = expected.get(model)
        if exp is None:
            verdict, pub = "not in the paper table", ""
        else:
            checked += 1
            ok = all(abs(a - b) < 1e-9 for a, b in zip(got, exp))
            failed += 0 if ok else 1
            verdict = "MATCH" if ok else "DIFFERS"
            pub = f"{exp[0]:.3f} / {_fmt(exp[1], 1)} / {_fmt(exp[2], 1, True)}"
        print(f"  {model:<26}{r.get('n_images', ''):>6}{_fmt(f1, 3):>9}{_fmt(mae, 1):>8}"
              f"{_fmt(bias, 1, True):>9}   {pub:<24} {verdict}")
    return checked, failed


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=Path("release/frozen_results"),
                    help="folder that contains or_matching/ (default: release/frozen_results)")
    ap.add_argument("--policy", default="or", choices=["or", "and"],
                    help="matching policy folder to read (the paper uses OR)")
    args = ap.parse_args(argv)

    folder = args.results / f"{args.policy}_matching"
    overall_p = folder / "summary_overall.csv"
    split_p = folder / "summary_by_split.csv"
    if not overall_p.is_file():
        print(f"not found: {overall_p}", file=sys.stderr)
        return 2
    try:
        overall = _read(overall_p)
    except Exception as exc:
        print(f"cannot read {overall_p}: {exc}", file=sys.stderr)
        return 2
    if not overall:
        print(f"{overall_p} has no rows", file=sys.stderr)
        return 2

    print(f"results : {folder}")
    if args.policy != "or":
        print("note    : the published numbers use OR matching; AND results are printed only")

    n_values = {int(float(r.get("n_images", 0) or 0)) for r in overall}
    n = max(n_values)
    checked = failed = 0
    if args.policy != "or":
        _compare(overall, {}, f"{args.policy.upper()} matching, {n} images")
    elif n == N_ALL:
        c, f = _compare(overall, EXPECTED_ALL, f"All {N_ALL:,} benchmark images (pooled, OR matching)")
        checked += c
        failed += f
        if split_p.is_file():
            test_rows = [r for r in _read(split_p) if r.get("split") == "test"]
            if test_rows:
                c, f = _compare(test_rows, EXPECTED_TEST,
                                f"Held-out test split, {N_TEST} images (pooled, OR matching)")
                checked += c
                failed += f
    elif n == N_TEST:
        c, f = _compare(overall, EXPECTED_TEST,
                        f"Held-out test split, {N_TEST} images (pooled, OR matching)")
        checked += c
        failed += f
    else:
        _compare(overall, {}, f"{n} images (not a published scope; printed without comparison)")

    print()
    if checked == 0:
        print("VERDICT: nothing compared (scope or models not in the published tables)")
        return 0
    if failed:
        print(f"VERDICT: {failed} of {checked} model rows DIFFER from the published values")
        return 1
    print(f"VERDICT: all {checked} model rows MATCH the published values")
    return 0


if __name__ == "__main__":
    sys.exit(main())
