#!/usr/bin/env python3
"""
scripts/rewrite_manifest_paths.py
=================================
Write a copy of the benchmark table with its file locations moved to a new
folder, for example after the data moved from scratch (/work) to the lab
project space (/proj).

The released table (release/manifests/dl_master_metadata_stage1_step3_consistency.csv)
is a locked artifact and is never modified: the copy goes elsewhere, and
configs/paths.local.yaml (paths.consistency_csv) points at the copy.

Only columns whose name ends in ``_fullpath`` are changed (or the columns given
with --columns), and only values that start with --old. Everything else,
including column order and row order, is kept.

Usage
-----
    python scripts/rewrite_manifest_paths.py \\
        --src release/manifests/dl_master_metadata_stage1_step3_consistency.csv \\
        --dst /proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/manifests/dl_master_metadata_stage1_step3_consistency_proj.csv \\
        --old /work/users/b/e/behnamie/ecDNA_Data/ \\
        --new /proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/

    python scripts/rewrite_manifest_paths.py --self-test

Exit status: 0 written and every rewritten path is readable, 1 some paths are
not readable (the copy is still written), 2 usage error or refused.
Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path
from typing import List, Optional


def rewrite(src: Path, dst: Path, old: str, new: str, columns: Optional[List[str]] = None,
            force: bool = False) -> int:
    src, dst = src.resolve(), dst.resolve()
    if src == dst:
        print("REFUSED: --dst must differ from --src (the released table is locked)")
        return 2
    if "release" in dst.parts and not force:
        print(f"REFUSED: {dst} is inside a release/ folder; choose another location (or --force)")
        return 2
    if not src.is_file():
        print(f"not found: {src}")
        return 2
    with open(src, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    targets = columns or [c for c in fields if c.endswith("_fullpath")]
    missing = [c for c in targets if c not in fields]
    if missing:
        print(f"REFUSED: columns not in the table: {missing}")
        return 2

    changed = unchanged = empty = unreadable = 0
    first_bad = ""
    for row in rows:
        for col in targets:
            value = row.get(col) or ""
            if not value:
                empty += 1
                continue
            if value.startswith(old):
                value = new + value[len(old):]
                row[col] = value
                changed += 1
            else:
                unchanged += 1
            if not os.access(value, os.R_OK):
                unreadable += 1
                first_bad = first_bad or value

    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, dst)
    try:
        os.chmod(dst, 0o644)
    except OSError:
        pass

    print(f"source      : {src} ({len(rows)} rows)")
    print(f"written     : {dst}")
    print(f"columns     : {', '.join(targets)}")
    print(f"rewritten   : {changed}   already elsewhere: {unchanged}   empty: {empty}")
    print(f"unreadable  : {unreadable}" + (f"   e.g. {first_bad}" if first_bad else ""))
    if unreadable:
        print("VERDICT: copy written, but some paths cannot be read by this account")
        return 1
    print("VERDICT: copy written; every path is readable")
    return 0


def self_test() -> int:
    import contextlib
    import io

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        new_root = tmp / "proj" / "ecDNA_Data"
        for sub, name in [("bioimage_archive/rgb", "a.tif"), ("bioimage_archive/gt_image", "a.tif")]:
            (new_root / sub).mkdir(parents=True, exist_ok=True)
            (new_root / sub / name).write_bytes(b"x")
        src = tmp / "release" / "manifests" / "table.csv"
        src.parent.mkdir(parents=True)
        src.write_text(
            "unique_id,rgb_fullpath,gt_fullpath,roi_fullpath,ecDNA_gt\n"
            "a,/work/u/ecDNA_Data/bioimage_archive/rgb/a.tif,/work/u/ecDNA_Data/bioimage_archive/gt_image/a.tif,,5\n",
            encoding="utf-8")
        dst = tmp / "proj" / "ecDNA_Data" / "manifests" / "table_proj.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            code = rewrite(src, dst, "/work/u/ecDNA_Data/", str(new_root) + "/")
        text = dst.read_text(encoding="utf-8")
        checks = [
            ("exit 0 when all readable", code == 0),
            ("paths rewritten", f"{new_root}/bioimage_archive/rgb/a.tif" in text),
            ("other columns kept", text.splitlines()[0] == "unique_id,rgb_fullpath,gt_fullpath,roi_fullpath,ecDNA_gt"
             and text.rstrip().endswith(",5")),
            ("source untouched", "/work/u/" in src.read_text(encoding="utf-8")),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            checks.append(("refuses dst == src", rewrite(src, src, "/a", "/b") == 2))
            checks.append(("refuses release/ destination",
                           rewrite(src, tmp / "release" / "x.csv", "/a", "/b") == 2))
            checks.append(("exit 1 when unreadable",
                           rewrite(src, tmp / "out" / "t.csv", "/work/u/ecDNA_Data/", "/nowhere/") == 1))
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok &= passed
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path)
    ap.add_argument("--dst", type=Path)
    ap.add_argument("--old", help="prefix to replace, e.g. /work/users/b/e/behnamie/ecDNA_Data/")
    ap.add_argument("--new", help="replacement prefix")
    ap.add_argument("--columns", nargs="*", help="columns to rewrite (default: every *_fullpath column)")
    ap.add_argument("--force", action="store_true", help="allow a destination inside release/")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not (args.src and args.dst and args.old and args.new):
        ap.print_usage()
        return 2
    return rewrite(args.src, args.dst, args.old, args.new, args.columns, args.force)


if __name__ == "__main__":
    sys.exit(main())
