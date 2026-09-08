#!/usr/bin/env python3
"""
scripts/verify_bia_upload.py
============================
Verify that what landed on the BioStudies FTP server matches what the file
lists claim.

This is the check EBI's validator will effectively perform, run before
submission rather than after. It answers three questions:

  1. Does every path named in the file lists exist on the server?
     A path in a file list with no file behind it fails validation.

  2. Does every file on the server appear in exactly one file list?
     An uploaded file that no list mentions is undescribed data. It does not
     block submission, but part of the deposition would arrive without
     metadata.

  3. Do the contents match, not just the names?
     A file truncated by a dropped connection has the right name and the
     wrong size. This is checked by running the upload's own `mirror` in
     --dry-run mode: if mirror would re-send nothing, every remote file
     matches its local counterpart in size and modification time.

Credentials are read from ~/.bia_ftp_login, exactly as the upload job does.

Note on failure modes: an empty listing is treated as an error, never as
"the server is empty". An earlier version of this script used an invalid
lftp option, got nothing on stdout, and reported all 24,094 files missing.

Usage
-----
    python scripts/verify_bia_upload.py \
        --file-lists release/deposition/file_lists \
        --source images=/proj/.../ecDNA_Data/bioimage_archive \
        --source predictions=/proj/.../ecDNA_Data/benchmark/predictions \
        --source splits=/proj/.../ecdna-bench/release/split_files

Omit --source entirely to run the name-only check.
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

LOGIN = Path.home() / ".bia_ftp_login"


def run_lftp(commands: list[str]) -> tuple[str, str, int]:
    """Run lftp with the saved login plus `commands`. Returns (out, err, rc)."""
    if not LOGIN.is_file():
        sys.exit(f"FATAL: {LOGIN} not found.")
    if oct(LOGIN.stat().st_mode)[-3:] != "600":
        sys.exit(f"FATAL: {LOGIN} must be mode 600.")

    old = os.umask(0o077)
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".lftp", delete=False) as f:
            f.write(LOGIN.read_text())
            f.write("\n" + "\n".join(commands) + "\nbye\n")
            script = f.name
    finally:
        os.umask(old)
    try:
        p = subprocess.run(["lftp", "-f", script], capture_output=True, text=True)
    finally:
        os.unlink(script)
    return p.stdout, p.stderr, p.returncode


def remote_listing() -> set[str]:
    """Every file on the server, as paths relative to the submission root."""
    out, err, rc = run_lftp(["find"])
    files = set()
    for line in out.splitlines():
        p = line.strip()
        if not p or p.endswith("/"):
            continue
        while p.startswith("./"):
            p = p[2:]
        files.add(p)
    if not files:
        # Never interpret silence as an empty server.
        print("FATAL: the server listing came back empty.", file=sys.stderr)
        print(f"lftp exit status: {rc}", file=sys.stderr)
        if err.strip():
            print("lftp stderr:", file=sys.stderr)
            print(err.strip(), file=sys.stderr)
        else:
            print("lftp produced no error output either; check the login file "
                  "contains its 'cd /<secret directory>' line.", file=sys.stderr)
        sys.exit(2)
    return files


def dry_run_mirror(sources: list[tuple[str, Path]]) -> list[str]:
    """
    Ask mirror what it would still transfer. Anything it names differs in size
    or modification time between local and remote, or is absent remotely.
    """
    cmds = []
    for label, path in sources:
        cmds.append(f"mirror -R --only-newer --no-perms --dry-run "
                    f"--verbose {path} {label}")
    out, err, rc = run_lftp(cmds)
    pending = []
    for line in (out + "\n" + err).splitlines():
        s = line.strip()
        if s.startswith("Transferring file"):
            pending.append(s)
    return pending


def read_file_lists(d: Path) -> dict[str, str]:
    """{path: declaring list}, refusing a path claimed by two sections."""
    declared: dict[str, str] = {}
    lists = sorted(d.glob("filelist_*.tsv"))
    if not lists:
        sys.exit(f"FATAL: no filelist_*.tsv found in {d}")
    for p in lists:
        with open(p, newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                path = row["Files"]
                if path in declared:
                    sys.exit(f"FATAL: {path} appears in both "
                             f"{declared[path]} and {p.name}")
                declared[path] = p.name
    return declared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file-lists", type=Path, required=True)
    ap.add_argument("--source", action="append", default=[], metavar="LABEL=PATH",
                    help="Upload source, as given to the upload job. Repeatable. "
                         "Omit to skip the content check.")
    args = ap.parse_args()

    sources: list[tuple[str, Path]] = []
    for spec in args.source:
        label, _, path = spec.partition("=")
        sources.append((label.strip(), Path(path.strip())))

    declared = read_file_lists(args.file_lists)
    print(f"file lists declare {len(declared)} files")

    print("listing the server...")
    remote = remote_listing()
    print(f"server holds       {len(remote)} files")

    missing = sorted(set(declared) - remote)
    undescribed = sorted(remote - set(declared))
    problems = 0

    print("\n=== DECLARED BUT NOT ON THE SERVER ===")
    if missing:
        problems += len(missing)
        print(f"{len(missing)} file(s) - these WILL fail validation")
        by_list: dict[str, int] = {}
        for m in missing:
            by_list[declared[m]] = by_list.get(declared[m], 0) + 1
        for name, n in sorted(by_list.items()):
            print(f"  {name}: {n}")
        for m in missing[:15]:
            print(f"  {m}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")
    else:
        print("none - every declared path exists on the server")

    print("\n=== ON THE SERVER BUT NOT DESCRIBED ===")
    if undescribed:
        print(f"{len(undescribed)} file(s) - would upload without metadata")
        for u in undescribed[:15]:
            print(f"  {u}")
        if len(undescribed) > 15:
            print(f"  ... and {len(undescribed) - 15} more")
        print("  (the file lists themselves are expected here once uploaded)")
    else:
        print("none - every uploaded file is described")

    print("\n=== CONTENT (dry-run mirror) ===")
    if not sources:
        print("skipped (no --source given)")
    else:
        pending = dry_run_mirror(sources)
        if pending:
            problems += len(pending)
            print(f"{len(pending)} file(s) differ from local or are absent:")
            for p in pending[:15]:
                print(f"  {p}")
            if len(pending) > 15:
                print(f"  ... and {len(pending) - 15} more")
            print("  Resubmit the upload job; --continue repairs partial files.")
        else:
            print("clean - mirror would re-send nothing")

    print("\n=== VERDICT ===")
    if problems:
        print(f"NOT READY TO SUBMIT - {problems} problem(s) above")
        sys.exit(1)
    print("READY TO SUBMIT")
    print("Remaining step: upload the file lists to the ROOT of the secret")
    print("directory, beside images/ and predictions/, then fill the web form.")


if __name__ == "__main__":
    main()