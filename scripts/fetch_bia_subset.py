#!/usr/bin/env python3
"""
scripts/fetch_bia_subset.py
===========================
Download all or part of the ecdna-bench imaging resource from the
BioImage Archive (accession S-BIAD4097) over HTTPS.

The archive publishes one file list per section (``filelist_images.tsv``,
``filelist_gt.tsv``, ``filelist_roi.tsv``, ``filelist_predictions.tsv``,
``filelist_predicted_roi.tsv``).  This script reads those lists, selects the
rows you ask for, and downloads the files into a local folder with the same
layout as the archive (``images/``, ``predictions/``, ``predicted_roi/``,
``splits/``).  Interrupted downloads resume: a file whose size already
matches the server is skipped.

Typical selections
------------------
    # What would be downloaded?  (reads the file lists only)
    python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test --dry-run

    # Re-score the published predictions on the 175 test images
    # (gold-standard masks, manual ROI masks and all prediction masks; no images)
    python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
        --sections gt,roi,predictions

    # Run ecCount yourself on the test images (adds the RGB images)
    python scripts/fetch_bia_subset.py --out ~/ecdna_data --split test \
        --sections images,gt,roi --image-types rgb

    # Everything (about 55 GB)
    python scripts/fetch_bia_subset.py --out ~/ecdna_data --all

Where the files come from
-------------------------
``--base-url`` (or ``$ECDNA_BIA_BASE_URL``) is either the URL of the study's
``Files`` folder or the study's share link
(``https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4097?key=...``).
For a share link, the script asks the BioStudies API for the current file
location of the private record; that location can change, the share link
does not.  Without either, the script asks the API for the study's public
location, then falls back to
``https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/097/S-BIAD4097/Files``.

The record is private until its release date.  Reviewers pass the share link
from the reviewer instructions with ``--base-url`` (or ``ECDNA_BIA_BASE_URL``).
It contains an access key: never commit it or paste it into a public issue.
The script never prints it in full.

Only the Python standard library is used.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ACCESSION = "S-BIAD4097"
INFO_API = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{ACCESSION}/info"
FALLBACK_BASE = f"https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/097/{ACCESSION}/Files"
SECTIONS = {
    "images": "filelist_images.tsv",
    "gt": "filelist_gt.tsv",
    "roi": "filelist_roi.tsv",
    "predictions": "filelist_predictions.tsv",
    "predicted_roi": "filelist_predicted_roi.tsv",
}
USER_AGENT = "ecdna-bench-fetch/1.0 (+https://github.com/Brunk-Lab/ecdna-bench)"


# ----------------------------------------------------------------------------
# URL handling
# ----------------------------------------------------------------------------

def mask_url(url: str) -> str:
    """Hide the access key of a private BioStudies location."""
    url = re.sub(r"(/\.private/\d+/)[^/]+", r"\1<key>", url)
    return re.sub(r"([?&](?:key|accessKey|token)=)[^&#/]+", r"\1<key>", url, flags=re.I)


def http_get(url: str, timeout: float = 60.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def remote_size(url: str, timeout: float = 30.0) -> Optional[int]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length is not None else None
    except (urllib.error.URLError, ValueError, OSError):
        return None


def share_link_key(url: str) -> Optional[str]:
    """Return the access key if ``url`` is a BioStudies study page with ?key=."""
    parts = urllib.parse.urlsplit(url)
    if "/studies/" not in parts.path:
        return None
    keys = urllib.parse.parse_qs(parts.query).get("key")
    return keys[0] if keys and keys[0] else None


def files_url_from_share_link(key: str) -> str:
    """Ask the BioStudies API where the private record's files are now."""
    url = f"{INFO_API}?key={urllib.parse.quote(key)}"
    try:
        info = json.loads(http_get(url, timeout=30).decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"The BioStudies API refused the share link (HTTP {exc.code}). "
                         "Check that the whole link was copied.")
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise SystemExit(f"Could not ask the BioStudies API about the share link: "
                         f"{exc.__class__.__name__}")
    link = info.get("httpLink") if isinstance(info, dict) else None
    if not link or not str(link).startswith(("http://", "https://")):
        raise SystemExit("The BioStudies API did not return a file location for this share link.")
    link = str(link).rstrip("/")
    return link if link.endswith("/Files") else link + "/Files"


def _given_location(value: str, origin: str) -> str:
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        raise SystemExit(f"{origin} is not a web address (it must start with https://). "
                         "Use the share link or the study's Files folder.")
    key = share_link_key(value)
    if key:
        return files_url_from_share_link(key)
    return value.rstrip("/")


def resolve_base_url(explicit: Optional[str]) -> str:
    if explicit:
        return _given_location(explicit, "--base-url")
    env = os.environ.get("ECDNA_BIA_BASE_URL")
    if env:
        return _given_location(env, "ECDNA_BIA_BASE_URL")
    try:
        info = json.loads(http_get(INFO_API, timeout=20).decode("utf-8"))
        link = info.get("httpLink") or info.get("ftpHttp_link")
        if link:
            link = link.rstrip("/")
            return link if link.endswith("/Files") else link + "/Files"
    except Exception:
        pass
    return FALLBACK_BASE


# ----------------------------------------------------------------------------
# File lists
# ----------------------------------------------------------------------------

@dataclass
class Row:
    section: str
    path: str
    uid: str
    fields: Dict[str, str]


def _uid_from(path: str) -> str:
    return Path(path).stem


def _norm_line(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def fetch_filelist(base: str, name: str, out: Path, refresh: bool) -> Optional[Path]:
    dest = out / name
    if dest.is_file() and dest.stat().st_size > 0 and not refresh:
        return dest
    url = f"{base}/{urllib.parse.quote(name)}"
    try:
        data = http_get(url)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest


def read_filelist(path: Path, section: str) -> List[Row]:
    text = path.read_text(encoding="utf-8-sig")
    rows = []
    for rec in csv.DictReader(io.StringIO(text), delimiter="\t"):
        p = (rec.get("Files") or "").strip()
        if not p:
            continue
        src = (rec.get("source_image") or "").strip()
        uid = _uid_from(src) if src else _uid_from(p)
        rows.append(Row(section, p, uid, rec))
    return rows


def classify_image(row: Row) -> Optional[str]:
    kind = row.fields.get("Data Type", "")
    if "RGB" in kind:
        return "rgb"
    if "DAPI" in kind:
        return "dapi"
    return None  # supporting file (index table, split lists)


def classify_gt(row: Row) -> str:
    kind = row.fields.get("Annotation Type", "").lower()
    if "point" in kind:
        return "points"
    if "sparse" in kind or row.path.endswith(".npz"):
        return "sparse"
    return "mask"


METHOD_FOLDERS = {
    "eccount_peaks": "ecCount (peaks)",
    "eccount_threshold": "ecCount (threshold mask)",
    "label_engine": "Label Engine",
    "mia": "MIA",
    "classical_optimised": "Classic (after opt)",
    "ecseg": "ecSeg",
    "classical_default": "Classic (before opt)",
}


def method_matches(row: Row, wanted: Set[str]) -> bool:
    if not wanted:
        return True
    method = row.fields.get("Method", "")
    parts = Path(row.path).parts
    folder = parts[1] if len(parts) > 2 else ""
    names = {method.lower(), folder.lower(), METHOD_FOLDERS.get(folder, "").lower()}
    return bool(names & wanted)


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------

def select(rows_by_section: Dict[str, List[Row]], args) -> Tuple[List[Row], Set[str]]:
    images = rows_by_section.get("images", [])
    if not images:
        raise SystemExit("filelist_images.tsv could not be read; nothing to select from.")

    wanted_lines = {_norm_line(c) for c in args.cell_line} if args.cell_line else set()
    explicit: Set[str] = set()
    if args.uids_file:
        for line in Path(args.uids_file).read_text().splitlines():
            tok = line.strip().split(",")[0].strip()
            if tok and tok != "unique_id":
                explicit.add(tok)

    image_rows = [r for r in images if classify_image(r) is not None]
    chosen: Dict[str, List[Row]] = {}
    for r in image_rows:
        f = r.fields
        if not args.all:
            if args.split != "any" and f.get("Split", "") != args.split:
                continue
            if args.subset != "any" and f.get("Subset", "") != args.subset:
                continue
        if wanted_lines and _norm_line(f.get("Cell Line", "")) not in wanted_lines:
            continue
        if explicit and r.uid not in explicit:
            continue
        chosen.setdefault(f.get("Cell Line", ""), []).append(r)

    uids: Set[str] = set()
    for line, rs in chosen.items():
        line_uids = sorted({r.uid for r in rs})
        if args.max_images:
            line_uids = line_uids[: args.max_images]
        uids.update(line_uids)

    sections = set(args.sections)
    image_types = set(args.image_types)
    gt_types = set(args.gt_types)
    methods = {m.lower() for m in args.methods} if args.methods else set()

    selected: List[Row] = []
    # Supporting files (index table, split lists) are small and always useful.
    selected.extend(r for r in images if classify_image(r) is None)
    for sec, rows in rows_by_section.items():
        if sec not in sections:
            continue
        for r in rows:
            if r.uid not in uids:
                continue
            if sec == "images" and classify_image(r) not in image_types:
                continue
            if sec == "gt" and classify_gt(r) not in gt_types:
                continue
            if sec == "predictions" and not method_matches(r, methods):
                continue
            selected.append(r)
    # De-duplicate, keep order.
    seen: Set[str] = set()
    unique = []
    for r in selected:
        if r.path not in seen:
            seen.add(r.path)
            unique.append(r)
    return unique, uids


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------

def download_one(base: str, rel: str, out: Path, retries: int, verify_size: bool) -> Tuple[str, str, int]:
    dest = out / rel
    url = f"{base}/{urllib.parse.quote(rel)}"
    size = remote_size(url) if verify_size else None
    if dest.is_file():
        have = dest.stat().st_size
        if (not verify_size and have > 0) or (size is not None and have == size):
            return rel, "present", have
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    last_err = ""
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=120) as resp, open(part, "wb") as fh:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    fh.write(chunk)
            got = part.stat().st_size
            if size is not None and got != size:
                raise IOError(f"size {got} != expected {size}")
            os.replace(part, dest)
            return rel, "downloaded", got
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            if exc.code in (401, 403, 404):
                break
        except Exception as exc:  # network hiccup: retry with back-off
            last_err = f"{exc.__class__.__name__}: {exc}"
        time.sleep(min(30, 2 ** attempt))
    if part.exists():
        part.unlink()
    return rel, f"FAILED ({last_err})", 0


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True,
                    help="local folder; files keep their archive paths below it")
    ap.add_argument("--base-url", default=None,
                    help="study share link (reviewers) or URL of the study's Files folder")
    ap.add_argument("--all", action="store_true", help="every image set (ignores --split/--subset)")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "any"],
                    help="benchmark split to take (default: test)")
    ap.add_argument("--subset", default="benchmark", choices=["benchmark", "extension", "any"],
                    help="benchmark = the 1,145 images with manual ROI masks (default)")
    ap.add_argument("--cell-line", action="append", default=[],
                    help="restrict to a cell line, e.g. NCI-H2170 (repeatable)")
    ap.add_argument("--uids-file", default=None, help="text/CSV file of unique_id values")
    ap.add_argument("--max-images", type=int, default=0,
                    help="at most this many image sets per cell line (0 = no limit)")
    ap.add_argument("--sections", default="images,gt,roi,predictions",
                    help="comma list from: " + ",".join(SECTIONS))
    ap.add_argument("--image-types", default="rgb,dapi", help="comma list from: rgb,dapi")
    ap.add_argument("--gt-types", default="mask",
                    help="comma list from: mask,points,sparse (gold-standard representations)")
    ap.add_argument("--methods", default="",
                    help="comma list of prediction folders or method names (default: all)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--no-verify-size", action="store_true",
                    help="skip the per-file size check (faster, less safe)")
    ap.add_argument("--refresh-lists", action="store_true", help="re-download the file lists")
    ap.add_argument("--dry-run", action="store_true", help="list what would be downloaded")
    args = ap.parse_args(argv)

    args.sections = [s.strip() for s in args.sections.split(",") if s.strip()]
    bad = [s for s in args.sections if s not in SECTIONS]
    if bad:
        ap.error(f"unknown section(s): {bad}")
    args.image_types = [s.strip() for s in args.image_types.split(",") if s.strip()]
    args.gt_types = [s.strip() for s in args.gt_types.split(",") if s.strip()]
    args.methods = [s.strip() for s in args.methods.split(",") if s.strip()]

    out = args.out.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = resolve_base_url(args.base_url)
    if "?" in base or "#" in base:
        print(f"--base-url looks like a web-page link ({mask_url(base)}).\n"
              "Use the study's share link (.../studies/S-BIAD4097?key=...) or the URL of "
              "the study's Files folder, which ends in /Files.",
              file=sys.stderr)
        return 2
    print(f"source : {mask_url(base)}")
    print(f"target : {out}")

    rows_by_section: Dict[str, List[Row]] = {}
    needed = set(args.sections) | {"images"}
    for sec in needed:
        try:
            p = fetch_filelist(base, SECTIONS[sec], out, args.refresh_lists)
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                print(f"\nAccess denied ({exc.code}) for {SECTIONS[sec]}. The record may still be "
                      "private: use the reviewer download location with --base-url.",
                      file=sys.stderr)
                return 2
            raise
        except urllib.error.URLError as exc:
            print(f"\nCannot reach the archive: {exc.reason}", file=sys.stderr)
            return 2
        if p is None:
            print(f"  note: {SECTIONS[sec]} not found on the server; section skipped")
            continue
        rows_by_section[sec] = read_filelist(p, sec)

    if "images" not in rows_by_section:
        print(f"\nfilelist_images.tsv was not found under {mask_url(base)}.\n"
              "Check --base-url: it must be the study's Files folder (the URL that ends in /Files).",
              file=sys.stderr)
        return 2
    selected, uids = select(rows_by_section, args)
    by_section: Dict[str, int] = {}
    for r in selected:
        by_section[r.section] = by_section.get(r.section, 0) + 1
    print(f"image sets selected : {len(uids)}")
    for sec, n in sorted(by_section.items()):
        print(f"  {sec:<14} {n:>6} files")
    print(f"  {'total':<14} {len(selected):>6} files")

    if args.dry_run:
        for r in selected[:15]:
            print(f"    {r.path}")
        if len(selected) > 15:
            print(f"    ... {len(selected) - 15} more")
        return 0

    log_path = out / "download_log.tsv"
    failures = 0
    done_bytes = 0
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool, \
            open(log_path, "a", encoding="utf-8") as log:
        futs = [pool.submit(download_one, base, r.path, out, args.retries,
                            not args.no_verify_size) for r in selected]
        for i, fut in enumerate(as_completed(futs), start=1):
            rel, status, nbytes = fut.result()
            done_bytes += nbytes
            log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{rel}\t{status}\t{nbytes}\n")
            if status.startswith("FAILED"):
                failures += 1
                print(f"  {status}: {rel}")
            if i % 200 == 0 or i == len(futs):
                rate = done_bytes / max(1.0, time.time() - started) / 1e6
                print(f"  {i}/{len(futs)} files, {done_bytes / 1e9:.2f} GB, {rate:.1f} MB/s")
    print(f"log    : {log_path}")
    if failures:
        print(f"VERDICT: {failures} file(s) failed; re-run the same command to retry them")
        return 1
    print("VERDICT: all selected files are present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
