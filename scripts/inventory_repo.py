#!/usr/bin/env python3
"""
inventory_repo.py
=================
Walk a project directory and produce a complete, human-readable inventory of
every file and folder, so you can decide what belongs on GitHub and what does
not.

It uses ONLY the Python standard library, so it runs in any environment with no
installs needed.

Outputs (written to --out-dir, default ./repo_inventory):
  1. inventory_summary.md     - the readable summary (send this back to refine)
  2. inventory_full.csv       - one row per file; the complete record
  3. inventory_by_dir.csv     - one row per directory, with aggregated stats

Typical use on Longleaf
-----------------------
    cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench
    python inventory_repo.py --root . --out-dir repo_inventory

Add --hash only if you also want SHA256 checksums of every file (slow on the
~30 GB of PNGs; you already have generate_manifest.py for the release manifest,
so you usually do NOT need this here):
    python inventory_repo.py --root . --out-dir repo_inventory --hash
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

# Directories skipped by default (noise / not useful in an inventory).
# Pass --include-all to walk these too.
IGNORE_DIRS = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".pytest_cache",
    ".mypy_cache",
    ".cache",
}

# Map file extension -> coarse category. Anything unlisted -> "other".
EXT_CATEGORY = {
    # code / scripts
    ".py": "code", ".sh": "code", ".ipynb": "notebook",
    ".pyx": "code", ".c": "code", ".cpp": "code", ".h": "code",
    # config
    ".yaml": "config", ".yml": "config", ".toml": "config",
    ".cfg": "config", ".ini": "config", ".json": "config",
    # docs / text
    ".md": "doc", ".txt": "doc", ".rst": "doc", ".pdf": "doc",
    ".docx": "doc", ".doc": "doc", ".cff": "doc", ".bib": "doc",
    # tabular data
    ".csv": "table", ".tsv": "table", ".parquet": "table", ".xlsx": "table",
    # images / figures
    ".png": "image", ".jpg": "image", ".jpeg": "image", ".tif": "image",
    ".tiff": "image", ".gif": "image", ".bmp": "image", ".webp": "image",
    ".svg": "vector",
    # arrays / models (large binaries)
    ".npy": "array", ".npz": "array", ".pth": "model", ".pt": "model",
    ".ckpt": "model", ".h5": "model", ".pkl": "model", ".joblib": "model",
    # archives
    ".zip": "archive", ".tar": "archive", ".gz": "archive",
    ".tgz": "archive", ".bz2": "archive", ".7z": "archive",
}

# Categories that almost certainly should NOT be committed to GitHub
# (large binaries / raw data -> BioImage Archive instead).
NON_GITHUB_CATEGORIES = {"image", "array", "model", "archive"}

# Text-like extensions for which we count lines (skipped if file > 2 MB).
TEXT_LINE_EXTS = {
    ".py", ".sh", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".json", ".md", ".txt", ".rst", ".cff", ".bib", ".csv", ".tsv",
}

LARGE_FILE_MB_DEFAULT = 5.0   # files at/above this size are flagged individually
MAX_LISTED_PER_DIR = 0        # 0 = don't inline file lists in the MD (CSV has all)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def human_size(n: int) -> str:
    """Bytes -> human-readable string."""
    step = 1024.0
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < step or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= step
    return f"{n:.1f} TB"


def category_for(path: Path) -> str:
    return EXT_CATEGORY.get(path.suffix.lower(), "other")


def count_lines(path: Path) -> int | None:
    """Count newlines in a text file; return None if not applicable/failed."""
    if path.suffix.lower() not in TEXT_LINE_EXTS:
        return None
    try:
        if path.stat().st_size > 2 * 1024 * 1024:  # skip >2 MB
            return None
        with open(path, "rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def sha256_file(path: Path, chunk: int = 1 << 20) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while block := f.read(chunk):
                h.update(block)
        return h.hexdigest()
    except OSError:
        return None


# --------------------------------------------------------------------------- #
# Core walk
# --------------------------------------------------------------------------- #

def walk_tree(root: Path, include_all: bool, do_hash: bool):
    """Return (file_rows, dir_stats, errors)."""
    file_rows = []          # list of dicts, one per file
    dir_stats = {}          # rel_dir -> {n_files, n_subdirs, size, ext_counts}
    errors = []

    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        if not include_all:
            dirnames[:] = [d for d in dirnames if d not in IGNORE_DIRS]
        dirnames.sort()
        filenames.sort()

        dp = Path(dirpath)
        rel_dir = "." if dp == root else str(dp.relative_to(root))
        depth = 0 if rel_dir == "." else rel_dir.count(os.sep) + 1

        stat = {
            "rel_dir": rel_dir,
            "depth": depth,
            "n_files": 0,
            "n_subdirs": len(dirnames),
            "size": 0,
            "ext_counts": defaultdict(int),
        }

        for name in filenames:
            fpath = dp / name
            try:
                st = fpath.stat()
                size = st.st_size
                mtime = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d")
            except OSError as exc:
                errors.append(f"{fpath}: {exc}")
                continue

            ext = fpath.suffix.lower()
            cat = category_for(fpath)
            rel_path = "." if fpath == root else str(fpath.relative_to(root))

            row = {
                "relative_path": rel_path,
                "name": name,
                "ext": ext or "(none)",
                "category": cat,
                "size_bytes": size,
                "size_human": human_size(size),
                "depth": depth + 1,
                "modified": mtime,
                "lines": count_lines(fpath),
                "github_candidate": cat not in NON_GITHUB_CATEGORIES,
            }
            if do_hash:
                row["sha256"] = sha256_file(fpath)
            file_rows.append(row)

            stat["n_files"] += 1
            stat["size"] += size
            stat["ext_counts"][ext or "(none)"] += 1

        dir_stats[rel_dir] = stat

    return file_rows, dir_stats, errors


# --------------------------------------------------------------------------- #
# Report writers
# --------------------------------------------------------------------------- #

def write_full_csv(file_rows, out_path: Path, do_hash: bool):
    fields = ["relative_path", "name", "ext", "category", "size_bytes",
              "size_human", "depth", "modified", "lines", "github_candidate"]
    if do_hash:
        fields.append("sha256")
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in file_rows:
            w.writerow({k: r.get(k, "") for k in fields})


def write_dir_csv(dir_stats, out_path: Path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["directory", "depth", "n_files", "n_subdirs",
                    "total_size_bytes", "total_size_human", "top_extensions"])
        for rel_dir in sorted(dir_stats):
            s = dir_stats[rel_dir]
            top = sorted(s["ext_counts"].items(), key=lambda kv: -kv[1])[:5]
            top_str = ", ".join(f"{e}:{c}" for e, c in top)
            w.writerow([rel_dir, s["depth"], s["n_files"], s["n_subdirs"],
                        s["size"], human_size(s["size"]), top_str])


def write_summary_md(file_rows, dir_stats, errors, root: Path,
                     out_path: Path, large_mb: float, include_all: bool):
    total_files = len(file_rows)
    total_size = sum(r["size_bytes"] for r in file_rows)

    # by category
    cat_files = defaultdict(int)
    cat_size = defaultdict(int)
    for r in file_rows:
        cat_files[r["category"]] += 1
        cat_size[r["category"]] += r["size_bytes"]

    # by extension
    ext_files = defaultdict(int)
    ext_size = defaultdict(int)
    for r in file_rows:
        ext_files[r["ext"]] += 1
        ext_size[r["ext"]] += r["size_bytes"]

    github_size = sum(r["size_bytes"] for r in file_rows if r["github_candidate"])
    non_github_size = total_size - github_size
    non_github_n = sum(1 for r in file_rows if not r["github_candidate"])

    large_bytes = large_mb * 1024 * 1024
    large_files = sorted(
        (r for r in file_rows if r["size_bytes"] >= large_bytes),
        key=lambda r: -r["size_bytes"],
    )

    root_files = [r for r in file_rows if r["depth"] == 1]

    lines = []
    A = lines.append
    A(f"# Repository inventory — `{root}`")
    A("")
    A(f"_Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} by inventory_repo.py_")
    if not include_all:
        A(f"_(skipped noise dirs: {', '.join(sorted(IGNORE_DIRS))})_")
    A("")
    A("## 1. Headline")
    A("")
    A(f"- **Total files:** {total_files:,}")
    A(f"- **Total directories:** {len(dir_stats):,}")
    A(f"- **Total size:** {human_size(total_size)}")
    A(f"- **GitHub-candidate size** (code/docs/configs/tables/vectors): "
      f"**{human_size(github_size)}**")
    A(f"- **Likely NOT-for-GitHub size** (images/arrays/models/archives): "
      f"**{human_size(non_github_size)}** across {non_github_n:,} files "
      f"→ these belong on the BioImage Archive, not Git.")
    A("")

    A("## 2. Files by category")
    A("")
    A("| Category | Files | Total size | GitHub? |")
    A("|---|---:|---:|:--:|")
    for cat in sorted(cat_files, key=lambda c: -cat_size[c]):
        gh = "no" if cat in NON_GITHUB_CATEGORIES else "yes"
        A(f"| {cat} | {cat_files[cat]:,} | {human_size(cat_size[cat])} | {gh} |")
    A("")

    A("## 3. Files by extension")
    A("")
    A("| Extension | Files | Total size |")
    A("|---|---:|---:|")
    for ext in sorted(ext_files, key=lambda e: -ext_size[e]):
        A(f"| `{ext}` | {ext_files[ext]:,} | {human_size(ext_size[ext])} |")
    A("")

    A(f"## 4. Large files (≥ {large_mb:g} MB)")
    A("")
    if large_files:
        A("| Size | Category | Path |")
        A("|---:|---|---|")
        for r in large_files[:60]:
            A(f"| {r['size_human']} | {r['category']} | `{r['relative_path']}` |")
        if len(large_files) > 60:
            A(f"\n_…and {len(large_files) - 60:,} more — see inventory_full.csv._")
    else:
        A("_None._")
    A("")

    A("## 5. Loose files at the repository root")
    A("")
    A("_(Files sitting directly in the root, not in a subfolder — usually the "
      "first cleanup target before committing.)_")
    A("")
    if root_files:
        A("| Size | Category | File |")
        A("|---:|---|---|")
        for r in sorted(root_files, key=lambda r: (r["category"], r["name"])):
            A(f"| {r['size_human']} | {r['category']} | `{r['name']}` |")
    else:
        A("_None._")
    A("")

    A("## 6. Directory tree (with aggregated stats)")
    A("")
    A("_Size shown per directory is the size of files directly in it "
      "(not including subfolders)._")
    A("")
    A("```")
    for rel_dir in sorted(dir_stats):
        s = dir_stats[rel_dir]
        indent = "  " * s["depth"]
        name = "." if rel_dir == "." else Path(rel_dir).name + "/"
        top = sorted(s["ext_counts"].items(), key=lambda kv: -kv[1])[:4]
        top_str = " ".join(f"{e}×{c}" for e, c in top)
        A(f"{indent}{name:<28} {s['n_files']:>4} files  "
          f"{human_size(s['size']):>9}   {top_str}")
    A("```")
    A("")

    if errors:
        A("## 7. Read errors")
        A("")
        A(f"{len(errors)} path(s) could not be read:")
        A("")
        A("```")
        for e in errors[:50]:
            A(e)
        if len(errors) > 50:
            A(f"... and {len(errors) - 50} more")
        A("```")
        A("")

    A("---")
    A("_Full per-file record: `inventory_full.csv`. "
      "Per-directory aggregates: `inventory_by_dir.csv`._")

    out_path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Inventory a project directory.")
    ap.add_argument("--root", type=Path,
                    default=Path("/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench"),
                    help="Directory to inventory (default: the ecdna-bench repo).")
    ap.add_argument("--out-dir", type=Path, default=Path("repo_inventory"),
                    help="Where to write the report files.")
    ap.add_argument("--hash", action="store_true",
                    help="Also compute SHA256 per file (SLOW on large data).")
    ap.add_argument("--include-all", action="store_true",
                    help="Do not skip .git/__pycache__/etc.")
    ap.add_argument("--large-mb", type=float, default=LARGE_FILE_MB_DEFAULT,
                    help="Flag files at/above this many MB (default: 5).")
    args = ap.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        sys.exit(f"ERROR: not a directory: {root}")

    print(f"Walking {root} ...", file=sys.stderr)
    if args.hash:
        print("  (hashing enabled — this can take a long time)", file=sys.stderr)

    file_rows, dir_stats, errors = walk_tree(root, args.include_all, args.hash)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    write_full_csv(file_rows, out_dir / "inventory_full.csv", args.hash)
    write_dir_csv(dir_stats, out_dir / "inventory_by_dir.csv")
    write_summary_md(file_rows, dir_stats, errors, root,
                     out_dir / "inventory_summary.md", args.large_mb,
                     args.include_all)

    total_size = sum(r["size_bytes"] for r in file_rows)
    print(f"\nDone. {len(file_rows):,} files, {human_size(total_size)} total.",
          file=sys.stderr)
    print(f"Wrote:\n  {out_dir/'inventory_summary.md'}\n"
          f"  {out_dir/'inventory_full.csv'}\n"
          f"  {out_dir/'inventory_by_dir.csv'}", file=sys.stderr)


if __name__ == "__main__":
    main()