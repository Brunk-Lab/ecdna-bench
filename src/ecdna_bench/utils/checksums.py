"""
ecdna_bench.utils.checksums — SHA-256 helpers for release manifests.

The dataset release and the committed prediction-mask bundles in
`release/baseline_predictions/` and `release/frozen_results/` each ship a
`manifest_v1.0.csv` that lists every file with its SHA-256 hash. Reviewers
can verify the integrity of the release they downloaded against the
manifest. This module provides the primitives used both to build manifests
and to verify them.

Usage
-----
    from ecdna_bench.utils.checksums import sha256_file, build_manifest, verify_manifest

    # Compute a single checksum
    digest = sha256_file("release/model_checkpoints/eccount_best.pt")

    # Build a manifest for a whole tree
    build_manifest(
        root="release/baseline_predictions/ecseg",
        out_csv="release/manifests/baseline_ecseg_manifest.csv",
    )

    # Verify a manifest
    failures = verify_manifest("release/manifests/dataset_v1.0.csv", root="/data/ecdna")
    assert not failures, failures
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the SHA-256 hex digest of a file.

    Parameters
    ----------
    path
        File path.
    chunk_size
        Bytes to read per iteration. 1 MiB is a reasonable default —
        large enough to amortize syscall overhead, small enough to keep
        memory usage bounded.

    Returns
    -------
    hex digest string (64 chars).
    """
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_manifest(
    root: str | Path,
    out_csv: str | Path,
    *,
    include_suffixes: tuple[str, ...] | None = None,
    recursive: bool = True,
) -> int:
    """
    Build a CSV manifest of all files under `root` with their SHA-256 digests
    and sizes.

    Parameters
    ----------
    root
        Directory to scan.
    out_csv
        Output CSV path. Created with columns `relpath,size_bytes,sha256`.
    include_suffixes
        If given, only files whose lowercased suffix is in this tuple are
        recorded. Example: `(".png", ".tif", ".csv")`. Pass `None` to
        include everything.
    recursive
        If True (default), descend into subdirectories.

    Returns
    -------
    Number of files written to the manifest.
    """
    root_p = Path(root).resolve()
    if not root_p.is_dir():
        raise FileNotFoundError(f"Not a directory: {root_p}")

    iterator = root_p.rglob("*") if recursive else root_p.iterdir()
    rows: list[tuple[str, int, str]] = []
    for p in sorted(iterator):
        if not p.is_file():
            continue
        if include_suffixes is not None and p.suffix.lower() not in include_suffixes:
            continue
        relpath = p.relative_to(root_p).as_posix()
        size = p.stat().st_size
        digest = sha256_file(p)
        rows.append((relpath, size, digest))

    out_p = Path(out_csv)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["relpath", "size_bytes", "sha256"])
        writer.writerows(rows)

    return len(rows)


def verify_manifest(
    manifest_csv: str | Path,
    root: str | Path,
) -> list[dict[str, str]]:
    """
    Check every file listed in a manifest CSV against the contents of `root`.

    Returns a list of dicts describing each failure. An empty list means the
    directory matches the manifest exactly.

    Failure types:
      - {"relpath": ..., "reason": "missing"}            file listed but not found
      - {"relpath": ..., "reason": "size_mismatch", ...} size does not match
      - {"relpath": ..., "reason": "digest_mismatch",...} SHA-256 does not match
    """
    root_p = Path(root).resolve()
    failures: list[dict[str, str]] = []

    with open(manifest_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rel = row.get("relpath") or row["relative_path"]
            expected_size = int(row["size_bytes"])
            expected_digest = row["sha256"]
            p = root_p / rel

            if not p.is_file():
                failures.append({"relpath": rel, "reason": "missing"})
                continue

            actual_size = p.stat().st_size
            if actual_size != expected_size:
                failures.append(
                    {
                        "relpath": rel,
                        "reason": "size_mismatch",
                        "expected": str(expected_size),
                        "actual": str(actual_size),
                    }
                )
                continue

            actual_digest = sha256_file(p)
            if actual_digest != expected_digest:
                failures.append(
                    {
                        "relpath": rel,
                        "reason": "digest_mismatch",
                        "expected": expected_digest,
                        "actual": actual_digest,
                    }
                )

    return failures
