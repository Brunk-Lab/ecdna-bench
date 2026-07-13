"""
ecdna_bench.cli.run_classical
================================
Parallel classical pipeline inference → binary mask PNGs.

Usage
-----
    python -m ecdna_bench.cli.run_classical --config configs/default.yaml

Required config keys (under ``paths``)
---------------------------------------
* ``consistency_csv``      — QC-passed metadata with full paths
* ``frozen_params_json``   — Stage-3 frozen params JSON
* ``classical_masks``      — output directory for prediction masks
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path
from typing import Optional

from ecdna_bench.cli._common import (
    get_path, load_config, require_file, resolve_path, setup_logging,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run classical pipeline inference on all benchmark images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    p.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--n-workers", type=int, default=None,
                   help="Number of worker processes. Defaults to "
                        "config.classical.n_workers or 8. On the Longleaf "
                        "login node, keep this ≤ 2.")
    p.add_argument("--force", action="store_true",
                   help="Re-render masks even if the output PNG already exists.")
    p.add_argument("--max-tasks-per-child", type=int, default=50,
                   help="Recycle each worker after this many tasks to bound "
                        "memory growth. Set to 0 to disable.")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _infer_one_worker(args):
    """Worker: infer one image and write its mask.

    Returns
    -------
    (uid, status, detail)
        status ∈ {"ok", "skip", "error"}
        detail is "" for ok/skip, or a multi-line traceback string for error.
    """
    (uid, rgb_path, roi_path, cell_line,
     out_path_str, params_json_str, force) = args

    import cv2  # noqa: F401  (imported here so each worker has its own handle)
    from ecdna_bench.classical.infer import (
        load_frozen_params, frozen_params_for_cell_line, infer_one,
    )

    out_path = Path(out_path_str)
    if out_path.exists() and not force:
        return uid, "skip", ""

    try:
        frozen = load_frozen_params(Path(params_json_str))
        params = frozen_params_for_cell_line(frozen, cell_line)

        rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        if rgb is None:
            return uid, "error", f"cv2.imread returned None for RGB: {rgb_path}"

        roi = None
        if roi_path and Path(roi_path).is_file():
            roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

        result = infer_one(rgb, roi, params)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = str(out_path) + ".tmp.png"
        cv2.imwrite(tmp, result.pred_mask)
        os.replace(tmp, str(out_path))
        return uid, "ok", ""

    except Exception:
        return uid, "error", traceback.format_exc()


def _make_executor(n_workers: int, max_tasks_per_child: int) -> ProcessPoolExecutor:
    """Build a ProcessPoolExecutor, using max_tasks_per_child if supported.

    Python 3.11+ supports max_tasks_per_child, which forces worker recycling
    after N tasks. This bounds peak memory growth from any leak in cv2 /
    numpy / our own code. On older interpreters, fall back silently.
    """
    if max_tasks_per_child and max_tasks_per_child > 0:
        try:
            return ProcessPoolExecutor(
                max_workers=n_workers,
                max_tasks_per_child=max_tasks_per_child,
            )
        except TypeError:
            logger.warning(
                "Python < 3.11: max_tasks_per_child not supported; "
                "running without worker recycling."
            )
    return ProcessPoolExecutor(max_workers=n_workers)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config)

    consistency_csv = get_path(cfg, "consistency_csv")
    frozen_json     = get_path(cfg, "frozen_params_json")
    out_dir         = get_path(cfg, "classical_masks", required=False) or \
                      resolve_path("release/harmonized_masks/classical")
    n_workers       = args.n_workers or cfg.get("classical", {}).get("n_workers", 8)

    require_file(consistency_csv, "consistency CSV")
    require_file(frozen_json,     "frozen params JSON")
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    df = pd.read_csv(consistency_csv)
    df = df[df["count_mask_consistent"].fillna(False)]
    if args.split != "all":
        df = df[df["split"] == args.split]

    logger.info("Running classical inference on %d images → %s", len(df), out_dir)
    logger.info("Workers: %d   max_tasks_per_child: %d   force: %s",
                n_workers, args.max_tasks_per_child, args.force)

    tasks = []
    for _, row in df.iterrows():
        uid = str(row["unique_id"])
        roi = str(row.get("roi_fullpath", "")) if "roi_fullpath" in row else ""
        tasks.append((
            uid,
            str(row["rgb_fullpath"]),
            roi,
            str(row["cell_line"]),
            str(out_dir / f"{uid}.png"),
            str(frozen_json),
            args.force,
        ))

    ok = skip = err = 0
    error_summary: Counter[str] = Counter()
    pool_died = False

    with _make_executor(n_workers, args.max_tasks_per_child) as ex:
        futs = {ex.submit(_infer_one_worker, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            uid = futs[fut]
            try:
                _, status, detail = fut.result()
                if status == "ok":
                    ok += 1
                elif status == "skip":
                    skip += 1
                else:
                    err += 1
                    # First line of traceback is usually the most informative
                    last_line = detail.strip().splitlines()[-1] if detail else "no detail"
                    error_summary[last_line] += 1
                    logger.warning("worker failed for %s: %s", uid, last_line)
                    logger.debug("worker traceback for %s:\n%s", uid, detail)

            except BrokenProcessPool as exc:
                pool_died = True
                err += 1
                error_summary["BrokenProcessPool (likely OOM-killed worker)"] += 1
                logger.error(
                    "Worker pool broke while processing %s: %s. "
                    "This usually means a worker was OOM-killed by the OS. "
                    "Reduce --n-workers or request more memory. Aborting "
                    "remaining tasks.",
                    uid, exc,
                )
                # Cancel everything else; futures past this point will all
                # fail with the same error.
                for f in futs:
                    if not f.done():
                        f.cancel()
                break

            except Exception as exc:
                err += 1
                error_summary[f"{type(exc).__name__}: {exc}"] += 1
                logger.warning("future raised for %s: %r", uid, exc)

            done = ok + skip + err
            if done % 100 == 0 or done == len(tasks):
                logger.info("Progress: %d/%d  ok=%d skip=%d err=%d",
                            done, len(tasks), ok, skip, err)

    logger.info("run_classical complete — ok=%d skip=%d err=%d", ok, skip, err)

    if error_summary:
        logger.warning("=== Error histogram (top causes) ===")
        for msg, n in error_summary.most_common(10):
            logger.warning("  %4d × %s", n, msg)

    # Exit non-zero if we lost the pool or had any failures, so SLURM marks
    # the job FAILED and the wrapper script's `set -e` trips.
    if pool_died:
        sys.exit(2)
    if err > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()