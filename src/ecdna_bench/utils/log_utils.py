"""
ecdna_bench.utils.log_utils — consistent logging setup for every CLI.

Usage
-----
At program entry (typically inside a `cli/*.py` module):

    from ecdna_bench.utils.log_utils import setup_logging
    from ecdna_bench import load_config

    cfg = load_config(args.config)
    setup_logging(cfg.logging, log_file=cfg.paths.logs_root / "my_run.log")

Inside any other module:

    from ecdna_bench.utils.log_utils import get_logger
    logger = get_logger(__name__)
    logger.info("processing %d samples", n)

Design choices
--------------
- `setup_logging` is idempotent: calling it multiple times does not duplicate
  handlers. This matters when one CLI imports another (e.g., benchmark calls
  run_classical internally).
- The root `ecdna_bench` logger is configured; individual modules use
  `get_logger(__name__)` to inherit the configuration.
- External libraries (matplotlib, bayesian_optimization, urllib3) are muted
  at WARNING level by default to keep logs readable.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ecdna_bench.config import LoggingConfig


_CONFIGURED: bool = False
_ROOT_LOGGER_NAME: str = "ecdna_bench"

# External library loggers that are muted to WARNING unless the user enables
# DEBUG mode explicitly. Keeps production logs clean without losing info.
_NOISY_LIBRARIES = (
    "matplotlib",
    "matplotlib.font_manager",
    "PIL",
    "urllib3",
    "bayes_opt",
    "bayesian_optimization",
)


def setup_logging(
    cfg: "LoggingConfig",
    log_file: str | Path | None = None,
    *,
    force: bool = False,
) -> None:
    """
    Configure the root `ecdna_bench` logger and selected third-party loggers.

    Parameters
    ----------
    cfg
        `LoggingConfig` section from the loaded `Config`. Supplies level,
        format string, and date format.
    log_file
        Optional path to a log file. If provided, a FileHandler is added in
        addition to the StreamHandler that writes to stderr. The parent
        directory is created if it does not exist.
    force
        If True, remove existing handlers on the root logger before adding
        new ones. Useful when a caller wants to redirect logging mid-run.
    """
    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    root = logging.getLogger(_ROOT_LOGGER_NAME)

    if force:
        for h in list(root.handlers):
            root.removeHandler(h)

    level = logging.getLevelName(cfg.level.upper())
    if not isinstance(level, int):
        raise ValueError(f"Unknown logging level: {cfg.level!r}")

    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(fmt=cfg.format, datefmt=cfg.datefmt)

    # Stream handler (stderr) — always on.
    stream_h = logging.StreamHandler(stream=sys.stderr)
    stream_h.setLevel(level)
    stream_h.setFormatter(formatter)
    root.addHandler(stream_h)

    # File handler — optional.
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_h = logging.FileHandler(log_path, mode="a", encoding="utf-8")
        file_h.setLevel(level)
        file_h.setFormatter(formatter)
        root.addHandler(file_h)

    # Mute noisy libraries unless we're in DEBUG mode.
    if level > logging.DEBUG:
        for name in _NOISY_LIBRARIES:
            logging.getLogger(name).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger that inherits the `ecdna_bench` root configuration.

    Parameters
    ----------
    name
        Typically the caller's `__name__` — which under the package layout
        starts with `"ecdna_bench."`. Names that do not start with
        `"ecdna_bench"` are rewritten to live under the package root so a
        single `setup_logging` call controls everything in the repo.
    """
    if not name.startswith(_ROOT_LOGGER_NAME):
        name = f"{_ROOT_LOGGER_NAME}.{name}"
    return logging.getLogger(name)


def is_configured() -> bool:
    """Return whether `setup_logging` has been called in this process."""
    return _CONFIGURED
