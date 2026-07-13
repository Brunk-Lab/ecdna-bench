"""
ecdna_bench.cli._common
========================
Shared utilities for all CLI modules:
* ``load_config`` — YAML loader that merges ``paths.local.yaml`` over defaults.
* ``setup_logging`` — consistent log format + level.
* ``resolve_path`` — expand ~ and make absolute.
* ``require_file`` / ``require_dir`` — guard helpers used across CLIs.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "load_config",
    "setup_logging",
    "resolve_path",
    "require_file",
    "require_dir",
]


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", log_file: Optional[Path] = None) -> None:
    """Configure root logger with a consistent format."""
    fmt = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handlers: list = [logging.StreamHandler(sys.stderr)]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_file)))
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt=datefmt,
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> Dict[str, Any]:
    """Load a YAML config file, merging ``paths.local.yaml`` if present.

    Merge order:
    1. ``config_path`` (default.yaml or eccount.yaml, etc.)
    2. ``{config_dir}/paths.local.yaml`` — user-local path overrides.

    Parameters
    ----------
    config_path:
        Path to the primary config YAML.

    Returns
    -------
    dict
        Merged configuration.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required. Install it with: pip install pyyaml"
        )

    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        cfg: Dict[str, Any] = yaml.safe_load(f) or {}

    # Merge paths.local.yaml if present alongside the config file
    local_paths = config_path.parent / "paths.local.yaml"
    if local_paths.exists():
        with open(local_paths) as f:
            local: Dict[str, Any] = yaml.safe_load(f) or {}
        # Deep-merge: local overrides top-level "paths" key
        if "paths" in local:
            cfg.setdefault("paths", {}).update(local["paths"])
        # Also handle bare path keys at top level
        for k, v in local.items():
            if k != "paths":
                cfg[k] = v

    return cfg


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def resolve_path(p: Any) -> Path:
    """Expand ``~`` and return an absolute Path."""
    return Path(str(p)).expanduser().resolve()


def require_file(path: Path, name: str = "file") -> Path:
    """Raise SystemExit if *path* does not exist as a file."""
    path = resolve_path(path)
    if not path.is_file():
        logging.error("Required %s not found: %s", name, path)
        sys.exit(1)
    return path


def require_dir(path: Path, name: str = "directory", create: bool = False) -> Path:
    """Raise SystemExit if *path* does not exist as a directory.

    If ``create=True``, create it instead of exiting.
    """
    path = resolve_path(path)
    if not path.is_dir():
        if create:
            path.mkdir(parents=True, exist_ok=True)
            logging.info("Created %s: %s", name, path)
        else:
            logging.error("Required %s not found: %s", name, path)
            sys.exit(1)
    return path


def get_path(cfg: Dict[str, Any], key: str, required: bool = True) -> Optional[Path]:
    """Extract a path from ``cfg['paths'][key]``."""
    val = cfg.get("paths", {}).get(key)
    if val is None:
        if required:
            logging.error("Config key 'paths.%s' is not set. "
                          "Add it to configs/paths.local.yaml.", key)
            sys.exit(1)
        return None
    return resolve_path(val)
