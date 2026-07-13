"""
ecdna_bench.utils.seeding — centralize all seed-setting behavior.

Every workflow that has a random component (splitting, augmentation, model
initialization, Bayesian optimization) calls `seed_everything(seed)` once
at startup. This is the only place torch is imported — if the user installed
the minimal `[eval]` extra and does not have torch available, the torch
branch is silently skipped.

Usage
-----
    from ecdna_bench.utils.seeding import seed_everything
    seed_everything(42)

Caveat
------
Fixed seeds produce deterministic behavior on a given machine but NOT
bit-exact reproduction across different GPUs. For GPU training, additionally
call `seed_everything(..., deterministic=True)` which sets
`torch.backends.cudnn.deterministic = True` — this trades speed for
tighter reproducibility.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, *, deterministic: bool = False) -> None:
    """
    Seed Python random, numpy, and (if available) PyTorch.

    Parameters
    ----------
    seed
        Integer seed value. Applied identically to all RNGs.
    deterministic
        If True, additionally set PyTorch to deterministic cuDNN mode.
        Default False because deterministic mode is ~10-30% slower for
        convolutional workloads. The paper results were produced with
        deterministic=False on GPU; see `docs/REPRODUCTION.md`.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch  # type: ignore
    except ImportError:
        return

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Newer PyTorch: use_deterministic_algorithms raises on nondeterministic ops.
        # We use warn_only=True so training does not crash on ops without a
        # deterministic implementation; the behavior will be as deterministic
        # as the current PyTorch version allows.
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except (AttributeError, TypeError):
            pass


def seed_worker(worker_id: int) -> None:
    """
    Seed a DataLoader worker process from the current numpy state.

    Pass as `worker_init_fn` when constructing a DataLoader so that augmentations
    performed inside workers are reproducible given the main-process seed.
    """
    worker_seed = (np.random.get_state()[1][0] + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
