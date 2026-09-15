#!/usr/bin/env python3
"""
scripts/check_checkpoint.py
===========================
Check that a checkpoint file is the released model before it is uploaded as a
GitHub release asset (or after it is downloaded).

Presets
-------
  eccount       7,849,601 parameters, 3 input channels, best epoch 49,
                best validation loss 0.6421 (rounded to four decimals)
  eccount-arch  the ecCount architecture only (for leave-one-cell-line-out
                and other retrained checkpoints)
  roi           35,923,337 parameters, 4 input channels (RGB + DAPI),
                best epoch 97, best validation loss 0.1741 when recorded.
                The superseded three-channel ROI model (22 May) fails this check.

The parameter count is the number of floating-point values in the state
dictionary, excluding normalization running statistics; the input channels
are read from the first four-dimensional weight. Epoch and loss are checked
only when the checkpoint records them. The SHA-256 of the file is printed
for SHA256SUMS.

Usage
-----
    python scripts/check_checkpoint.py eccount release/model_checkpoints/eccount_best.pt
    python scripts/check_checkpoint.py roi /path/to/best_checkpoint.pth
    python scripts/check_checkpoint.py --self-test

Exit status: 0 all checks pass, 1 a check fails, 2 the file cannot be read.
Requires PyTorch.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

PRESETS: Dict[str, Dict[str, Any]] = {
    "eccount": {"params": 7_849_601, "in_channels": 3, "epoch": 49, "val_loss": 0.6421},
    # leave-one-cell-line-out and other retrained ecCount models: architecture only
    "eccount-arch": {"params": 7_849_601, "in_channels": 3, "epoch": None, "val_loss": None},
    "roi": {"params": 35_923_337, "in_channels": 4, "epoch": 97, "val_loss": 0.1741},
}
STATE_KEYS = ("model_state_dict", "state_dict", "model", "net", "model_state")
EPOCH_KEYS = ("best_epoch", "epoch", "epoch_completed")
LOSS_KEYS = ("best_val_loss", "val_loss", "best_loss", "best_metric")
BUFFER_SUFFIXES = ("running_mean", "running_var", "num_batches_tracked")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(path: Path):
    import torch

    try:
        return torch.load(str(path), map_location="cpu", weights_only=True), "weights only"
    except Exception:
        # Our own training checkpoints may hold plain Python objects next to
        # the weights. Only load files you produced or verified.
        return torch.load(str(path), map_location="cpu", weights_only=False), "full pickle"


def find_state_dict(obj: Any) -> Tuple[Optional[Dict[str, Any]], str]:
    import torch

    def is_state(d: Any) -> bool:
        return isinstance(d, dict) and d and all(isinstance(v, torch.Tensor) for v in d.values())

    if is_state(obj):
        return obj, "(top level)"
    if isinstance(obj, dict):
        for key in STATE_KEYS:
            value = obj.get(key)
            if hasattr(value, "state_dict") and callable(value.state_dict):
                value = value.state_dict()
            if is_state(value):
                return value, key
    if hasattr(obj, "state_dict") and callable(obj.state_dict):
        return obj.state_dict(), "(module object)"
    return None, ""


def summarize(state: Dict[str, Any]) -> Tuple[int, Optional[int], str]:
    n = 0
    first_conv, first_key = None, ""
    for key, t in state.items():
        if key.endswith(BUFFER_SUFFIXES) or not t.is_floating_point():
            continue
        n += t.numel()
        if first_conv is None and t.dim() == 4:
            first_conv, first_key = int(t.shape[1]), key
    return n, first_conv, first_key


def meta_value(obj: Any, keys) -> Tuple[Optional[float], str]:
    if not isinstance(obj, dict):
        return None, ""
    for key in keys:
        value = obj.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value), key
        if hasattr(value, "item"):
            try:
                return float(value.item()), key
            except (TypeError, ValueError):
                pass
    return None, ""


def check(preset: str, path: Path) -> int:
    exp = PRESETS[preset]
    if not path.is_file():
        print(f"not found: {path}")
        return 2
    try:
        obj, mode = load(path)
    except Exception as exc:
        print(f"cannot load {path}: {exc}")
        return 2
    state, where = find_state_dict(obj)
    if state is None:
        print(f"no state dictionary found in {path}")
        return 2

    n, in_ch, conv_key = summarize(state)
    epoch, epoch_key = meta_value(obj, EPOCH_KEYS)
    loss, loss_key = meta_value(obj, LOSS_KEYS)
    rows = [("parameters", f"{n:,}", f"{exp['params']:,}", n == exp["params"]),
            ("input channels", str(in_ch), str(exp["in_channels"]), in_ch == exp["in_channels"])]
    if epoch is not None and exp["epoch"] is not None:
        rows.append((f"epoch ({epoch_key})", f"{int(epoch)}", str(exp["epoch"]), int(epoch) == exp["epoch"]))
    elif epoch is not None:
        rows.append((f"epoch ({epoch_key})", f"{int(epoch)}", "(any)", True))
    if loss is not None and exp["val_loss"] is not None:
        rows.append((f"validation loss ({loss_key})", f"{loss:.4f}", f"{exp['val_loss']:.4f}",
                     round(loss, 4) == exp["val_loss"]))
    elif loss is not None:
        rows.append((f"validation loss ({loss_key})", f"{loss:.4f}", "(any)", True))

    print(f"file      : {path}")
    print(f"size      : {path.stat().st_size / 1e6:.1f} MB")
    print(f"sha256    : {sha256(path)}")
    print(f"loaded as : {mode}; weights under {where}; first convolution {conv_key}")
    print(f"\n  {'check':<34}{'found':>14}{'expected':>14}")
    failed = 0
    for name, got, want, ok in rows:
        failed += 0 if ok else 1
        print(f"  {name:<34}{got:>14}{want:>14}   {'ok' if ok else 'DIFFERS'}")
    if (epoch is None or loss is None) and exp["epoch"] is not None:
        print("  (epoch or validation loss not recorded in the file; not checked)")
    print()
    label = {"eccount": "the released ecCount model", "eccount-arch": "the ecCount architecture",
             "roi": "the released ROI model"}[preset]
    print(f"VERDICT: {'matches' if not failed else 'does NOT match'} {label}")
    return 1 if failed else 0


def self_test() -> int:
    import torch

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def fake(params: int, in_ch: int, name: str, **meta) -> Path:
            conv = torch.zeros(64, in_ch, 3, 3)
            rest = params - conv.numel()
            state = {"enc.0.weight": conv, "body.weight": torch.zeros(rest),
                     "norm.running_mean": torch.zeros(64)}
            path = tmp / name
            torch.save({"model_state_dict": state, **meta}, path)
            return path

        cases = [
            ("eccount released", "eccount",
             fake(7_849_601, 3, "e.pt", best_epoch=49, best_val_loss=0.642057, epoch_completed=49), 0),
            ("eccount wrong epoch", "eccount", fake(7_849_601, 3, "e2.pt", best_epoch=58, best_val_loss=0.6415), 1),
            ("eccount-arch accepts a retrained model", "eccount-arch",
             fake(7_849_601, 3, "l.pt", best_epoch=63, best_val_loss=0.7012), 0),
            ("eccount-arch rejects the ROI model", "eccount-arch", fake(35_923_337, 4, "x.pth"), 1),
            ("roi four channels", "roi", fake(35_923_337, 4, "r.pth", epoch=97, best_val_loss=0.17409), 0),
            ("roi superseded three-channel", "roi", fake(35_923_337 - 64 * 9, 3, "r3.pth"), 1),
            ("missing file", "roi", tmp / "absent.pth", 2),
        ]
        import contextlib
        import io
        for name, preset, path, want in cases:
            with contextlib.redirect_stdout(io.StringIO()):
                got = check(preset, path)
            passed = got == want
            ok &= passed
            print(f"  {'ok  ' if passed else 'FAIL'} {name} (exit {got}, expected {want})")
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset", nargs="?", choices=sorted(PRESETS))
    ap.add_argument("checkpoint", nargs="?", type=Path)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.preset or not args.checkpoint:
        ap.print_usage()
        return 2
    return check(args.preset, args.checkpoint)


if __name__ == "__main__":
    sys.exit(main())
