#!/usr/bin/env bash
# scripts/assemble_repo.sh
#
# Phase 12 — End-to-end verification script.
# Run this on the assembled repository (all phase outputs merged) to:
#   1. Verify the install and test suite.
#   2. Check all required config keys are present.
#   3. Print a pre-flight checklist.
#
# Usage:
#   bash scripts/assemble_repo.sh [--data-root /path/to/data]

set -euo pipefail

PYTHON="${PYTHON:-python3}"
DATA_ROOT="${1:-}"
LOG="phase12_preflight.log"

echo "============================================================"
echo "  ecdna-bench Phase 12 — Pre-flight checklist"
echo "  $(date)"
echo "============================================================"

# ── 1. Package install ──────────────────────────────────────────
echo ""
echo "[1/6] Installing package ..."
$PYTHON -m pip install -e . --quiet
echo "      OK — pip install -e ."

# ── 2. Test suite ───────────────────────────────────────────────
echo ""
echo "[2/6] Running test suite ..."
$PYTHON -m pytest tests/ -q --tb=short 2>&1 | tee -a "$LOG"
echo "      OK — pytest"

# ── 3. Config check ─────────────────────────────────────────────
echo ""
echo "[3/6] Checking configs ..."
if [ ! -f configs/paths.local.yaml ]; then
    echo "  ⚠  configs/paths.local.yaml not found."
    echo "     Copy configs/paths.example.yaml and edit it."
else
    echo "      OK — configs/paths.local.yaml exists"
fi

# ── 4. Data root ────────────────────────────────────────────────
echo ""
echo "[4/6] Checking data root ..."
if [ -n "$DATA_ROOT" ] && [ -d "$DATA_ROOT" ]; then
    echo "      OK — data root: $DATA_ROOT"
else
    echo "  ⚠  Pass data root as first argument: bash scripts/assemble_repo.sh /path/to/data"
fi

# ── 5. Release directories ──────────────────────────────────────
echo ""
echo "[5/6] Checking release directories ..."
for d in release/split_files release/frozen_results release/manifests; do
    if [ -d "$d" ]; then
        echo "      OK — $d"
    else
        mkdir -p "$d"
        echo "      Created — $d"
    fi
done

# ── 6. Split files ──────────────────────────────────────────────
echo ""
echo "[6/6] Checking split files ..."
for f in release/split_files/train_ids.csv release/split_files/val_ids.csv release/split_files/test_ids.csv; do
    if [ -f "$f" ]; then
        n=$(tail -n +2 "$f" | wc -l | tr -d ' ')
        echo "      OK — $f ($n UIDs)"
    else
        echo "  ⚠  Missing: $f  (copy from project metadata)"
    fi
done

# ── Summary ─────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Pre-flight complete.  Next steps:"
echo ""
echo "  1. Edit configs/paths.local.yaml with real paths."
echo "  2. make build-metadata run-qc"
echo "  3. make run-classical  (or load existing masks)"
echo "  4. make train-eccount  (or load checkpoint)"
echo "  5. make run-eccount"
echo "  6. make run-baselines"
echo "  7. make benchmark"
echo "  8. make sensitivity"
echo "  9. make figures"
echo " 10. Populate README.md benchmark table."
echo " 11. Insert DOI in CITATION.cff."
echo " 12. git tag v1.0.0 && git push --tags"
echo "============================================================"
