#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH -t 8:00:00
#SBATCH --qos=gpu_access
#SBATCH --job-name=ecdna-river-roi
#SBATCH --output=logs/river_roi_%j.out
#SBATCH --error=logs/river_roi_%j.err

# ---------------------------------------------------------------------------
# ecCount inside River's PREDICTED ROI masks.
#
# Prerequisite (run on a login node first, it is CPU work):
#     python scripts/prepare_river_roi_run.py --dry-run
#     python scripts/prepare_river_roi_run.py --apply
#
# Stage 1  ecCount inference, benchmark set   (1,145 images, ~20 min)
# Stage 2  ecCount inference, extension set   (~1,841 images, ~35 min)
# Stage 3  scoring, benchmark set             (CPU, 8 workers)
#
# Nothing under release/frozen_results/ is touched. All output lands in
# release/river_roi_run/.
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python
PROJECT_ROOT="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench"
RUN_DIR="${PROJECT_ROOT}/release/river_roi_run"

export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

cd "$PROJECT_ROOT"
mkdir -p logs

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"
echo "Run dir    : $RUN_DIR"
echo "=========================================="

# --- fail early if preparation has not been run ----------------------------
for f in "${RUN_DIR}/consistency_benchmark.csv" \
         "${RUN_DIR}/config_river_eccount.yaml" \
         "${RUN_DIR}/config_river_benchmark.yaml"; do
    if [[ ! -f "$f" ]]; then
        echo "ERROR: missing $f"
        echo "Run scripts/prepare_river_roi_run.py --apply first."
        exit 1
    fi
done

"$PYTHON" -c "
import torch, ecdna_bench
print('CUDA available:', torch.cuda.is_available())
print('GPU          :', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('ecdna_bench  : OK')
"

# --- Stage 1: benchmark set ------------------------------------------------
echo; echo "--- Stage 1: ecCount inference, benchmark set (River ROI) ---"
"$PYTHON" -m ecdna_bench.cli.run_eccount \
    --config "${RUN_DIR}/config_river_eccount.yaml" \
    --split all \
    --log-level INFO

# --- Stage 2: extension set ------------------------------------------------
if [[ -f "${RUN_DIR}/consistency_extension.csv" ]]; then
    echo; echo "--- Stage 2: ecCount inference, extension set (River ROI) ---"
    sed "s|consistency_benchmark.csv|consistency_extension.csv|" \
        "${RUN_DIR}/config_river_eccount.yaml" \
        > "${RUN_DIR}/config_river_eccount_extension.yaml"
    "$PYTHON" -m ecdna_bench.cli.run_eccount \
        --config "${RUN_DIR}/config_river_eccount_extension.yaml" \
        --split all \
        --log-level INFO
else
    echo; echo "--- Stage 2 skipped: no consistency_extension.csv ---"
fi

# --- Stage 3: scoring ------------------------------------------------------
echo; echo "--- Stage 3: scoring the benchmark set inside the predicted ROI ---"
"$PYTHON" -m ecdna_bench.cli.benchmark \
    --config "${RUN_DIR}/config_river_benchmark.yaml" \
    --models eccount_peaks eccount_threshold \
    --output-dir "${RUN_DIR}/frozen_results" \
    --log-level INFO

echo
echo "Results:"
ls -la "${RUN_DIR}/frozen_results/or_matching/" 2>/dev/null || \
    echo "  (or_matching/ not written — check the log above)"

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="
