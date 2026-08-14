#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH -t 4:00:00
#SBATCH --qos=gpu_access
#SBATCH --job-name=ecdna-eccount-infer
#SBATCH --output=logs/eccount_infer_%j.out
#SBATCH --error=logs/eccount_infer_%j.err

set -euo pipefail

PYTHON="${ECDNA_PYTHON:-python}"

echo "Python : $PYTHON"
"$PYTHON" --version

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

PROJECT_ROOT="${ECDNA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

cd "$PROJECT_ROOT"
mkdir -p logs

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"
echo "=========================================="

"$PYTHON" -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import ecdna_bench
print('ecdna_bench    : OK')
"

"$PYTHON" -m ecdna_bench.cli.run_eccount \
    --config configs/default.yaml \
    --split all \
    --log-level INFO

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="