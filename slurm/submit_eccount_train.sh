#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH --qos=gpu_access
#SBATCH --job-name=ecdna-eccount-train
#SBATCH --output=logs/eccount_train_%j.out
#SBATCH --error=logs/eccount_train_%j.err

set -euo pipefail

# ---------------------------------------------------------------------------
# Use the conda env's Python directly by absolute path.
# This bypasses the module system PATH conflict entirely — no need for
# `module load anaconda` or `conda activate` in batch scripts.
# ---------------------------------------------------------------------------
PYTHON=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python

# Confirm we have the right Python before doing anything else
echo "Python : $PYTHON"
"$PYTHON" --version

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

PROJECT_ROOT="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench"

cd "$PROJECT_ROOT"
mkdir -p logs outputs/eccount_training

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "=========================================="

# Sanity check: GPU visible + ecdna_bench importable
"$PYTHON" -c "
import torch
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
import ecdna_bench
print('ecdna_bench    : OK')
"

# Run training
"$PYTHON" -m ecdna_bench.cli.train_eccount \
    --config configs/default.yaml \
    --log-level INFO

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="