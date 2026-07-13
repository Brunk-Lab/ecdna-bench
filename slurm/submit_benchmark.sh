#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH -t 4:00:00
#SBATCH --job-name=ecdna-benchmark
#SBATCH --output=logs/benchmark_%j.out
#SBATCH --error=logs/benchmark_%j.err

set -euo pipefail

PYTHON=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python
PROJECT_ROOT=/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench

echo "Python : $PYTHON"
"$PYTHON" --version

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

cd "$PROJECT_ROOT"
mkdir -p logs release/frozen_results

echo "=========================================="
echo "Job ID     : $SLURM_JOB_ID"
echo "Node       : $SLURMD_NODENAME"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "=========================================="

"$PYTHON" -c "import ecdna_bench; print('ecdna_bench: OK')"

"$PYTHON" -m ecdna_bench.cli.benchmark \
    --config configs/default.yaml \
    --log-level INFO \
    --n-workers 8 \
    --skip-harmonize
    
echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="
