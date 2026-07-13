#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 4:00:00
#SBATCH --job-name=ecdna-classic-benchmark
#SBATCH --output=logs/benchmark_classic_bfr_aft_%j.out
#SBATCH --error=logs/benchmark_classic_bfr_aft_%j.err

set -euo pipefail

PYTHON=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python
PROJECT_ROOT=/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

cd "$PROJECT_ROOT"

mkdir -p logs
mkdir -p outputs/classical_bfr_aft

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID:-NA}"
echo "Node       : ${SLURMD_NODENAME:-NA}"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "Python     : $PYTHON"
echo "=========================================="

"$PYTHON" --version
"$PYTHON" -c "import ecdna_bench; print('ecdna_bench: OK')"

echo
echo "Running classical before/after optimization benchmark..."
echo

"$PYTHON" -m ecdna_bench.cli.benchmark \
    --config configs/default.yaml \
    --models classical classical_before_opt \
    --skip-harmonize \
    --output-dir outputs/classical_bfr_aft \
    --log-level INFO \
    --n-workers 8 \
    --force

echo
echo "=========================================="
echo "Finished   : $(date)"
echo "Output dir : outputs/classical_bfr_aft"
echo "=========================================="

echo
echo "Produced files:"
find outputs/classical_bfr_aft -maxdepth 3 -type f | sort