#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --mem=96g
#SBATCH -t 6:00:00
#SBATCH --job-name=ecdna-river-roi
#SBATCH --output=logs/river_roi_bench_%j.out
#SBATCH --error=logs/river_roi_bench_%j.err

# Submit from the repository root:
#   cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench
#   mkdir -p logs
#   sbatch slurm/submit_benchmark_river_roi.sh
#
# PROJECT_ROOT falls back to $SLURM_SUBMIT_DIR, so submitting from your own
# clone keeps the output in your own clone.

set -euo pipefail

PYTHON="${ECDNA_PYTHON:-/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench}}"

# WORKERS must match --cpus-per-task above. SLURM allocates the cores you asked
# for, not the node's, so oversubscribing here is what crashes the job.
WORKERS="${SLURM_CPUS_PER_TASK:-8}"

# Keep BLAS single-threaded: the parallelism is at the process level.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

# The environment is self-contained; never let ~/.local leak in.
export PYTHONNOUSERSITE=1

cd "$PROJECT_ROOT"
mkdir -p logs

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID:-interactive}"
echo "Node       : ${SLURMD_NODENAME:-$(hostname)}"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "Python     : $PYTHON"
echo "Workers    : $WORKERS"
echo "=========================================="

"$PYTHON" --version
"$PYTHON" -c "import ecdna_bench; print('ecdna_bench: OK')"

"$PYTHON" scripts/benchmark_river_roi.py \
    --repo-root "$PROJECT_ROOT" \
    --workers "$WORKERS" \
    --checkpoint-every 200 \
    --resume

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="
