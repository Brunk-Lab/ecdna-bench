#!/bin/bash
#SBATCH --job-name=ecdna-opt-h2170
#SBATCH --array=0
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=4
#SBATCH --mem=96g
#SBATCH -t 24:00:00
#SBATCH --output=logs/opt_h2170_%A_%a.out
#SBATCH --error=logs/opt_h2170_%A_%a.err

set -euo pipefail

PYTHON="${ECDNA_PYTHON:-python}"
PROJECT_ROOT="${ECDNA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR
export MALLOC_ARENA_MAX=2

# Hard safety caps for pathological dense images.
export ECDNA_OPT_MAX_OBJECTS="${ECDNA_OPT_MAX_OBJECTS:-2000}"
export ECDNA_OPT_MAX_MATCH_PAIRS="${ECDNA_OPT_MAX_MATCH_PAIRS:-1000000}"

cd "$PROJECT_ROOT"
mkdir -p logs

CELL_LINE="NCI-H2170"

STAGE1_MAX_TRAIN="${STAGE1_MAX_TRAIN:-25}"
STAGE23_MAX_TRAIN="${STAGE23_MAX_TRAIN:-100}"
STAGE23_MAX_VAL="${STAGE23_MAX_VAL:-50}"
STAGE23_MAX_TEST="${STAGE23_MAX_TEST:-50}"

# For NCI-H2170, keep this low. This is the biggest OOM lever.
N_WORKERS="${N_WORKERS:-1}"
STAGE1_WORKERS="${STAGE1_WORKERS:-1}"
SEED="${SEED:-42}"

FORCE_FLAG=""
if [[ "${FORCE_ALL:-0}" == "1" ]]; then
    FORCE_FLAG="--force"
    echo "FORCE_ALL=1 → existing outputs will be recomputed."
else
    echo "FORCE_ALL=0 → existing partial CSVs will be resumed when possible."
fi

echo "=========================================="
echo "Job ID              : ${SLURM_JOB_ID}"
echo "Array task          : ${SLURM_ARRAY_TASK_ID}"
echo "Node                : ${SLURMD_NODENAME}"
echo "Project             : ${PROJECT_ROOT}"
echo "Python              : ${PYTHON}"
"$PYTHON" --version
echo "Cell line           : ${CELL_LINE}"
echo "Stages              : 2 3"
echo "Stage 1 max train   : ${STAGE1_MAX_TRAIN}"
echo "Stage 2/3 max train : ${STAGE23_MAX_TRAIN}"
echo "Stage 2/3 max val   : ${STAGE23_MAX_VAL}"
echo "Stage 2/3 max test  : ${STAGE23_MAX_TEST}"
echo "Stage 2/3 workers   : ${N_WORKERS}"
echo "Object cap          : ${ECDNA_OPT_MAX_OBJECTS}"
echo "Match-pair cap      : ${ECDNA_OPT_MAX_MATCH_PAIRS}"
echo "Seed                : ${SEED}"
echo "Started             : $(date)"
echo "=========================================="

"$PYTHON" -c "import ecdna_bench; print('ecdna_bench : OK')"

# Stage 1 already completed for NCI-H2170, so only recover Stage 2 and Stage 3.
for STAGE in 2 3; do
    echo "=========================================="
    echo "Running Stage ${STAGE} for ${CELL_LINE}"
    echo "=========================================="

    "$PYTHON" -m ecdna_bench.cli.optimize_classical \
        --config              configs/default.yaml \
        --stage               "${STAGE}" \
        --cell-line           "${CELL_LINE}" \
        --stage1-max-train    "${STAGE1_MAX_TRAIN}" \
        --stage23-max-train   "${STAGE23_MAX_TRAIN}" \
        --stage23-max-val     "${STAGE23_MAX_VAL}" \
        --stage23-max-test    "${STAGE23_MAX_TEST}" \
        --stage1-workers      "${STAGE1_WORKERS}" \
        --n-workers           "${N_WORKERS}" \
        --seed                "${SEED}" \
        --log-level           INFO \
        ${FORCE_FLAG}
done

echo "=========================================="
echo "Running freeze after NCI-H2170 recovery"
echo "=========================================="

"$PYTHON" -m ecdna_bench.cli.optimize_classical \
    --config configs/default.yaml \
    --stage freeze \
    --log-level INFO

echo "=========================================="
echo "Finished            : $(date)"
echo "=========================================="