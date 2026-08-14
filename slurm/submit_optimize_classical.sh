#!/bin/bash
#SBATCH --job-name=ecdna-opt
#SBATCH --array=0-3
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 24:00:00
#SBATCH --output=logs/opt_%A_%a.out
#SBATCH --error=logs/opt_%A_%a.err

set -euo pipefail

PYTHON="${ECDNA_PYTHON:-python}"
PROJECT_ROOT="${ECDNA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

echo "Python : $PYTHON"
"$PYTHON" --version

export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

export ECDNA_OPT_MAX_OBJECTS="${ECDNA_OPT_MAX_OBJECTS:-2000}"
export ECDNA_OPT_MAX_MATCH_PAIRS="${ECDNA_OPT_MAX_MATCH_PAIRS:-1000000}"

cd "$PROJECT_ROOT"
mkdir -p logs

CELL_LINES=("NCI-H2170" "SNU16" "NCI-H716" "COLO320DM")
CELL_LINE="${CELL_LINES[$SLURM_ARRAY_TASK_ID]}"

STAGE1_MAX_TRAIN="${STAGE1_MAX_TRAIN:-25}"
STAGE23_MAX_TRAIN="${STAGE23_MAX_TRAIN:-100}"
STAGE1_WORKERS="${STAGE1_WORKERS:-2}"
N_WORKERS="${N_WORKERS:-4}"
SEED="${SEED:-42}"

FORCE_FLAG=""
if [[ "${FORCE_ALL:-0}" == "1" ]]; then
    FORCE_FLAG="--force"
    echo "FORCE_ALL=1 → existing outputs will be recomputed."
fi

echo "=========================================="
echo "Job ID              : $SLURM_JOB_ID"
echo "Array task          : $SLURM_ARRAY_TASK_ID"
echo "Node                : $SLURMD_NODENAME"
echo "Cell line           : $CELL_LINE"
echo "Stage 1 max train   : $STAGE1_MAX_TRAIN"
echo "Stage 2/3 max train : $STAGE23_MAX_TRAIN"
echo "Stage 1 workers     : $STAGE1_WORKERS"
echo "Stage 2/3 workers   : $N_WORKERS"
echo "Object cap          : $ECDNA_OPT_MAX_OBJECTS"
echo "Match-pair cap      : $ECDNA_OPT_MAX_MATCH_PAIRS"
echo "Seed                : $SEED"
echo "Started             : $(date)"
echo "=========================================="

"$PYTHON" -c "import ecdna_bench; print('ecdna_bench : OK')"

for STAGE in 1 2 3; do
    echo "=========================================="
    echo "Running Stage $STAGE for $CELL_LINE"
    echo "=========================================="

    "$PYTHON" -m ecdna_bench.cli.optimize_classical \
        --config             configs/default.yaml \
        --stage              "$STAGE" \
        --cell-line          "$CELL_LINE" \
        --stage1-max-train   "$STAGE1_MAX_TRAIN" \
        --stage23-max-train  "$STAGE23_MAX_TRAIN" \
        --stage1-workers     "$STAGE1_WORKERS" \
        --seed               "$SEED" \
        --n-workers          "$N_WORKERS" \
        --log-level          INFO \
        $FORCE_FLAG
done

echo "=========================================="
echo "Finished            : $(date)"
echo "=========================================="

if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    echo "Task 0: scheduling freeze job with --dependency=afterok:${SLURM_ARRAY_JOB_ID}"

    sbatch \
        --job-name=ecdna-opt-freeze \
        --dependency=afterok:"${SLURM_ARRAY_JOB_ID}" \
        -N 1 --ntasks=1 \
        -p general \
        --cpus-per-task=2 \
        --mem=4g \
        -t 00:30:00 \
        --output="$PROJECT_ROOT/logs/opt_freeze_%j.out" \
        --error="$PROJECT_ROOT/logs/opt_freeze_%j.err" \
        --wrap="
set -euo pipefail
export PYTHONPATH=$PROJECT_ROOT/src:\${PYTHONPATH:-}
cd $PROJECT_ROOT
echo 'Freeze job started: \$(date)'
$PYTHON -m ecdna_bench.cli.optimize_classical \
    --config configs/default.yaml \
    --stage freeze \
    --log-level INFO
echo 'Freeze job finished: \$(date)'
        "
fi