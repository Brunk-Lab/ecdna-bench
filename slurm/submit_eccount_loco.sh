#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p a100-gpu,l40-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH -t 08:00:00
#SBATCH --qos=gpu_access
#SBATCH --job-name=eccount-loco
#SBATCH --array=0-3
#SBATCH --output=logs/eccount_loco_%A_%a.out
#SBATCH --error=logs/eccount_loco_%A_%a.err

# ---------------------------------------------------------------------------
# Leave-one-cell-line-out training, inference and scoring for ecCount.
#
# One array task per held-out cell line. Each task runs the full chain:
#     train  ->  inference on the held-out line  ->  benchmark scoring
#
# Submit from the root of YOUR copy of the repository, inside an `ecdna`
# session (so ECDNA_PYTHON and ECDNA_PROJECT_ROOT are set):
#     sbatch slurm/submit_eccount_loco.sh
#
# Add the size-matched control for NCI-H2170 as a fifth task:
#     sbatch --array=0-4 slurm/submit_eccount_loco.sh
#
# BEFORE SUBMITTING, run the dry run on the login node. It takes two seconds
# and it is the only check that the splits are right:
#     python scripts/train_eccount_loco.py --hold-out all --dry-run
#
# Expected wall time per task: about 1 h 40 m for COLO320DM, NCI-H716 and
# SNU16 (roughly 750 training images, 70 epochs at ~79 s), and about 35 m for
# NCI-H2170, which trains on only 179 images. The eight-hour limit is slack,
# not an estimate.
#
# Where things go (changed 15 September 2026): the project root is taken from
# ECDNA_PROJECT_ROOT, else the folder the job was submitted from, so a lab
# member's runs land in their own copy. Outputs go to
# <project root>/outputs/eccount_loco unless ECDNA_LOCO_OUT is set.
# ---------------------------------------------------------------------------

set -euo pipefail

PYTHON="${ECDNA_PYTHON:-/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python}"
PROJECT_ROOT="${ECDNA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
OUT_ROOT="${ECDNA_LOCO_OUT:-${PROJECT_ROOT}/outputs/eccount_loco}"

if [[ ! -f "${PROJECT_ROOT}/configs/default.yaml" ]]; then
    echo "ERROR: ${PROJECT_ROOT} is not an ecdna-bench repository (no configs/default.yaml)." >&2
    echo "       Submit from the repository root, or set ECDNA_PROJECT_ROOT." >&2
    exit 1
fi
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Python not found at ${PYTHON}. Start an 'ecdna' session before sbatch." >&2
    exit 1
fi

# Array index -> (held-out cell line, control flag)
CELL_LINES=("COLO320DM" "NCI-H2170" "NCI-H716" "SNU16" "NCI-H2170")
CONTROL_FLAGS=("" "" "" "" "--size-matched-control")

IDX="${SLURM_ARRAY_TASK_ID}"
HOLD_OUT="${CELL_LINES[$IDX]}"
CONTROL="${CONTROL_FLAGS[$IDX]}"

if [[ -n "$CONTROL" ]]; then
    SUFFIX="_control"
else
    SUFFIX=""
fi
SLUG=$(echo "$HOLD_OUT" | tr '[:upper:]' '[:lower:]' | tr '-' '_')
RUN_DIR="${OUT_ROOT}/holdout_${SLUG}${SUFFIX}"
RUN_CFG="${RUN_DIR}/run_config.yaml"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR
export PYTHONNOUSERSITE=1

cd "$PROJECT_ROOT"
mkdir -p logs "$OUT_ROOT"
if [[ ! -w "$OUT_ROOT" ]]; then
    echo "ERROR: cannot write to ${OUT_ROOT}." >&2
    exit 1
fi

echo "=========================================="
echo "Job        : ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Node       : ${SLURMD_NODENAME}"
echo "Held out   : ${HOLD_OUT} ${CONTROL}"
echo "Project    : ${PROJECT_ROOT}"
echo "Run dir    : ${RUN_DIR}"
echo "Python     : ${PYTHON}"
echo "Started    : $(date)"
echo "=========================================="

"$PYTHON" --version
"$PYTHON" -c "
import torch, ecdna_bench
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')
print('ecdna_bench   :', ecdna_bench.__file__)
"

# --- Stage 1: train -------------------------------------------------------
echo ""
echo "--- Stage 1/3: training ---"
"$PYTHON" scripts/train_eccount_loco.py \
    --hold-out "$HOLD_OUT" \
    $CONTROL \
    --config configs/default.yaml \
    --out-root "$OUT_ROOT" \
    --log-level INFO

# The training step writes run_config.yaml. If it is missing, training failed
# in a way that did not raise, and the next two stages would score a stale
# checkpoint. Stop here instead.
if [[ ! -f "$RUN_CFG" ]]; then
    echo "ERROR: ${RUN_CFG} was not written. Training did not complete." >&2
    exit 1
fi
if [[ ! -f "${RUN_DIR}/best_model.pt" ]]; then
    echo "ERROR: ${RUN_DIR}/best_model.pt is missing." >&2
    exit 1
fi

# --- Stage 2: inference on the held-out cell line -------------------------
# run_config.yaml points consistency_csv at eval_metadata.csv, which contains
# only the held-out rows, so --split all means "all held-out images".
echo ""
echo "--- Stage 2/3: inference ---"
"$PYTHON" -m ecdna_bench.cli.run_eccount \
    --config "$RUN_CFG" \
    --split all \
    --log-level INFO

# --- Stage 3: score -------------------------------------------------------
# Registry keys are 'eccount_peaks' and 'eccount_mask'. The threshold model's
# key is NOT 'eccount_threshold' — passing that silently scores nothing.
echo ""
echo "--- Stage 3/3: scoring ---"
"$PYTHON" -m ecdna_bench.cli.benchmark \
    --config "$RUN_CFG" \
    --models eccount_peaks eccount_mask \
    --skip-harmonize \
    --output-dir "${RUN_DIR}/results" \
    --log-level INFO

echo ""
echo "=========================================="
echo "Finished   : $(date)"
echo "Results    : ${RUN_DIR}/results"
echo "=========================================="
