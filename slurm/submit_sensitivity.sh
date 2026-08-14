#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 8:00:00
#SBATCH --job-name=ecdna-sensitivity
#SBATCH --output=logs/sensitivity_%j.out
#SBATCH --error=logs/sensitivity_%j.err

set -Eeuo pipefail
trap 'echo ""; echo "ERROR on line $LINENO"; echo "Command: $BASH_COMMAND"; echo "Exit code: $?"; echo "Finished with error at: $(date)"' ERR

# ---------------------------------------------------------------------------
# Use the conda env's Python directly by absolute path.
# Same convention as submit_eccount_train.sh / submit_classical_default.sh.
# ---------------------------------------------------------------------------
PYTHON="${ECDNA_PYTHON:-python}"
PROJECT_ROOT="${ECDNA_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

CONFIG="$PROJECT_ROOT/configs/default.yaml"

# Override-able via environment variables — defaults run all 6 models on test.
MODEL="${MODEL:-all}"
SPLIT="${SPLIT:-test}"
LOG_LEVEL="${LOG_LEVEL:-INFO}"
FORCE_FLAG="${FORCE:-1}"

cd "$PROJECT_ROOT"
mkdir -p logs

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENCV_LOG_LEVEL=ERROR

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID:-NA}"
echo "Node       : ${SLURMD_NODENAME:-NA}"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "Python     : $PYTHON"
echo "Config     : $CONFIG"
echo "Model      : $MODEL"
echo "Split      : $SPLIT"
echo "Force      : $FORCE_FLAG"
echo "=========================================="

"$PYTHON" --version

echo ""
echo "===== Git / environment diagnostics ====="
git rev-parse --show-toplevel || true
git rev-parse HEAD || true
git status --short || true
which "$PYTHON" || true

echo ""
echo "===== Importability check ====="
"$PYTHON" -c "
import ecdna_bench
from ecdna_bench.benchmark.registry import MODEL_REGISTRY
print('ecdna_bench       : OK')
print('Registered models :', list(MODEL_REGISTRY.keys()))
"

echo ""
echo "===== Run sensitivity sweep ====="

CMD=("$PYTHON" -m ecdna_bench.cli.sensitivity
     --config "$CONFIG"
     --model "$MODEL"
     --split "$SPLIT"
     --log-level "$LOG_LEVEL")

if [[ "$FORCE_FLAG" == "1" ]]; then
    CMD+=(--force)
fi

echo "Command: ${CMD[*]}"
echo ""

"${CMD[@]}"

echo ""
echo "===== Output inventory ====="
OUT_DIR="$PROJECT_ROOT/release/frozen_results/sensitivity"
ls -lh "$OUT_DIR" || true

echo ""
echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="