#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH -p general
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH -t 2:00:00
#SBATCH --job-name=ecdna-classical-optimised
#SBATCH --output=logs/classical_optimised_%j.out
#SBATCH --error=logs/classical_optimised_%j.err

set -Eeuo pipefail
trap 'echo ""; echo "ERROR on line $LINENO"; echo "Command: $BASH_COMMAND"; echo "Exit code: $?"; echo "Finished with error at: $(date)"' ERR

PYTHON=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python
PROJECT_ROOT=/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench

CONFIG="$PROJECT_ROOT/configs/default.classical_optimised.yaml"
CONSISTENCY_CSV="$PROJECT_ROOT/release/manifests/dl_master_metadata_stage1_step3_consistency.csv"
FROZEN_JSON="$PROJECT_ROOT/configs/classical/stage3_frozen_params.json"
OUT_DIR="/work/users/b/e/behnamie/ecDNA_Data/benchmark/predictions/classical_optimised"
RUN_LOG="$PROJECT_ROOT/logs/classical_optimised_run_${SLURM_JOB_ID}.log"

cd "$PROJECT_ROOT"
mkdir -p logs configs/classical "$OUT_DIR"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT/src:${PYTHONPATH:-}"

# Pin BLAS/OpenMP threads to 1 — we get our parallelism from worker processes,
# not from BLAS-level threading. Without this, 16 workers × N BLAS threads
# saturates the node and slows everything down.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Suppress the TIFF-tag warnings (50838 / 50839) from the input images.
export OPENCV_LOG_LEVEL=ERROR

echo "=========================================="
echo "Job ID     : ${SLURM_JOB_ID:-NA}"
echo "Node       : ${SLURMD_NODENAME:-NA}"
echo "Started    : $(date)"
echo "Project    : $PROJECT_ROOT"
echo "Python     : $PYTHON"
echo "Config     : $CONFIG"
echo "Frozen JSON: $FROZEN_JSON"
echo "Output dir : $OUT_DIR"
echo "Run log    : $RUN_LOG"
echo "=========================================="

"$PYTHON" --version

echo ""
echo "===== Git / environment diagnostics ====="
git rev-parse --show-toplevel || true
git rev-parse HEAD || true
git status --short || true
which "$PYTHON" || true

echo ""
echo "===== Required files ====="
ls -lh "$CONSISTENCY_CSV"
ls -lh "$FROZEN_JSON"
ls -lh "$PROJECT_ROOT/src/ecdna_bench/classical/infer.py"
ls -lh "$PROJECT_ROOT/src/ecdna_bench/classical/preprocess.py"
ls -lh "$PROJECT_ROOT/src/ecdna_bench/cli/run_classical.py"

echo ""
echo "===== Create optimised run config ====="
cat > "$CONFIG" <<YAML
paths:
  consistency_csv: $CONSISTENCY_CSV
  frozen_params_json: $FROZEN_JSON
  classical_masks: $OUT_DIR

classical:
  n_workers: 16
YAML

cat "$CONFIG"

echo ""
echo "===== Optional clean output ====="
echo "CLEAN_OUTPUT=${CLEAN_OUTPUT:-0}"
if [[ "${CLEAN_OUTPUT:-0}" == "1" ]]; then
    echo "Removing old output directory: $OUT_DIR"
    rm -rf "$OUT_DIR"
    mkdir -p "$OUT_DIR"
else
    echo "Keeping existing output directory. Existing masks may be skipped by run_classical.py."
fi

echo ""
echo "===== Syntax check changed files ====="
"$PYTHON" -m py_compile \
    "$PROJECT_ROOT/src/ecdna_bench/classical/infer.py" \
    "$PROJECT_ROOT/src/ecdna_bench/classical/preprocess.py" \
    "$PROJECT_ROOT/src/ecdna_bench/cli/run_classical.py"

echo ""
echo "===== Smoke test: imports, config, JSON, first image ====="
"$PYTHON" - <<'PY'
import inspect
import json
from pathlib import Path

import cv2
import pandas as pd

from ecdna_bench.cli._common import load_config, get_path
import ecdna_bench.classical.infer as infer_mod
import ecdna_bench.classical.preprocess as prep_mod
from ecdna_bench.classical.infer import (
    load_frozen_params,
    frozen_params_for_cell_line,
    infer_one,
)

config_path = Path("configs/default.classical_optimised.yaml")
cfg = load_config(config_path)

consistency_csv = get_path(cfg, "consistency_csv")
frozen_json_path = get_path(cfg, "frozen_params_json")
out_dir = get_path(cfg, "classical_masks")

print("infer.py source      :", inspect.getsourcefile(infer_mod))
print("preprocess.py source :", inspect.getsourcefile(prep_mod))
print("consistency_csv      :", consistency_csv)
print("frozen_json          :", frozen_json_path)
print("classical_masks      :", out_dir)

with open(frozen_json_path) as f:
    raw = json.load(f)

print("frozen JSON top-level keys:", list(raw.keys()))
expected = {"COLO320DM", "NCI-H2170", "NCI-H716", "SNU16"}
missing = expected - set(raw.keys())
assert not missing, f"frozen JSON missing cell lines: {missing}"

for cl in sorted(expected):
    p = frozen_params_for_cell_line(raw, cl)
    print(
        "params:",
        cl,
        "combo=", p.combo,
        "thr_factor=", p.threshold_factor,
        "merge=", p.merge_distance,
    )

df = pd.read_csv(consistency_csv)
print("CSV shape:", df.shape)
print("CSV columns sample:", list(df.columns[:12]))

df = df[df["count_mask_consistent"].fillna(False)].copy()
print("QC-passed rows:", len(df))
assert len(df) > 0, "No QC-passed rows found"

# Smoke-test one image per cell line so we exercise every preprocessing chain
for cl in sorted(expected):
    sub = df[df["cell_line"] == cl]
    if len(sub) == 0:
        print(f"WARN: no QC-passed rows for cell line {cl}")
        continue
    row = sub.iloc[0]
    uid = str(row["unique_id"])
    rgb_path = Path(str(row["rgb_fullpath"]))
    roi_path = Path(str(row.get("roi_fullpath", "")))

    assert rgb_path.is_file(), f"RGB file not found: {rgb_path}"
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    assert rgb is not None, f"cv2 could not read RGB: {rgb_path}"
    roi = None
    if str(roi_path) and roi_path.is_file():
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)

    params = frozen_params_for_cell_line(raw, str(row["cell_line"]))
    result = infer_one(rgb, roi, params)
    print(f"Smoke {cl}: uid={uid} status={result.status} "
          f"shape={result.pred_mask.shape} "
          f"counts=({result.n_total}, {result.n_ecdna}, {result.n_chromosome})")
    assert result.status == "ok", f"infer_one returned non-ok status: {result.status}"
PY

echo ""
echo "===== Run classical optimised inference ====="
set -o pipefail
"$PYTHON" -m ecdna_bench.cli.run_classical \
    --config "$CONFIG" \
    --split all \
    --n-workers 16 \
    --max-tasks-per-child 50 \
    --log-level INFO 2>&1 | tee "$RUN_LOG"

echo ""
echo "===== Output summary ====="
echo "Number of PNG masks:"
find "$OUT_DIR" -maxdepth 1 -name '*.png' | wc -l

echo ""
echo "First few masks:"
find "$OUT_DIR" -maxdepth 1 -name '*.png' | head

echo ""
echo "Check for warnings/errors in run log:"
grep -iE "warning|error" "$RUN_LOG" | grep -vE "TIFF_Warning|TIFFReadDirectory" || true

echo "=========================================="
echo "Finished   : $(date)"
echo "=========================================="