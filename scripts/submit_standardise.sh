#!/bin/bash
#SBATCH --job-name=standardise
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=logs/standardise_%j.out
#SBATCH --error=logs/standardise_%j.err

set -eo pipefail
export PYTHONNOUSERSITE=1
PY=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python

cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench

# preflight: fail in seconds with a clear message if the interpreter is wrong
"$PY" -c "import sys, imageio, numpy; print('interpreter:', sys.executable); print('imageio', imageio.__version__, '| numpy', numpy.__version__)"

"$PY" -u scripts/standardise_formats.py --apply --delete-originals
