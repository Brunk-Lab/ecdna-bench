#!/bin/bash
#SBATCH --job-name=manifest
#SBATCH --partition=general
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=logs/manifest_%j.out
#SBATCH --error=logs/manifest_%j.err

set -eo pipefail
export PYTHONNOUSERSITE=1
PY=/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python

cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench

"$PY" -u scripts/generate_manifest.py \
    --data-root /proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data \
    --out release/manifests/manifest_v1.0.csv
