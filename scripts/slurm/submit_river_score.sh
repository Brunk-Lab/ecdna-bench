#!/bin/bash
#SBATCH -p general
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=96g -t 3:00:00
#SBATCH --job-name=ecdna-river-score
#SBATCH --output=logs/river_score_%j.out
#SBATCH --error=logs/river_score_%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench
/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python \
    -m ecdna_bench.cli.benchmark \
    --config release/river_roi_run/config_river_benchmark.yaml \
    --models eccount_peaks eccount_mask \
    --output-dir release/river_roi_run/frozen_results \
    --n-workers 4 --log-level INFO
