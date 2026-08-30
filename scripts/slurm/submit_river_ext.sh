#!/bin/bash
#SBATCH -p a100-gpu,l40-gpu --gres=gpu:1 --qos=gpu_access
#SBATCH -N 1 --ntasks=1 --cpus-per-task=8 --mem=48g -t 12:00:00
#SBATCH --job-name=ecdna-river-ext
#SBATCH --output=logs/river_ext_%j.out
#SBATCH --error=logs/river_ext_%j.err
set -euo pipefail
export PYTHONNOUSERSITE=1
export OPENCV_LOG_LEVEL=ERROR
cd /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench
/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench/bin/python \
    -m ecdna_bench.cli.run_eccount \
    --config release/river_roi_run/config_river_eccount_extension.yaml \
    --split all --log-level INFO
