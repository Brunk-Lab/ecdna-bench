"""
ecdna_bench.cli
================
Command-line entry points for every pipeline stage.

Each module follows the same pattern:

    python -m ecdna_bench.cli.<module> --config configs/default.yaml [options]

Modules
-------
build_metadata      Build master metadata + counts CSVs.
run_qc              File audit + count-mask consistency check.
optimize_classical  Run Stage-1/2/3 Bayesian optimisation.
run_classical       Parallel classical inference → binary mask PNGs.
train_eccount       Train the ecCount U-Net.
run_eccount         Run ecCount inference → threshold + peaks mask PNGs.
run_baseline        Harmonize one external baseline (ecseg/mia/label_engine).
benchmark           Full cross-model benchmark evaluation.
sensitivity         Sensitivity sweep (d_max × IoU_min × OR/AND).
make_figures        Generate paper figures from frozen CSVs.

No business logic lives in CLI modules — each is a thin argparse wrapper
that loads config, calls the library API, and logs results.
"""
