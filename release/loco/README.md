# Leave-one-cell-line-out runs (Supplementary §10.9)

Copied unchanged from `outputs/eccount_loco/` of the run on Longleaf.

- `holdout_<line>/`: one folder per run. `run_config.yaml` (configuration),
  `split_composition.csv` (images and objects per line and split),
  `train_history.csv` (per-epoch losses), `run_manifest.json` (seed, sizes,
  checks, timestamps), `eval_metadata.csv` (the images the run was scored on).
  The `*_seen_lines` files score the same run on the test images of the three
  cell lines it was trained with.
- `holdout_nci_h2170_control/`: the size-matched control for NCI-H2170.
- `figures_partB/`: panels and source data for Fig. 6e–g and Extended Data Fig. 8e.
- `figures/`: earlier panels of the same analysis, with their source data.

The trained checkpoints are attached to the GitHub release, not stored here.
Produced by `scripts/train_eccount_loco.py` (via `slurm/submit_eccount_loco.sh`),
`scripts/collect_loco_results.py`, `scripts/rescore_in_distribution_by_scope.py`,
`scripts/loco_figures.py`, `scripts/loco_figures_partB.py` and
`scripts/build_source_extended_data_fig8.py`.
