# SLURM submission scripts

Batch scripts for running the ecdna-bench pipeline on a SLURM cluster. They
were written for UNC's Longleaf but contain no Longleaf-specific paths — the
partition names in the `#SBATCH -p` lines are the only thing you are likely to
need to change for another cluster.

## Before you submit

**Submit from the repository root, not from inside `slurm/`.** The scripts
resolve the project root from `$SLURM_SUBMIT_DIR`, and the log paths in the
`#SBATCH --output` / `--error` directives are relative to it:

```bash
cd /path/to/ecdna-bench
mkdir -p logs
sbatch slurm/submit_benchmark.sh
```

**Tell the scripts which Python to use.** They do not call `conda activate`,
because activating inside a batch job interacts badly with the module system's
PATH. Instead they invoke an interpreter directly. Set it once:

```bash
export ECDNA_PYTHON=/path/to/envs/ecdna-bench/bin/python
```

Put that line in your `~/.bashrc`. Without it the scripts fall back to whatever
`python` is first on `PATH`, which in a batch job is usually the system Python
and will fail at `import ecdna_bench`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ECDNA_PYTHON` | `python` | Interpreter used for every stage |
| `ECDNA_PROJECT_ROOT` | `$SLURM_SUBMIT_DIR`, else `$PWD` | Repository root |

Several scripts expose additional tuning variables with defaults — for example
`N_WORKERS`, `SEED`, `STAGE1_MAX_TRAIN` and `STAGE23_MAX_TRAIN` in
`submit_optimize_classical.sh`. Override them the same way:

```bash
N_WORKERS=8 SEED=1234 sbatch slurm/submit_optimize_classical.sh
```

## The scripts

| Script | Partition | Notes |
|---|---|---|
| `submit_classical_default.sh` | `general` | Classical inference, pre-optimisation parameters |
| `submit_optimize_classical.sh` | `general` | 3-stage Bayesian optimisation; job array, one task per cell line |
| `submit_optimize_classical_h2170_recovery.sh` | `general` | NCI-H2170 re-run |
| `submit_classical_optimized.sh` | `general` | Classical inference, frozen post-optimisation parameters |
| `submit_eccount_train.sh` | `a100-gpu,l40-gpu` | ecCount training, ~12 h, needs `--qos=gpu_access` |
| `submit_eccount_infer.sh` | `a100-gpu,l40-gpu` | ecCount inference → threshold + peaks masks |
| `submit_benchmark.sh` | `general` | Cross-model benchmark evaluation |
| `submit_benchmark_classical_before_after.sh` | `general` | Before/after BO comparison |
| `submit_sensitivity.sh` | `general` | `d_max × IoU_min × policy` sweep |

## Checking partitions

Partition and QOS names change when a cluster is upgraded. If a job is rejected
at submission, list what is currently available:

```bash
sinfo -s
```

and update the `#SBATCH -p` line accordingly.

## Logs

All scripts write to `logs/<stage>_%j.out` and `logs/<stage>_%j.err` relative to
the repository root. `logs/` is gitignored. Create it before your first
submission — SLURM will not create it for you, and the job will fail
immediately if it is missing.