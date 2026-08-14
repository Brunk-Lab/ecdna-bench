# ecdna-bench — Longleaf Environment Setup

This guide describes how to set up the environment on UNC's Longleaf HPC
cluster. If you are not on Longleaf, the only Longleaf-specific parts are the
`module load` lines and the `/proj/<onyen>/...` paths — everything else
(`conda env create`, `pip install -e .`, `pytest`) is standard.

> **Cluster note.** Longleaf now runs RHEL 9 and its module tree changed with
> that migration. In particular there is no longer a `mamba` module, and
> `cuda/12.8` has been retired. The instructions below reflect the current
> module set. If a `module load` fails, check `module avail <name>` — module
> versions are revised periodically and this document may lag behind.

## 1. Filesystem decisions (read this first)

| Question | Answer |
|---|---|
| Where to put the conda env? | `/proj/<onyen>/envs/ecdna-bench` — **not** your home dir and **not** `/tmp`. Home quota is typically 50 GB; a full PyTorch env with CUDA wheels is ~8 GB. `/tmp` is small and periodically purged. |
| Where to put pip/conda cache? | Set both to `/proj/<onyen>/.cache` so cache misses don't fill home. |
| `mamba` or `conda`? | **Use `conda`.** Conda 23.10 and later use the libmamba solver by default, so dependency resolution is already fast and a separate `mamba` install is unnecessary. |

---

## 2. One-time cache redirect

Add to your `~/.bashrc` (or run manually before install):

```bash
export CONDA_PKGS_DIRS=/proj/<onyen>/.cache/conda/pkgs
export PIP_CACHE_DIR=/proj/<onyen>/.cache/pip
mkdir -p $CONDA_PKGS_DIRS $PIP_CACHE_DIR
```

Replace `<onyen>` with your actual UNC ONYEN everywhere below.

---

## 3. Load required modules

Load a compatible GCC and the CUDA toolkit before installing CUDA-linked
packages:

```bash
module purge
module load gcc/11.2.0
module load cuda/12.9        # torch 2.8.0 ships cu128 wheels; 12.9 is
                             # binary-compatible and is Longleaf's default
```

Confirm what is available on your login node before assuming:

```bash
module avail cuda
```

> **Tip:** Put these `module load` lines in a project-specific `env.sh` so you
> don't forget them.

---

## 4. Isolate from your user site-packages

```bash
export PYTHONNOUSERSITE=1
```

This is not optional for a clean install. Without it, Python silently falls
back to `~/.local/lib/python3.10/site-packages`, so packages that are missing
from the environment appear to be present. An environment built without this
variable may work on your account and fail on everyone else's.

---

## 5. Create the environment

```bash
cd /proj/<onyen>/ecdna-bench        # your repo root

conda env create \
    --prefix /proj/<onyen>/envs/ecdna-bench \
    --file env/environment.yml
```

This takes ~15–25 minutes, mostly downloading PyTorch wheels.

> **If conda reports solver conflicts** for `bayesian-optimization` or
> `statannotations`, that is expected — they are pip-only packages handled by
> pip at the end of the install. Proceed.

---

## 6. Activate and verify the install path

```bash
conda activate /proj/<onyen>/envs/ecdna-bench

which python
# → /proj/<onyen>/envs/ecdna-bench/bin/python

python --version
# → Python 3.10.16
```

---

## 7. Install the package and run the test suite

```bash
cd /proj/<onyen>/ecdna-bench
pip install -e .
pytest -q
```

Expected (last line):

```
124 passed, 2 skipped in ~2s
```

The two skips are I/O tests that require sample images from the released
dataset; they are expected on a code-only clone. The remaining tests use
synthetic fixtures and do **not** require the real data. If they fail on a
fresh clone, the problem is the install, not the data.

---

## 8. Two sanity checks

```bash
# 8a. Confirm ecCount parameter count (runs anywhere, no GPU needed)
python -c "
from ecdna_bench.eccount.model import build_model, ModelConfig
m = build_model(ModelConfig())
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
"
# → 7849601

# 8b. Confirm CUDA is visible (GPU node only)
python -c "import torch; print(torch.cuda.is_available())"
# → True
```

`torch.cuda.is_available()` returns `False` on Longleaf login nodes (no GPU).
Run it inside an interactive GPU session:

```bash
srun -p a100-gpu,l40-gpu --gres=gpu:1 --qos=gpu_access \
     --cpus-per-task=2 --mem=8g -t 00:10:00 --pty bash
module load gcc/11.2.0 cuda/12.9
export PYTHONNOUSERSITE=1
conda activate /proj/<onyen>/envs/ecdna-bench
python -c "import torch; print(torch.cuda.is_available())"
```

Partition and QOS names are also revised periodically. Check the current set
with `sinfo -s` if the `srun` above is rejected.

---

## 9. Troubleshooting

### `ModuleNotFoundError: No module named 'ecdna_bench'`
The editable install points at the path where you ran `pip install -e .`.
If you moved the repo:

```bash
conda activate /proj/<onyen>/envs/ecdna-bench
pip install -e /proj/<onyen>/ecdna-bench
```

### A package imports but is not in the environment
You are picking it up from `~/.local`. Set `PYTHONNOUSERSITE=1` (section 4)
and re-check with:

```bash
python -c "import cv2, sys; print(cv2.__file__)"
```

The path must be inside `/proj/<onyen>/envs/ecdna-bench`.

### `ImportError` for `cv2`
The headless OpenCV build sometimes conflicts with a system `libGL`:

```bash
pip install --force-reinstall opencv-python-headless
```

### `torch.cuda.is_available()` returns `False` on a GPU node
Check the CUDA module is compatible with the torch wheel:

```bash
nvcc --version                                      # 12.x
python -c "import torch; print(torch.version.cuda)" # 12.8
```

Any CUDA 12.x module works with a cu128 torch build; a CUDA 11.x or 13.x
module does not.

### `module load` reports an unknown module
Longleaf's module tree changes with OS upgrades. Use `module avail <name>` or
`module spider <name>` to find the current version, and prefer the one marked
`(D)` for default.

---

## 10. Convenience alias (optional)

```bash
alias activate-ecdna='module load gcc/11.2.0 cuda/12.9 && export PYTHONNOUSERSITE=1 && conda activate /proj/<onyen>/envs/ecdna-bench'
```

Then just type `activate-ecdna` at the start of any session.