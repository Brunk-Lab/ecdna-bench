# ecdna-bench — Longleaf Environment Setup

This guide describes how to set up the environment on UNC's Longleaf HPC
cluster. If you are not on Longleaf, the only Longleaf-specific parts are the
`module load` lines and the `/proj/<onyen>/...` paths — everything else
(`conda env create`, `pip install -e .`, `pytest`) is standard.

## 1. Filesystem decisions (read this first)

| Question | Answer |
|---|---|
| Where to put the conda env? | `/proj/<onyen>/envs/ecdna-bench` — **not** your home dir. Home quota is typically 50 GB; a full PyTorch env with CUDA wheels is ~8 GB. |
| Where to put pip/conda cache? | Set both to `/proj/<onyen>/.cache` so cache misses don't fill home. |
| `mamba` or `conda`? | **Use `mamba`** to create the env — it is far faster at solving dependencies. `conda activate` then works as usual. |

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

Longleaf requires you to load the CUDA toolkit and a compatible GCC before
installing CUDA-linked packages. For the A100 / L40 GPU nodes (CUDA 12.x):

```bash
module purge
module load gcc/11.2.0
module load cuda/12.8        # match the nvidia-cuda-runtime-cu12 wheel
module load mamba            # loads mamba + conda
```

> **Tip:** Add these three `module load` lines to a project-specific `env.sh`
> so you don't forget them.

---

## 4. Create the environment

```bash
cd /proj/<onyen>/ecdna-bench        # your repo root

mamba env create \
    --prefix /proj/<onyen>/envs/ecdna-bench \
    --file env/environment.yml
```

This takes ~5–10 minutes (mostly downloading PyTorch wheels).

> **If mamba reports solver conflicts** for `bayesian-optimization` or
> `statannotations`, that is expected — they are pip-only packages handled by
> pip at the end of the install. Proceed.

---

## 5. Activate and verify the install path

```bash
conda activate /proj/<onyen>/envs/ecdna-bench

which python
# → /proj/<onyen>/envs/ecdna-bench/bin/python

python --version
# → Python 3.10.x
```

---

## 6. Run the test suite

```bash
cd /proj/<onyen>/ecdna-bench
pytest -q
```

Expected (last line):

```
112 passed in ~4s
```

The unit tests use synthetic fixtures and do **not** require the real dataset.
If they fail on a fresh clone, the problem is the install, not the data.

---

## 7. Two sanity checks

```bash
# 7a. Confirm ecCount parameter count (runs anywhere, no GPU needed)
python -c "
from ecdna_bench.eccount.model import build_model, ModelConfig
m = build_model(ModelConfig())
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
"
# → 7849601

# 7b. Confirm CUDA is visible (GPU node only)
python -c "import torch; print(torch.cuda.is_available())"
# → True
```

`torch.cuda.is_available()` returns `False` on Longleaf login nodes (no GPU).
Run it inside an interactive GPU session, using the same partitions as the
project's SLURM jobs:

```bash
srun -p a100-gpu,l40-gpu --gres=gpu:1 --qos=gpu_access \
     --cpus-per-task=2 --mem=8g -t 00:10:00 --pty bash
module load gcc/11.2.0 cuda/12.8 mamba
conda activate /proj/<onyen>/envs/ecdna-bench
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 8. Troubleshooting

### `ModuleNotFoundError: No module named 'ecdna_bench'`
The editable install requires the repo to be present at the path where you ran
`mamba env create`. If you moved the repo:

```bash
conda activate /proj/<onyen>/envs/ecdna-bench
pip install -e /proj/<onyen>/ecdna-bench
```

### `ImportError` for `cv2`
The headless OpenCV build sometimes conflicts with a system `libGL`:

```bash
pip install --force-reinstall opencv-python-headless==4.10.0
```

### `torch.cuda.is_available()` returns `False` on a GPU node
Check the CUDA module matches the torch wheel:

```bash
nvcc --version                                      # should show 12.8.x
python -c "import torch; print(torch.version.cuda)" # should show 12.8
```

### `pytest` reports fewer than 112 passed
Run with `-v --tb=long` to see which tests fail. Fresh-clone failures are
almost always install-related, not data-related.

---

## 9. Convenience alias (optional)

```bash
alias activate-ecdna='module load gcc/11.2.0 cuda/12.8 mamba && conda activate /proj/<onyen>/envs/ecdna-bench'
```

Then just type `activate-ecdna` at the start of any session.
