#!/bin/bash
# =============================================================================
# ecdna-bench — new user setup
#
# Sets up everything a new user needs on Longleaf: shell settings, workspace,
# a clone of the code, a local paths file, and a Jupyter kernel.
#
# USAGE
#   bash setup_new_user.sh
#
# Safe to run more than once. It checks for existing settings and skips
# anything already done, so re-running after a failure is fine.
# =============================================================================

set -uo pipefail

GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'

ok()    { echo "  ${GREEN}✓${NC} $*"; }
warn()  { echo "  ${YELLOW}!${NC} $*"; }
fail()  { echo "  ${RED}✗${NC} $*"; }
step()  { echo; echo "${BOLD}${BLUE}▸ $*${NC}"; }
die()   { echo; fail "$*"; echo; echo "Setup stopped. Nothing was left half-done that matters."; exit 1; }

echo
echo "${BOLD}=============================================${NC}"
echo "${BOLD}  ecdna-bench — setting up your account${NC}"
echo "${BOLD}=============================================${NC}"

# -----------------------------------------------------------------------------
# 1. Who are you
# -----------------------------------------------------------------------------
step "Step 1 of 9 — identifying you"

ONYEN_GUESS="$(whoami)"
read -r -p "  Your ONYEN [${ONYEN_GUESS}]: " ONYEN_INPUT
ONYEN="${ONYEN_INPUT:-$ONYEN_GUESS}"

[[ "$ONYEN" =~ ^[a-zA-Z0-9_-]+$ ]] || die "'$ONYEN' does not look like a valid ONYEN."
ok "ONYEN: $ONYEN"

PROJ=/proj/brunk_ecdna_cv_project
MYDIR="$PROJ/$ONYEN"
REPO="$MYDIR/repos/ecdna-bench"
ENV_CANONICAL="$PROJ/Poorya/envs/ecdna-bench"
DATA="$PROJ/Poorya/ecDNA_Data"
SOURCE_REPO="$PROJ/Poorya/ecdna-bench"

# -----------------------------------------------------------------------------
# 2. Can you reach the shared storage
# -----------------------------------------------------------------------------
step "Step 2 of 9 — checking access to the lab storage"

[[ -d "$PROJ" ]] || die "Cannot see $PROJ.
  You are probably not in the brunk_ecdna_cv_project group yet.
  Request access at https://help.rc.unc.edu"
ok "Lab storage reachable"

[[ -r "$SOURCE_REPO" ]] && ok "Source repository readable" \
    || warn "Cannot read $SOURCE_REPO — the clone will come from GitHub instead"

[[ -r "$DATA" ]] && ok "Image data readable" \
    || warn "Cannot read $DATA — ask about group permissions"

[[ -x "$ENV_CANONICAL/bin/python" ]] || die "Environment not found at $ENV_CANONICAL"
ok "Environment found"

# -----------------------------------------------------------------------------
# 3. Home directory space
# -----------------------------------------------------------------------------
step "Step 3 of 9 — checking your home directory"

HOME_USE=$(df -P "$HOME" 2>/dev/null | awk 'NR==2 {gsub(/%/,"",$5); print $5}')
if [[ -n "${HOME_USE:-}" && "$HOME_USE" -gt 90 ]]; then
    warn "Home is ${HOME_USE}% full. Jupyter may refuse to start."
    warn "After this script finishes, run:  conda clean --all --yes"
else
    ok "Home has room (${HOME_USE:-?}% used)"
fi

# -----------------------------------------------------------------------------
# 4. Shell settings
# -----------------------------------------------------------------------------
step "Step 4 of 9 — adding your shell settings"

MARKER="# >>> ecdna-bench settings >>>"

if grep -qF "$MARKER" "$HOME/.bashrc" 2>/dev/null; then
    ok "Settings already present in ~/.bashrc — leaving them alone"
else
    BACKUP="$HOME/.bashrc.backup-$(date +%Y%m%d-%H%M%S)"
    cp "$HOME/.bashrc" "$BACKUP" 2>/dev/null && ok "Backed up ~/.bashrc to $(basename "$BACKUP")"

    cat >> "$HOME/.bashrc" <<BASHRC_END

$MARKER
# Added by setup_new_user.sh on $(date +%Y-%m-%d)
# To remove: delete from this marker down to the matching <<< line.

export ONYEN="$ONYEN"
export PROJ="$PROJ"
export MYDIR="$MYDIR"
export REPO="$REPO"
export ENV_CANONICAL="$ENV_CANONICAL"
export DATA="$DATA"
export SOURCE_REPO="$SOURCE_REPO"

# Required by every SLURM job script — without this, jobs use the wrong Python
export ECDNA_PYTHON="$ENV_CANONICAL/bin/python"

# Keep downloaded package caches off the 50 GB home quota
export PIP_CACHE_DIR="$MYDIR/.cache/pip"
export XDG_CACHE_HOME="$MYDIR/.cache"
export CONDA_PKGS_DIRS="$MYDIR/.cache/conda/pkgs"

# One word to start a working session
alias ecdna='module load anaconda >/dev/null 2>&1 && conda activate "\$ENV_CANONICAL" && cd "\$REPO"'
# <<< ecdna-bench settings <<<
BASHRC_END
    ok "Settings written to ~/.bashrc"
fi

export ONYEN PROJ MYDIR REPO ENV_CANONICAL DATA SOURCE_REPO
export ECDNA_PYTHON="$ENV_CANONICAL/bin/python"
export PIP_CACHE_DIR="$MYDIR/.cache/pip"
export XDG_CACHE_HOME="$MYDIR/.cache"
export CONDA_PKGS_DIRS="$MYDIR/.cache/conda/pkgs"

# -----------------------------------------------------------------------------
# 5. Workspace
# -----------------------------------------------------------------------------
step "Step 5 of 9 — creating your workspace"

mkdir -p "$MYDIR"/{repos,runs,logs,envs,.cache} || die "Could not create $MYDIR"
mkdir -p "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$CONDA_PKGS_DIRS"
ok "Workspace ready at $MYDIR"

# -----------------------------------------------------------------------------
# 6. The code
# -----------------------------------------------------------------------------
step "Step 6 of 9 — getting the code"

if [[ -d "$REPO/.git" ]]; then
    ok "Repository already present at $REPO"
elif git clone --quiet https://github.com/PooryaBehnamie/ecdna-bench.git "$REPO" 2>/dev/null; then
    ok "Cloned from GitHub"
elif [[ -d "$SOURCE_REPO" ]]; then
    warn "GitHub unreachable — copying from the shared repository instead"
    cp -r "$SOURCE_REPO" "$REPO" || die "Copy failed"
    ok "Copied from $SOURCE_REPO"
else
    die "Could not obtain the code from GitHub or from $SOURCE_REPO"
fi

cd "$REPO" || die "Cannot enter $REPO"
mkdir -p logs
ok "logs/ created (SLURM will not create it for you)"

# -----------------------------------------------------------------------------
# 7. Local paths file
# -----------------------------------------------------------------------------
step "Step 7 of 9 — writing your paths file"

if [[ -f configs/paths.local.yaml ]]; then
    warn "configs/paths.local.yaml already exists — keeping yours"
    warn "Delete it and re-run this script if you want it regenerated"
else
    cat > configs/paths.local.yaml <<YAML_END
# Local paths, generated $(date +%Y-%m-%d) for $ONYEN
# INPUTS point at the shared data and are never written to.
# OUTPUTS point at this user's own folder.
paths:
  data_root:       $DATA
  metadata_csv:    $SOURCE_REPO/release/manifests/metadata.csv
  consistency_csv: $SOURCE_REPO/release/manifests/dl_master_metadata_stage1_step3_consistency.csv

  splits:
    train: release/split_files/train_ids.csv
    val:   release/split_files/val_ids.csv
    test:  release/split_files/test_ids.csv

  results_root:    $MYDIR/runs/results
  logs_root:       $MYDIR/runs/logs
  eccount_out_dir: $MYDIR/runs/eccount_training
YAML_END
    ok "Written to configs/paths.local.yaml"
fi

# -----------------------------------------------------------------------------
# 8. Jupyter kernel
# -----------------------------------------------------------------------------
step "Step 8 of 9 — registering the Jupyter kernel"

PY="$ENV_CANONICAL/bin/python"

if ! PYTHONNOUSERSITE=1 "$PY" -c "import ipykernel" 2>/dev/null; then
    warn "ipykernel not installed in the shared environment."
    warn "Ask the environment owner to run:"
    warn "  PYTHONNOUSERSITE=1 $PY -m pip install 'ipykernel<7'"
elif PYTHONNOUSERSITE=1 "$PY" -m ipykernel install --user \
        --name ecdna-bench --display-name "ecdna-bench (canonical)" >/dev/null 2>&1; then
    ok "Kernel registered as 'ecdna-bench (canonical)'"
else
    warn "Kernel registration failed — you can retry it later"
fi

# -----------------------------------------------------------------------------
# 9. Verify
# -----------------------------------------------------------------------------
step "Step 9 of 9 — checking that everything works"

FAILED=0

if PYTHONNOUSERSITE=1 "$PY" -c "import ecdna_bench" 2>/dev/null; then
    ok "Project code imports"
else
    fail "Project code does not import"; FAILED=1
fi

NPARAM=$(PYTHONNOUSERSITE=1 "$PY" -c "
from ecdna_bench.eccount.model import build_model, ModelConfig
m = build_model(ModelConfig())
print(sum(p.numel() for p in m.parameters() if p.requires_grad))
" 2>/dev/null)
if [[ "$NPARAM" == "7849601" ]]; then
    ok "Model matches the published architecture (7,849,601 parameters)"
else
    fail "Model parameter count is '$NPARAM', expected 7849601"; FAILED=1
fi

RESULTS_ROOT=$(PYTHONNOUSERSITE=1 "$PY" -c "
from ecdna_bench.config import load_config
print(load_config('configs/default.yaml')['paths']['results_root'])
" 2>/dev/null)
if [[ "$RESULTS_ROOT" == *"$ONYEN"* ]]; then
    ok "Output will go to your own folder"
else
    fail "Output would go to '$RESULTS_ROOT' — that is not yours"; FAILED=1
fi

echo "  … running the test suite (about 30 seconds)"
if PYTHONNOUSERSITE=1 "$PY" -m pytest -q >/tmp/ecdna_pytest_$$.log 2>&1; then
    ok "$(tail -1 /tmp/ecdna_pytest_$$.log | tr -d '\n')"
else
    fail "Tests failed — see /tmp/ecdna_pytest_$$.log"; FAILED=1
fi

# -----------------------------------------------------------------------------
echo
echo "${BOLD}=============================================${NC}"
if [[ "$FAILED" -eq 0 ]]; then
    echo "${GREEN}${BOLD}  Setup complete.${NC}"
else
    echo "${YELLOW}${BOLD}  Setup finished with warnings above.${NC}"
fi
echo "${BOLD}=============================================${NC}"
cat <<SUMMARY

  Your workspace : $MYDIR
  Your code      : $REPO
  Environment    : $ENV_CANONICAL

  Load your new settings once:

      source ~/.bashrc

  After that, every session starts with a single word:

      ecdna

  Your first job:

      cd \$REPO
      sbatch slurm/submit_benchmark.sh
      squeue -u \$USER

  Read Part 4 of the tutorial before submitting anything, so you know
  where the output lands.

SUMMARY
