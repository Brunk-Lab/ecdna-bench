#!/bin/bash
# =============================================================================
# ecdna-bench — set up a lab member's account on UNC Longleaf   (version 2)
#
# What changed from version 1, and why
# -------------------------------------
# Version 1 appended exports to ~/.bashrc (PIP_CACHE_DIR, XDG_CACHE_HOME,
# CONDA_PKGS_DIRS, PROJ, DATA, REPO, ...).  Because ~/.bashrc runs in every
# shell, those settings changed the behavior of EVERY conda environment the
# person uses: where conda and pip keep their package caches, where
# matplotlib, Jupyter and other tools keep theirs, and generic variable names
# such as $DATA and $REPO that other projects also use.
#
# Version 2 changes nothing globally:
#   * all settings live in ONE file, <your folder>/ecdna-bench.env, and are
#     applied only when you type `ecdna` (and undone with `ecdna_off`);
#   * every variable is prefixed ECDNA_ so it cannot collide with other work;
#   * the only line added to ~/.bashrc is the `ecdna` shortcut (skip it with
#     --no-bashrc);
#   * the Jupyter kernel carries its own settings, so other kernels are
#     untouched;
#   * nothing is ever installed into the shared environment or into ~/.local.
#
# USAGE
#   bash setup_new_user.sh              set up (asks for your ONYEN)
#   bash setup_new_user.sh --migrate    also remove the version-1 block from ~/.bashrc
#   bash setup_new_user.sh --uninstall  remove the shortcut and the kernel
#   bash setup_new_user.sh --check      only run the checks
#   Options: --onyen NAME  --no-bashrc  --yes (no questions)  --skip-tests
#
# Safe to run more than once.
# =============================================================================

set -uo pipefail

VERSION=2
GREEN=$'\033[0;32m'; RED=$'\033[0;31m'; YELLOW=$'\033[1;33m'
BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'
[[ -t 1 ]] || { GREEN=""; RED=""; YELLOW=""; BLUE=""; BOLD=""; NC=""; }

ok()    { echo "  ${GREEN}ok${NC}   $*"; }
warn()  { echo "  ${YELLOW}note${NC} $*"; }
fail()  { echo "  ${RED}FAIL${NC} $*"; }
step()  { echo; echo "${BOLD}${BLUE}> $*${NC}"; }
die()   { echo; fail "$*"; echo; echo "Setup stopped."; exit 1; }

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
ONYEN_ARG=""; DO_BASHRC=1; ASSUME_YES=0; MODE="install"; MIGRATE=0; RUN_TESTS=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --onyen)      ONYEN_ARG="${2:-}"; shift ;;
        --no-bashrc)  DO_BASHRC=0 ;;
        --yes|-y)     ASSUME_YES=1 ;;
        --migrate)    MIGRATE=1 ;;
        --uninstall)  MODE="uninstall" ;;
        --check)      MODE="check" ;;
        --skip-tests) RUN_TESTS=0 ;;
        -h|--help)    sed -n '2,32p' "$0"; exit 0 ;;
        *)            die "unknown option: $1 (see --help)" ;;
    esac
    shift
done

ask() {  # ask "question" -> returns 0 for yes
    [[ "$ASSUME_YES" -eq 1 ]] && return 0
    [[ -t 0 ]] || return 1
    local reply
    read -r -p "  $1 [Y/n] " reply
    [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

# ---------------------------------------------------------------------------
# Locations (ECDNA_PROJ_ROOT exists only so the script can be tested)
# ---------------------------------------------------------------------------
PROJ="${ECDNA_PROJ_ROOT:-/proj/brunk_ecdna_cv_project}"
SHARED="$PROJ/Poorya"
ENV_CANONICAL="$SHARED/envs/ecdna-bench"
DATA="$SHARED/ecDNA_Data"
SOURCE_REPO="$SHARED/ecdna-bench"
# Benchmark table: the lab copy with /proj file locations if it exists,
# otherwise the released table in the shared repository.
CONS_LAB="$DATA/manifests/dl_master_metadata_stage1_step3_consistency_proj.csv"
CONS_RELEASED="$SOURCE_REPO/release/manifests/dl_master_metadata_stage1_step3_consistency.csv"
if [[ -f "$CONS_LAB" ]]; then CONS_DEFAULT="$CONS_LAB"; else CONS_DEFAULT="$CONS_RELEASED"; fi
GITHUB_HTTPS="https://github.com/Brunk-Lab/ecdna-bench.git"
PY="$ENV_CANONICAL/bin/python"
MARK_BEGIN="# >>> ecdna-bench shortcut (setup v2) >>>"
MARK_END="# <<< ecdna-bench shortcut (setup v2) <<<"
V1_BEGIN="# >>> ecdna-bench settings >>>"
V1_MANUAL="# ecdna-bench project settings"

echo
echo "${BOLD}ecdna-bench account setup, version $VERSION${NC}"

step "1. Who you are"
ONYEN_GUESS="$(whoami)"
if [[ -n "$ONYEN_ARG" ]]; then
    ONYEN="$ONYEN_ARG"
elif [[ -t 0 && "$ASSUME_YES" -eq 0 ]]; then
    read -r -p "  Your ONYEN [${ONYEN_GUESS}]: " ONYEN_INPUT
    ONYEN="${ONYEN_INPUT:-$ONYEN_GUESS}"
else
    ONYEN="$ONYEN_GUESS"
fi
[[ "$ONYEN" =~ ^[A-Za-z0-9_-]+$ ]] || die "'$ONYEN' does not look like an ONYEN."
MYDIR="$PROJ/$ONYEN"
REPO="$MYDIR/repos/ecdna-bench"
ENVFILE="$MYDIR/ecdna-bench.env"
KERNEL_DIR="${JUPYTER_DATA_DIR:-$HOME/.local/share/jupyter}/kernels/ecdna-bench"
ok "ONYEN $ONYEN; your folder is $MYDIR"

# ---------------------------------------------------------------------------
# ~/.bashrc helpers (Python does the editing; a backup is always made first)
# ---------------------------------------------------------------------------
BASHRC="$HOME/.bashrc"
edit_bashrc() {  # edit_bashrc <remove-v2|remove-v1|detect-v1>
    local python_bin="/usr/bin/python3"
    [[ -x "$python_bin" ]] || python_bin="$PY"
    PYTHONNOUSERSITE=1 "$python_bin" - "$BASHRC" "$1" "$MARK_BEGIN" "$MARK_END" \
        "$V1_BEGIN" "$V1_MANUAL" <<'PYEOF'
import re, sys, shutil, time
from pathlib import Path
path, action, b2, e2, b1, manual = sys.argv[1:7]
p = Path(path)
if not p.exists():
    print("absent"); sys.exit(0)
lines = p.read_text().splitlines(keepends=True)

def span(begin, end):
    try:
        i = next(k for k, l in enumerate(lines) if l.strip() == begin)
        j = next(k for k in range(i, len(lines)) if lines[k].strip() == end)
        return i, j
    except StopIteration:
        return None

KNOWN = re.compile(
    r"^\s*(#.*|export (ONYEN|PROJ|MYDIR|REPO|ENV_CANONICAL|DATA|SOURCE_REPO|PIP_CACHE_DIR|"
    r"XDG_CACHE_HOME|CONDA_PKGS_DIRS|ECDNA_PYTHON)=.*|alias ecdna=.*|)\s*$")

def manual_span():
    for i, l in enumerate(lines):
        if l.strip().startswith(manual):
            start = i - 1 if i > 0 and lines[i - 1].strip().startswith("# ====") else i
            for j in range(i + 1, len(lines)):
                if not KNOWN.match(lines[j]):
                    return ("unexpected", j + 1)
                if lines[j].strip().startswith("alias ecdna="):
                    return (start, j)
            return ("unterminated", i + 1)
    return None

def backup():
    dest = p.with_name(p.name + ".backup-" + time.strftime("%Y%m%d-%H%M%S"))
    shutil.copy2(p, dest)
    return dest

if action == "detect-v1":
    found = []
    if any(l.strip() == b1 for l in lines):
        found.append("v1-script")
    m = manual_span()
    if m:
        found.append("v1-manual" if isinstance(m[0], int) else f"v1-manual-{m[0]}-line{m[1]}")
    print(",".join(found) or "none")
elif action in ("remove-v1", "remove-v2"):
    spans = []
    if action == "remove-v2":
        s = span(b2, e2)
        if s: spans.append(s)
    else:
        s = span(b1, "# <<< ecdna-bench settings <<<")
        if s: spans.append(s)
        m = manual_span()
        if m and not isinstance(m[0], int):
            print(f"refused: the hand-added block has an unexpected line near line {m[1]}; edit ~/.bashrc by hand")
            sys.exit(3)
        if m: spans.append(m)
    if not spans:
        print("nothing-to-remove"); sys.exit(0)
    b = backup()
    for i, j in sorted(spans, reverse=True):
        # also drop one blank line left before the block
        if i > 0 and not lines[i - 1].strip():
            i -= 1
        del lines[i:j + 1]
    p.write_text("".join(lines))
    print(f"removed {len(spans)} block(s); backup {b}")
PYEOF
}

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if [[ "$MODE" == "uninstall" ]]; then
    step "Removing the ecdna-bench shortcut and kernel"
    res="$(edit_bashrc remove-v2)"; ok "~/.bashrc: $res"
    if [[ -f "$KERNEL_DIR/kernel.json" ]] && grep -q "$ENV_CANONICAL" "$KERNEL_DIR/kernel.json"; then
        rm -rf "$KERNEL_DIR" && ok "Jupyter kernel 'ecdna-bench' removed"
    else
        warn "no ecdna-bench kernel of ours to remove"
    fi
    warn "Your folder $MYDIR (clone, runs, $ENVFILE) was left in place."
    exit 0
fi

# ---------------------------------------------------------------------------
# 2. Shared storage
# ---------------------------------------------------------------------------
step "2. Access to the lab storage"
[[ -d "$PROJ" ]] || die "Cannot see $PROJ. Ask Research Computing (https://help.rc.unc.edu) to add you to the brunk_ecdna_cv_project group."
ok "lab storage reachable"
[[ -x "$PY" ]] || die "shared environment not found at $ENV_CANONICAL"
ok "shared environment: $ENV_CANONICAL"
[[ -r "$DATA" ]] && ok "image data readable" || warn "cannot read $DATA (ask about group permissions)"
[[ -r "$SOURCE_REPO" ]] && ok "shared repository readable" || warn "cannot read $SOURCE_REPO; the code will come from GitHub"

# ---------------------------------------------------------------------------
# 3. Old (version 1) settings
# ---------------------------------------------------------------------------
step "3. Settings left by the old setup"
V1_STATE="$(edit_bashrc detect-v1)"
case "$V1_STATE" in
    none|absent) ok "no old settings in ~/.bashrc" ;;
    *)
        warn "found old ecdna-bench settings in ~/.bashrc ($V1_STATE)."
        warn "They change conda/pip cache locations for all your environments."
        if [[ "$MIGRATE" -eq 1 ]] || ask "Remove them now? (a backup of ~/.bashrc is kept)"; then
            res="$(edit_bashrc remove-v1)"
            if [[ "$res" == refused* ]]; then
                fail "$res"
            else
                ok "$res"
                warn "Open a NEW terminal after setup so the old values are gone."
            fi
        else
            warn "left in place; re-run with --migrate to remove them"
        fi
        ;;
esac
if [[ -n "${XDG_CACHE_HOME:-}" && "${XDG_CACHE_HOME}" == "$MYDIR"* ]]; then
    warn "this shell still has XDG_CACHE_HOME=$XDG_CACHE_HOME from the old settings (open a new terminal)"
fi

[[ "$MODE" == "check" ]] || {
# ---------------------------------------------------------------------------
# 4. Workspace
# ---------------------------------------------------------------------------
step "4. Your workspace"
mkdir -p "$MYDIR"/{repos,runs,logs} || die "could not create $MYDIR"
ok "$MYDIR/{repos,runs,logs}"

# ---------------------------------------------------------------------------
# 5. The code (your own clone)
# ---------------------------------------------------------------------------
step "5. Your copy of the code"
export GIT_TERMINAL_PROMPT=0
if [[ -d "$REPO/.git" ]]; then
    ok "already present: $REPO"
elif [[ -d "$SOURCE_REPO/.git" ]] && git clone --quiet "$SOURCE_REPO" "$REPO" 2>/dev/null; then
    ok "cloned from the shared repository (full history)"
elif git clone --quiet "$GITHUB_HTTPS" "$REPO" 2>/dev/null; then
    ok "cloned from GitHub"
else
    die "could not obtain the code from $SOURCE_REPO or $GITHUB_HTTPS"
fi
mkdir -p "$REPO/logs"

# ---------------------------------------------------------------------------
# 6. Local paths file (outputs go to your folder)
# ---------------------------------------------------------------------------
step "6. configs/paths.local.yaml in your copy"
if [[ -f "$REPO/configs/paths.local.yaml" ]]; then
    ok "already present (kept as is)"
else
    cat > "$REPO/configs/paths.local.yaml" <<YAML_END
# Local paths for $ONYEN, written by setup_new_user.sh v$VERSION on $(date +%Y-%m-%d).
# Inputs are shared and read-only. Outputs go to your own folder.
paths:
  data_root:       $DATA
  metadata_csv:    $SOURCE_REPO/release/manifests/metadata.csv
  consistency_csv: $CONS_DEFAULT
  splits:
    train: release/split_files/train_ids.csv
    val:   release/split_files/val_ids.csv
    test:  release/split_files/test_ids.csv
  results_root:       $MYDIR/runs/results
  logs_root:          $MYDIR/runs/logs
  frozen_results_dir: $MYDIR/runs/benchmark
  eccount_out_dir:    $MYDIR/runs/eccount_training
YAML_END
    CKPT_REL="release/model_checkpoints/eccount_best.pt"
    if [[ ! -f "$REPO/$CKPT_REL" && -f "$SOURCE_REPO/$CKPT_REL" ]]; then
        echo "  eccount_checkpoint: $SOURCE_REPO/$CKPT_REL" >> "$REPO/configs/paths.local.yaml"
    fi
    ok "written"
fi

# ---------------------------------------------------------------------------
# 7. The session file and the `ecdna` shortcut
# ---------------------------------------------------------------------------
step "7. Session settings (applied only when you type 'ecdna')"
cat > "$ENVFILE" <<ENV_END
# ecdna-bench session settings for $ONYEN (setup_new_user.sh v$VERSION).
# Use:  source $ENVFILE     (the 'ecdna' shortcut does this)
# Undo: ecdna_off
# Nothing here is applied unless you source this file.

if [[ -n "\${ECDNA_ACTIVE:-}" ]]; then
    echo "ecdna-bench session already active (type ecdna_off to leave)"
    cd "\$ECDNA_REPO" 2>/dev/null
    return 0 2>/dev/null || exit 0
fi

export ECDNA_ONYEN="$ONYEN"
export ECDNA_PROJ="$PROJ"
export ECDNA_MYDIR="$MYDIR"
export ECDNA_REPO="$REPO"
export ECDNA_ENV="$ENV_CANONICAL"
export ECDNA_DATA="$DATA"
export ECDNA_SOURCE_REPO="$SOURCE_REPO"

# Remember what we change so ecdna_off can put it back.
_ECDNA_OLD_PATH="\$PATH"
_ECDNA_OLD_PYTHONPATH="\${PYTHONPATH-__unset__}"
_ECDNA_OLD_PYTHONNOUSERSITE="\${PYTHONNOUSERSITE-__unset__}"
_ECDNA_OLD_PWD="\$PWD"
_ECDNA_CONDA=0

if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load anaconda >/dev/null 2>&1
fi
if command -v conda >/dev/null 2>&1; then
    eval "\$(conda shell.bash hook 2>/dev/null)" >/dev/null 2>&1
    if conda activate "\$ECDNA_ENV" >/dev/null 2>&1; then
        _ECDNA_CONDA=1
    fi
fi
if [[ "\$_ECDNA_CONDA" -ne 1 ]]; then
    export PATH="\$ECDNA_ENV/bin:\$PATH"   # fallback when conda cannot activate
fi

# Your own copy of the code wins over the shared installation, and packages
# in ~/.local never shadow the shared environment (and pip cannot fall back
# to installing into ~/.local, which would affect your other environments).
export PYTHONPATH="\$ECDNA_REPO/src\${PYTHONPATH:+:\$PYTHONPATH}"
export PYTHONNOUSERSITE=1

# Used by the SLURM scripts in slurm/ (sbatch passes these to the job).
export ECDNA_PYTHON="\$ECDNA_ENV/bin/python"
export ECDNA_PROJECT_ROOT="\$ECDNA_REPO"
export ECDNA_ACTIVE=1

ecdna_off() {
    if [[ "\$_ECDNA_CONDA" -eq 1 ]]; then conda deactivate >/dev/null 2>&1; fi
    export PATH="\$_ECDNA_OLD_PATH"
    if [[ "\$_ECDNA_OLD_PYTHONPATH" == "__unset__" ]]; then unset PYTHONPATH; else export PYTHONPATH="\$_ECDNA_OLD_PYTHONPATH"; fi
    if [[ "\$_ECDNA_OLD_PYTHONNOUSERSITE" == "__unset__" ]]; then unset PYTHONNOUSERSITE; else export PYTHONNOUSERSITE="\$_ECDNA_OLD_PYTHONNOUSERSITE"; fi
    unset ECDNA_PYTHON ECDNA_PROJECT_ROOT ECDNA_ACTIVE ECDNA_ONYEN ECDNA_PROJ ECDNA_MYDIR ECDNA_REPO ECDNA_ENV ECDNA_DATA ECDNA_SOURCE_REPO
    cd "\$_ECDNA_OLD_PWD" 2>/dev/null
    unset -f ecdna_off
    echo "ecdna-bench session closed"
}

cd "\$ECDNA_REPO" && echo "ecdna-bench session: \$(python --version 2>&1) from \$ECDNA_ENV; code from \$ECDNA_REPO (ecdna_off to leave)"
ENV_END
ok "written: $ENVFILE"

if [[ "$DO_BASHRC" -eq 1 ]]; then
    edit_bashrc remove-v2 >/dev/null
    {
        echo ""
        echo "$MARK_BEGIN"
        echo "# One shortcut only; it changes nothing until you type 'ecdna'."
        echo "alias ecdna='source \"$ENVFILE\"'"
        echo "$MARK_END"
    } >> "$BASHRC"
    ok "added the 'ecdna' shortcut to ~/.bashrc (nothing else)"
else
    warn "--no-bashrc: start a session with   source $ENVFILE"
fi

# ---------------------------------------------------------------------------
# 8. Jupyter kernel with its own settings
# ---------------------------------------------------------------------------
step "8. Jupyter kernel 'ecdna-bench (canonical)'"
if ! PYTHONNOUSERSITE=1 "$PY" -c "import ipykernel" 2>/dev/null; then
    warn "ipykernel is not installed in the shared environment; ask the environment owner"
elif PYTHONNOUSERSITE=1 "$PY" -m ipykernel install --user --name ecdna-bench \
        --display-name "ecdna-bench (canonical)" >/dev/null 2>&1; then
    PYTHONNOUSERSITE=1 "$PY" - "$KERNEL_DIR/kernel.json" "$REPO/src" <<'PYEOF'
import json, sys
path, src = sys.argv[1], sys.argv[2]
with open(path) as fh:
    spec = json.load(fh)
env = spec.setdefault("env", {})
env["PYTHONNOUSERSITE"] = "1"
env["PYTHONPATH"] = src
with open(path, "w") as fh:
    json.dump(spec, fh, indent=1)
PYEOF
    ok "kernel registered (settings stored in the kernel only): $KERNEL_DIR"
else
    warn "kernel registration failed; re-run this script later"
fi
}

# ---------------------------------------------------------------------------
# 9. Checks (run in a clean sub-shell, exactly as an 'ecdna' session)
# ---------------------------------------------------------------------------
step "9. Checks"
[[ -f "$ENVFILE" ]] || die "$ENVFILE is missing; run without --check first"
CHECK_OUT="$(env -i HOME="$HOME" USER="${USER:-$ONYEN}" PATH="/usr/bin:/bin" TERM=dumb \
    ECDNA_PROJ_ROOT="${ECDNA_PROJ_ROOT:-}" bash --noprofile --norc -c "
source '$ENVFILE' >/dev/null 2>&1
echo \"PY=\$(command -v python)\"
python - <<'PYEOF'
import sys, os
print('SITE_OK=%s' % (not any('.local' in p for p in sys.path)))
try:
    import ecdna_bench
    print('IMPORT_FROM=%s' % os.path.dirname(ecdna_bench.__file__))
except Exception as exc:
    print('IMPORT_FAIL=%s' % exc)
try:
    from ecdna_bench.eccount.model import build_model, ModelConfig
    m = build_model(ModelConfig())
    print('NPARAM=%d' % sum(p.numel() for p in m.parameters() if p.requires_grad))
except Exception as exc:
    print('NPARAM_FAIL=%s' % exc)
try:
    from ecdna_bench.cli._common import load_config
    p = load_config('configs/default.yaml').get('paths', {})
    for k in ('results_root', 'logs_root', 'frozen_results_dir', 'eccount_out_dir', 'consistency_csv'):
        print('CFG_%s=%s' % (k, p.get(k) or ''))
    ck = p.get('eccount_checkpoint') or ''
    print('CKPT=%s' % (os.path.abspath(ck) if ck else ''))
    print('CKPT_OK=%s' % bool(ck and os.path.isfile(ck)))
    # Can this account read the image files the benchmark table points at?
    import csv
    checked = readable = 0
    first_bad = ''
    with open(p.get('consistency_csv') or '', newline='') as fh:
        for i, row in enumerate(csv.DictReader(fh)):
            if i >= 5:
                break
            for col in ('rgb_fullpath', 'gt_fullpath', 'roi_fullpath'):
                path = (row.get(col) or '').strip()
                if not path:
                    continue
                checked += 1
                if os.access(path, os.R_OK):
                    readable += 1
                elif not first_bad:
                    first_bad = path
    print('IMG_CHECKED=%d' % checked)
    print('IMG_READABLE=%d' % readable)
    print('IMG_FIRST_BAD=%s' % first_bad)
except Exception as exc:
    print('CFG_FAIL=%s' % exc)
PYEOF
" 2>&1)"
FAILED=0
PY_USED="$(sed -n 's/^PY=//p' <<<"$CHECK_OUT")"
if [[ "$PY_USED" == "$ENV_CANONICAL/bin/python"* ]]; then ok "session uses the shared environment"; else fail "session python is '$PY_USED'"; FAILED=1; fi
grep -q '^SITE_OK=True' <<<"$CHECK_OUT" && ok "~/.local packages are ignored in the session" || { fail "~/.local is on sys.path"; FAILED=1; }
FROM="$(sed -n 's/^IMPORT_FROM=//p' <<<"$CHECK_OUT")"
if [[ "$FROM" == "$REPO/src/"* ]]; then ok "code is imported from your copy ($FROM)"
elif [[ -n "$FROM" ]]; then fail "code is imported from $FROM, not from your copy"; FAILED=1
else fail "project code does not import: $(sed -n 's/^IMPORT_FAIL=//p' <<<"$CHECK_OUT")"; FAILED=1; fi
NPARAM="$(sed -n 's/^NPARAM=//p' <<<"$CHECK_OUT")"
[[ "$NPARAM" == "7849601" ]] && ok "ecCount builds with 7,849,601 parameters" || { fail "ecCount parameter count '$NPARAM'"; FAILED=1; }
BAD=0
for KEY in results_root logs_root frozen_results_dir eccount_out_dir; do
    VAL="$(sed -n "s/^CFG_${KEY}=//p" <<<"$CHECK_OUT")"
    [[ -z "$VAL" || "$VAL" != /* ]] && continue
    if [[ "$VAL" != "$MYDIR"* ]]; then fail "paths.$KEY writes to $VAL (outside your folder)"; BAD=1; FAILED=1; fi
done
[[ "$BAD" -eq 0 ]] && ok "every output path is inside $MYDIR"
CONS="$(sed -n 's/^CFG_consistency_csv=//p' <<<"$CHECK_OUT")"
[[ -n "$CONS" && -r "$CONS" ]] && ok "benchmark table readable" || warn "benchmark table not readable: '$CONS'"
IMG_CHECKED="$(sed -n 's/^IMG_CHECKED=//p' <<<"$CHECK_OUT")"
IMG_READABLE="$(sed -n 's/^IMG_READABLE=//p' <<<"$CHECK_OUT")"
if [[ -n "$IMG_CHECKED" && "$IMG_CHECKED" -gt 0 ]]; then
    if [[ "$IMG_READABLE" == "$IMG_CHECKED" ]]; then
        ok "image files listed in the benchmark table are readable (sample of $IMG_CHECKED)"
    else
        warn "only $IMG_READABLE of $IMG_CHECKED sampled image paths are readable, e.g. $(sed -n 's/^IMG_FIRST_BAD=//p' <<<"$CHECK_OUT")"
        warn "benchmark and training jobs will fail until these paths are readable"
        if [[ -f "$CONS_LAB" && "$CONS" != "$CONS_LAB" ]]; then
            warn "fix: in $REPO/configs/paths.local.yaml set   consistency_csv: $CONS_LAB"
        else
            warn "fix: ask the data owner (see scripts/rewrite_manifest_paths.py)"
        fi
    fi
fi
CKPT_PATH="$(sed -n 's/^CKPT=//p' <<<"$CHECK_OUT")"
if grep -q '^CKPT_OK=True' <<<"$CHECK_OUT"; then
    ok "ecCount weights found: $CKPT_PATH"
else
    warn "ecCount weights not found at '$CKPT_PATH' (needed only to run ecCount);"
    warn "add 'eccount_checkpoint: <path to eccount_best.pt>' under paths: in $REPO/configs/paths.local.yaml"
fi
CFG_ERR="$(sed -n 's/^CFG_FAIL=//p' <<<"$CHECK_OUT")"
[[ -z "$CFG_ERR" ]] || warn "configuration check: $CFG_ERR"
if [[ -f "$KERNEL_DIR/kernel.json" ]] && grep -q PYTHONNOUSERSITE "$KERNEL_DIR/kernel.json"; then
    ok "Jupyter kernel carries its own settings"
else
    warn "Jupyter kernel not registered with settings"
fi
if [[ "$RUN_TESTS" -eq 1 && "$MODE" != "check" ]]; then
    TLOG="$MYDIR/logs/setup_pytest_$(date +%Y%m%d-%H%M%S).log"
    if (cd "$REPO" && env -i HOME="$HOME" PATH="/usr/bin:/bin" ECDNA_PROJ_ROOT="${ECDNA_PROJ_ROOT:-}" \
            bash --noprofile --norc -c "source '$ENVFILE' >/dev/null 2>&1 && python -m pytest -q" >"$TLOG" 2>&1); then
        ok "test suite: $(tail -1 "$TLOG")"
    else
        fail "test suite failed; see $TLOG"; FAILED=1
    fi
fi

echo
if [[ "$FAILED" -eq 0 ]]; then
    echo "${GREEN}${BOLD}Setup complete.${NC}"
else
    echo "${YELLOW}${BOLD}Setup finished with problems (see FAIL lines above).${NC}"
fi
cat <<SUMMARY

  Your folder : $MYDIR
  Your code   : $REPO
  Session file: $ENVFILE

  Open a NEW terminal, then start every working session with:

      ecdna

  and leave it with:

      ecdna_off

  Nothing else in your account was changed.
SUMMARY
exit "$FAILED"
