#!/bin/bash
# =============================================================================
# slurm/submit_bia_upload.sh
# =============================================================================
# Upload the ecdna-bench deposition into the BioStudies private FTP staging
# area, ready for submission through the BioImage Archive web form.
#
# Resumable by design.  `mirror --continue --only-newer` re-transfers only what
# is missing or partial, so if this job hits the wall clock, the connection
# drops, or you simply want to confirm completeness, resubmit the identical
# script.  Nothing is re-sent unnecessarily.
#
# Credentials and the secret directory live in ~/.bia_ftp_login (mode 600) and
# are never passed as command-line arguments, because arguments are visible in
# `ps` to every user on the node, and the secret directory is the only thing
# protecting the staging area (the FTP username and password are shared by
# every BioStudies submitter).
#
# ~/.bia_ftp_login must contain, with your own values:
#
#     open ftp-private.ebi.ac.uk
#     user "bs-upload" "PASSWORD"
#     cd /f8/YOUR-SECRET-DIRECTORY
#
# The `cd` line is mandatory.  Without it the transfer would land at the FTP
# root instead of inside your submission space.
#
# Usage
# -----
#   mkdir -p logs
#   sbatch slurm/submit_bia_upload.sh
#   tail -f logs/bia_upload_<jobid>.out
# =============================================================================

#SBATCH --job-name=bia_upload
#SBATCH --partition=datamover
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/bia_upload_%j.out
#SBATCH --error=logs/bia_upload_%j.err

set -eo pipefail          # not -u; it has broken the conda hook before

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA=/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data
REPO=/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench
LOGIN="$HOME/.bia_ftp_login"

# Parallel streams. Four is a reasonable balance: enough to fill a transatlantic
# link, few enough that the server does not throttle or drop connections.
PARALLEL=4

# -----------------------------------------------------------------------------
# Preflight - fail in seconds, not hours
# -----------------------------------------------------------------------------

echo "=== preflight ==="
date

command -v lftp >/dev/null || { echo "FATAL: lftp not found" >&2; exit 1; }
echo "lftp: $(command -v lftp)"

if [[ ! -f "$LOGIN" ]]; then
    echo "FATAL: $LOGIN does not exist. See the header of this script." >&2
    exit 1
fi

perms=$(stat -c '%a' "$LOGIN")
if [[ "$perms" != "600" ]]; then
    echo "FATAL: $LOGIN has mode $perms; must be 600. Run: chmod 600 $LOGIN" >&2
    exit 1
fi

if grep -q 'FILL_IN\|YOUR-SECRET' "$LOGIN"; then
    echo "FATAL: $LOGIN still contains placeholder values." >&2
    exit 1
fi

# The cd into the secret directory is not optional; without it everything
# would be written to the FTP root and the submission would find nothing.
if ! grep -qE '^[[:space:]]*cd[[:space:]]+/' "$LOGIN"; then
    echo "FATAL: $LOGIN has no 'cd /<secret directory>' line." >&2
    exit 1
fi
echo "login file: ok (mode 600, has open/user/cd)"

for d in "$DATA/bioimage_archive" "$DATA/benchmark/predictions" "$REPO/release/split_files"; do
    [[ -d "$d" ]] || { echo "FATAL: missing source directory $d" >&2; exit 1; }
    echo "source ok: $d"
done

echo "local census (apparent size; /proj compresses, so plain du misleads):"
du --apparent-size -sh "$DATA/bioimage_archive" "$DATA/benchmark/predictions"

# -----------------------------------------------------------------------------
# Build the lftp command file
# -----------------------------------------------------------------------------
# Order matters: settings first so that cmd:fail-exit is already active when
# the open/user/cd lines from the login file run. A failed cd must abort the
# job rather than silently uploading to the wrong place.

umask 077
CMDS=$(mktemp "${TMPDIR:-/tmp}/bia_upload_cmds.XXXXXX")
trap 'rm -f "$CMDS"' EXIT

{
    cat <<'LFTP'
set cmd:fail-exit yes
set net:max-retries 5
set net:reconnect-interval-base 15
set net:timeout 60
set xfer:clobber on
LFTP
    cat "$LOGIN"
    echo "pwd"
    echo "mirror -R --continue --only-newer --no-perms --parallel=$PARALLEL --verbose $DATA/bioimage_archive images"
    echo "mirror -R --continue --only-newer --no-perms --parallel=$PARALLEL --verbose $DATA/benchmark/predictions predictions"
    echo "mirror -R --continue --only-newer --no-perms --verbose --include-glob *_ids.csv $REPO/release/split_files splits"
    echo "bye"
} > "$CMDS"

# -----------------------------------------------------------------------------
# Transfer
# -----------------------------------------------------------------------------

echo
echo "=== transfer starting ==="
date
SECONDS=0

set +e
lftp -f "$CMDS"
rc=$?
set -e

echo
echo "=== transfer finished ==="
date
printf 'elapsed: %02d:%02d:%02d\n' $((SECONDS/3600)) $(((SECONDS/60)%60)) $((SECONDS%60))
echo "lftp exit status: $rc"

if [[ $rc -ne 0 ]]; then
    echo "Transfer did not complete cleanly. Resubmit this identical script;" >&2
    echo "--continue --only-newer resumes rather than restarts." >&2
fi

exit $rc
