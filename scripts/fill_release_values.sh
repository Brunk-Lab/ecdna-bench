#!/usr/bin/env bash
# =============================================================================
# fill_release_values.sh
# =============================================================================
# Substitutes the release-time values that are not known while the repository
# is being prepared. Run this from the repo root, ONCE, immediately before
# tagging v1.0.0 — after the BioImage Archive deposition has issued an
# accession, and after the Zenodo DOI (if any) has been minted.
#
# Usage:
#   1. Edit the six values below.
#   2. bash scripts/fill_release_values.sh
#   3. Re-run the marker sweep (printed at the end) — it must return nothing.
#
# The script is idempotent for tokens that have already been replaced: a
# second run simply finds nothing to substitute.
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# EDIT THESE
# ---------------------------------------------------------------------------
ACCESSION="S-BIAD#####"                       # BioImage Archive accession
ZENODO_DOI="10.5281/zenodo.XXXXXXX"           # after minting the release DOI
PAPER_DOI="10.1038/s43588-XXX-XXXXX-X"        # after journal acceptance
RELEASE_DATE="2026-09-XX"                     # YYYY-MM-DD of the v1.0.0 tag
BRUNK_ORCID="0000-0000-0000-0000"             # PI's ORCID (digits only)

# ---------------------------------------------------------------------------
# Files that contain tokens
# ---------------------------------------------------------------------------
FILES=(
  "README.md"
  "DATASET.md"
  "CHANGELOG.md"
  "CITATION.cff"
  "pyproject.toml"
  "docs/EXTERNAL_BASELINES.md"
)

# ---------------------------------------------------------------------------
# Substitute
# ---------------------------------------------------------------------------
for f in "${FILES[@]}"; do
  [[ -f "$f" ]] || { echo "skip (missing): $f"; continue; }
  sed -i \
    -e "s|{{ACCESSION}}|${ACCESSION}|g" \
    -e "s|{{ZENODO_DOI}}|${ZENODO_DOI}|g" \
    -e "s|{{PAPER_DOI}}|${PAPER_DOI}|g" \
    -e "s|{{RELEASE_DATE}}|${RELEASE_DATE}|g" \
    -e "s|{{BRUNK_ORCID}}|${BRUNK_ORCID}|g" \
    "$f"
  echo "filled: $f"
done

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------
echo
echo "Remaining tokens (this list must be empty):"
grep -rn "{{[A-Z_]*}}" . \
  --include="*.md" --include="*.toml" --include="*.cff" \
  --include="*.py" --include="*.yaml" --include="*.sh" \
  || echo "  none — good."

echo
echo "Full marker sweep:"
grep -rn "<<FILL\|make_figures\|TODO\|FIXME\|XXX\|\[REF:\|\[CITE:\|{{" . \
  --include="*.md" --include="*.toml" --include="*.cff" \
  --include="*.py" --include="*.yaml" \
  || echo "  clean — ready to tag."
