"""
ecdna_bench.data.ids — filename → unique-identifier normalization.

The released dataset ships with a `metadata.csv` that already contains
canonical UIDs, so most user-facing workflows never touch this module.
However, when rebuilding the metadata table from raw files (or when mapping
externally-produced prediction masks to UIDs) we need the original filename
logic. This module preserves that logic from the legacy `id_extraction.py`.

Two naming styles are supported:

1. **counts-only style** — simple trailing-suffix stripping (`_Merge.tif`,
   `_DAPI.tif`, etc.). Used for the full resource of n = 2,984 images.

2. **localized-GT style** — normalized UID (lowercased, special tokens
   stripped) plus a strict Jaccard-on-tokens matcher. Used for the 1,145
   benchmark-subset images that have manual ROI masks and point annotations.

Cell lines
----------
Five cell lines are supported: NCI-H2170, NCI-H716, SNU16, COLO320DM,
SUM159PT. An alias table maps common spellings (lowercase, hyphenated,
etc.) to their canonical uppercase token. `infer_cell_line(uid)` returns
the canonical name, or `None` if no match is found.
"""

from __future__ import annotations

import os
import re
from typing import Literal

DatasetType = Literal["counts_only", "localized_gt"]


# ==============================================================================
# Constants
# ==============================================================================

SIDE_TAGS: frozenset[str] = frozenset({"a", "b"})

# Canonical cell-line tokens. These are the strings that appear in the
# released `metadata.csv` `cell_line` column. Note these use NO hyphens
# (NCIH2170, not NCI-H2170) to make string comparison trivial.
KNOWN_CELL_LINES: frozenset[str] = frozenset(
    {"COLO320DM", "COLO320HSR", "NCIH2170", "SNU16", "SKGT2", "NCIH716", "SUM159PT"}
)

# Lowercase aliases → canonical name. Populated with every spelling we have
# seen in filenames across the 2,984-image dataset. Add to this table rather
# than to `infer_cell_line`.
CELL_LINE_ALIASES: dict[str, str] = {
    "colo320dm": "COLO320DM",
    "colo320hsr": "COLO320HSR",
    "ncih2170": "NCIH2170",
    "snu16": "SNU16",
    "skgt2": "SKGT2",
    "ncih716": "NCIH716",
    "sum159pt": "SUM159PT",
    "sum159": "SUM159PT",
}

VALID_EXTS: frozenset[str] = frozenset(
    {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp"}
)

# Minimum Jaccard on token sets for the localized-GT matcher to accept a
# cross-modality file as belonging to the same metaphase spread.
STRICT_JACCARD_MIN: float = 0.95


# ==============================================================================
# Suffixes used by the counts-only style
# ==============================================================================

COUNTS_DAPI_SUFFIXES: tuple[str, ...] = (
    "_DAPI",
    "_DAPI.tif",
    "_Merge.tif (RGB)",
    "_Merge.tif(RGB)",
    "_Merge.tif (RGB).tif",
    "_Merge.tif(RGB).tif",
)

# Extended set — covers edge cases seen in the long tail of the 2,984-image
# resource (naming inconsistencies from different acquisition sessions).
COUNTS_DAPI_SUFFIXES_EXTENDED: tuple[str, ...] = (
    "_Mask.tif",
    "_DAPI_ROI.tif",
    "_Merge.tif (RGB)",
    "_DAPI",
    "_DAPI.tif",
    "_Merge.tif(RGB)",
    "_Merge.tif (RGB).tif",
    "_Merge.tif(RGB).tif",
    "_Merge(RGB)",
    "_Merge (RGB)",
    "_Merge(RGB).tif",
    "_Merge (RGB).tif",
    "_Merge",
)

# Suffixes stripped by the localized-GT normalizer.
_LOCALIZED_STRIP_SUFFIXES: tuple[str, ...] = (
    "_dapi",
    "_merge",
    "_roi",
    "_mask",
    "_fitc",
    "_nucleus",
    "_rgb",
    "_gt",
    "_le",
)


# ==============================================================================
# Low-level helpers
# ==============================================================================


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.lower() == "thumbs.db"


def _is_image_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in VALID_EXTS


def _tokens(s: str) -> list[str]:
    """Split on underscores / whitespace, drop empty fragments, lowercase."""
    return [t for t in re.split(r"[_\s]+", s.lower()) if t]


def _side_tag_from_tokens(tokens: list[str]) -> str | None:
    """Return 'a' or 'b' if it appears among the last three tokens."""
    for t in reversed(tokens[-3:]):
        if t in SIDE_TAGS:
            return t
    return None


def _last_numeric_token(tokens: list[str]) -> str | None:
    for t in reversed(tokens):
        if t.isdigit():
            return t
    return None


def _norm_alnum(s: str) -> str:
    """Lowercase + strip everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ==============================================================================
# Counts-only naming style
# ==============================================================================


def extract_uid_counts_only(filename: str, suffixes: tuple[str, ...]) -> str:
    """
    Strip a known suffix from the end of `filename` (without its extension).

    Only strips from the very end; otherwise returns the base name unchanged.

    Example:
        extract_uid_counts_only("Sample1_Merge.tif", ("_Merge",)) == "Sample1"
    """
    base = os.path.splitext(filename)[0]
    for suf in suffixes:
        if base.endswith(suf):
            return base[: -len(suf)]
    return base


def find_corresponding_dapi_counts(unique_id: str, dapi_folder: str) -> str | None:
    """
    Locate the DAPI file for a given RGB UID using the strict suffix list.

    Returns the first path whose UID matches, or None.
    """
    return _find_corresponding_by_suffix(unique_id, dapi_folder, COUNTS_DAPI_SUFFIXES)


def find_corresponding_dapi_counts_extended(unique_id: str, dapi_folder: str) -> str | None:
    """Same as `find_corresponding_dapi_counts`, using the extended suffix list."""
    return _find_corresponding_by_suffix(unique_id, dapi_folder, COUNTS_DAPI_SUFFIXES_EXTENDED)


def _find_corresponding_by_suffix(
    unique_id: str, folder: str, suffixes: tuple[str, ...]
) -> str | None:
    if not folder or not os.path.isdir(folder):
        return None
    for fname in os.listdir(folder):
        if _is_hidden(fname) or not _is_image_file(fname):
            continue
        uid = extract_uid_counts_only(fname, suffixes)
        if uid == unique_id:
            return os.path.join(folder, fname)
    return None


# ==============================================================================
# Localized-GT naming style
# ==============================================================================


def extract_uid_localized(filename: str) -> str:
    """
    Normalize a filename into a localized-GT UID.

    Steps
    -----
    1. Drop the extension and lowercase.
    2. Remove parenthesized content (e.g., ` (RGB)`).
    3. Replace hyphens and whitespace with underscores; collapse runs.
    4. Strip known channel / tag suffixes from the end, repeating until no
       suffix is removed (handles chains like `..._merge_roi`).
    5. Also remove literal `merge` / `dapi` anywhere in the name to avoid
       biasing the token set.
    6. Collapse repeated underscores one more time.
    """
    name = os.path.splitext(os.path.basename(filename))[0].lower()
    name = re.sub(r"\(.*?\)", "", name)
    name = re.sub(r"[-\s]", "_", name)
    name = re.sub(r"__+", "_", name).strip("_")

    changed = True
    while changed and name:
        changed = False
        for suf in _LOCALIZED_STRIP_SUFFIXES:
            if name.endswith(suf):
                name = name[: -len(suf)].rstrip("_")
                changed = True

    name = name.replace("merge", "").replace("dapi", "")
    name = re.sub(r"__+", "_", name).strip("_")
    return name


def find_corresponding_file_localized(
    uid: str,
    folder: str,
    jaccard_min: float = STRICT_JACCARD_MIN,
) -> str | None:
    """
    Strict cross-modality matcher for the localized-GT style.

    Requirements
    ------------
    - If the query UID has an A/B side tag, the candidate must have the same tag.
    - If the query UID contains a last numeric token (typical metaphase index),
      the candidate must have an identical last numeric token.
    - Token-set Jaccard must be >= `jaccard_min` (default 0.95).

    If multiple candidates satisfy all constraints, the one with the highest
    Jaccard score (minus a small length-mismatch penalty) wins. Returns the
    absolute path of the best candidate, or None if nothing qualifies.
    """
    if not folder or not os.path.isdir(folder):
        return None

    uid_norm = extract_uid_localized(uid)
    uid_tok = _tokens(uid_norm)
    set_uid = set(uid_tok)
    uid_side = _side_tag_from_tokens(uid_tok)
    uid_last_num = _last_numeric_token(uid_tok)

    best_name: str | None = None
    best_score: float = -1.0

    for fn in os.listdir(folder):
        if _is_hidden(fn) or not _is_image_file(fn):
            continue

        fn_uid = extract_uid_localized(fn)
        if fn_uid == uid_norm:
            return os.path.join(folder, fn)  # exact hit

        fn_tok = _tokens(fn_uid)
        set_fn = set(fn_tok)

        if uid_side and _side_tag_from_tokens(fn_tok) != uid_side:
            continue

        fn_last_num = _last_numeric_token(fn_tok)
        if uid_last_num is not None and fn_last_num != uid_last_num:
            continue

        union = len(set_uid | set_fn) or 1
        inter = len(set_uid & set_fn)
        jacc = inter / union
        if jacc < jaccard_min:
            continue

        score = jacc - 0.01 * abs(len(fn_uid) - len(uid_norm))
        if score > best_score:
            best_score = score
            best_name = fn

    return os.path.join(folder, best_name) if best_name is not None else None


# ==============================================================================
# Generic entry points
# ==============================================================================


def extract_unique_id(filename: str, dataset_type: DatasetType) -> str:
    """
    Dispatch to the appropriate UID extractor based on `dataset_type`.

    - `dataset_type="counts_only"`  → strict suffix-strip using the extended list
    - `dataset_type="localized_gt"` → localized normalization
    """
    if dataset_type == "localized_gt":
        return extract_uid_localized(filename)
    if dataset_type == "counts_only":
        return extract_uid_counts_only(filename, COUNTS_DAPI_SUFFIXES_EXTENDED)
    raise ValueError(f"Unknown dataset_type: {dataset_type!r}")


# ==============================================================================
# Cell-line inference
# ==============================================================================


def infer_cell_line(uid: str) -> str | None:
    """
    Infer the canonical cell-line name for a UID.

    Strategy (first match wins):
      1. Lookup in `CELL_LINE_ALIASES` by normalized alnum token.
      2. Normalized-alnum equality against known canonical names.
      3. Substring fallback (one token contains the normalized canonical form).

    Returns None if no match.
    """
    base = extract_uid_localized(uid)
    toks = _tokens(base)
    joined = {_norm_alnum(t) for t in toks}

    for t in joined:
        if t in CELL_LINE_ALIASES:
            return CELL_LINE_ALIASES[t]

    canon_norm = {_norm_alnum(k): k for k in KNOWN_CELL_LINES}
    for t in joined:
        if t in canon_norm:
            return canon_norm[t]

    for norm_key, canon in canon_norm.items():
        if any(norm_key in t for t in joined):
            return canon

    return None


def bucket_uids_by_cell_line(uids: list[str]) -> dict[str, list[str]]:
    """
    Group UIDs by inferred cell line. UIDs whose cell line cannot be
    inferred are dropped. Returned keys are a subset of `KNOWN_CELL_LINES`.
    """
    buckets: dict[str, list[str]] = {cl: [] for cl in KNOWN_CELL_LINES}
    for uid in uids:
        cl = infer_cell_line(uid)
        if cl and cl in buckets:
            buckets[cl].append(uid)
    return {cl: lst for cl, lst in buckets.items() if lst}


__all__ = [
    "DatasetType",
    "SIDE_TAGS",
    "KNOWN_CELL_LINES",
    "CELL_LINE_ALIASES",
    "VALID_EXTS",
    "STRICT_JACCARD_MIN",
    "COUNTS_DAPI_SUFFIXES",
    "COUNTS_DAPI_SUFFIXES_EXTENDED",
    "extract_uid_counts_only",
    "find_corresponding_dapi_counts",
    "find_corresponding_dapi_counts_extended",
    "extract_uid_localized",
    "find_corresponding_file_localized",
    "extract_unique_id",
    "infer_cell_line",
    "bucket_uids_by_cell_line",
]
