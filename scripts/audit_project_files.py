#!/usr/bin/env python
"""
audit_project_files.py — one sweep over the repository for the five
submission-cleanup jobs that touch many files at once.

    # look first, change nothing
    python scripts/audit_project_files.py --root . --report audit_report.md

    # then apply only the spelling conversion, only to prose files
    python scripts/audit_project_files.py --root . --apply --checks spelling

WHAT IT DOES
------------
    spelling   British -> American spelling in prose files (.md .txt .rst)
    journal    reports every journal-name mention, excluding real citations
    figures    validates every figure and panel reference against the locked
               6 main / 9 Extended Data structure
    authors    writes the canonical author block in four formats and reports
               every file that names an author
    mia        reports every MIA path, module and manifest-column mention

WHAT IT REFUSES TO DO
---------------------
It never rewrites code. Spelling conversion applies to .md, .txt and .rst only;
for .py, .ipynb, .yaml, .yml, .toml, .cff and .json it reports candidates and
stops, because `default_classical_optimised.yaml` is a filename, `harmonised`
may be a directory, and a "fix" to either breaks the pipeline silently.

Inside a prose file it further skips fenced code blocks, inline code spans,
URLs, anything that looks like a path or a filename, and every token on the
PROTECTED list below.

It never edits a reference list, and it never deletes a journal name: removing
one almost always means rewriting the sentence around it, so those are reported
for a human to action.

It never reads its own output. See "REVISION 2026-09-08" below.

REVISION 2026-09-08 — four bugs fixed
-------------------------------------
1. The audit read its own report. Every finding quotes the matched line as
   context, so `audit_report.md` contains, verbatim, every pattern the audit
   searches for. Run 1 wrote a 170 KB file of match patterns into the scanned
   tree, run 2 read it, run 3 read both copies: 2,141 of 2,703 findings were
   the audit reading itself. Now the report and json paths are excluded
   wherever they are written, files with those basenames are excluded anywhere
   in the tree, and any Markdown file whose first line is this script's own
   report heading is skipped on sight.

2. Generated and shadow trees were scanned. `outputs`, `results`, `logs`,
   `build` (a copy of `src/`), `dist`, `env`, `.venv`, `.virtual_documents`
   (Jupyter's shadow copy of `notebooks/`) and `repo_inventory` are now
   skipped, so nothing is counted twice.

3. Notebooks were read as raw JSON, so matplotlib repr strings in *output*
   cells — `<Figure size 640x480>`, `Figure(1417.32x708.661)` — were parsed as
   figure references, producing findings like "Figure 1417 does not exist".
   All 60 figure findings in the 8 September run were of that form. `.ipynb`
   is now parsed as JSON and only markdown-cell prose is read; code cells and
   output cells are ignored. Notebook findings report a `md cell N, line M`
   locator rather than a meaningless raw-file line number.

4. The British-spelling list had a systematic hole: it listed `normalise` but
   not `normalises`, and `\bnormalise\b` does not match "normalises". Fifty-
   seven common British forms were absent, including `tumour` and the `-metre`
   compounds, so a clean spelling audit was a false all-clear. The `-ise`
   family is now generated from stems rather than typed out; the list holds 376
   forms rather than 84. See FIX 4 at the SPELLING definition below.

Three consecutive runs over the same tree now return identical counts.

EXIT CODES
----------
    0  no findings that block submission
    1  findings present (bad figure/panel references, or journal mentions
       outside reference lists)
    2  bad invocation
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# LOCKED FACTS — edit these here, nowhere else
# ---------------------------------------------------------------------------

# Final display-item structure: 6 main figures, 9 Extended Data figures.
FIG_PANELS: dict[int, str] = {
    1: "abcdefgh",
    2: "abcdefghi",
    3: "abcdefg",
    4: "abcde",
    5: "abcdef",
    6: "abcdefghi",
}
ED_PANELS: dict[int, str] = {
    1: "abcd",
    2: "abcdefgh",
    3: "abc",
    4: "abcdef",
    5: "abcdefg",
    6: "abcde",
    7: "",          # single-panel figure, no lettered panels
    8: "abcde",
    9: "abcde",
}

AUTHORS = [
    ("Poorya",    "Behnamie",    ["a"]),
    ("River",     "Summers",     ["b"]),
    ("Jingting",  "Chen",        ["c"]),
    ("Aarav",     "Mehta",       ["b"]),
    ("Adesuwa",   "Igbinigie",   ["a"]),
    ("Qin",       "Liu",         ["b"]),
    ("Molly",     "Murray",      ["d"]),
    ("Damien",    "Guilbaud",    ["a"]),
    ("Marc",      "Niethammer",  ["b"]),
    ("Elizabeth", "Brunk",       ["a", "d", "e", "f", "g"]),
]
CORRESPONDING = "Elizabeth Brunk"
CORRESPONDING_EMAIL = "elizabeth_brunk@med.unc.edu"

AFFILIATIONS = {
    "a": "Integrative Program for Biological and Genome Sciences (IBGS), "
         "University of North Carolina at Chapel Hill, Chapel Hill, NC 27516",
    "b": "Department of Computer Science, University of North Carolina at "
         "Chapel Hill, Chapel Hill, NC 27516",
    "c": "Department of Biochemistry and Biophysics, University of North "
         "Carolina at Chapel Hill, Chapel Hill, NC 27516",
    "d": "Department of Chemistry, University of North Carolina at Chapel "
         "Hill, Chapel Hill, NC 27516",
    "e": "Department of Pharmacology, University of North Carolina at Chapel "
         "Hill, Chapel Hill, NC 27516",
    "f": "Computational Medicine Program, University of North Carolina at "
         "Chapel Hill, Chapel Hill, NC 27516",
    "g": "Lineberger Comprehensive Cancer Center, University of North "
         "Carolina at Chapel Hill, Chapel Hill, NC 27516",
}

# Journal names that must not appear outside a reference list.
JOURNAL_PATTERNS = [
    (r"Nature\s+Computational\s+Science", "high"),
    (r"Nat\.?\s*Comput\.?\s*Sci\.?", "high"),
    (r"Nature\s+Comput(?:ational)?\s+Sci", "high"),
    (r"Nature\s+Biotechnology", "high"),
    (r"Nature\s+Biotech\b", "high"),
    (r"Nat\.?\s*Biotechnol\.?", "high"),
    (r"\bNCS\b", "low"),
]

# British -> American. Case is preserved for a leading capital.
#
# FIX 4 (2026-09-08): the list had a systematic hole. Only some inflections of
# each verb were present, and `\bnormalise\b` does not match "normalises", so
# prose containing "normalises", "visualises", "tumour" or "micrometre" passed
# a clean audit. Fifty-seven common British forms were missing, including
# `tumour` and the metric `-metre` compounds, both near-certain in this
# manuscript. The `-ise` family is now GENERATED from stems so that no
# inflection can be missed again; irregulars stay in the explicit list below.
#
# `analyse` stays out of the generated family on purpose: "analyses" is also
# the plural of the noun "analysis" and must never be converted.

_ISE_STEMS = [
    "optimis", "normalis", "harmonis", "hybridis", "visualis", "localis",
    "characteris", "generalis", "standardis", "parameteris", "initialis",
    "summaris", "minimis", "maximis", "recognis", "utilis", "categoris",
    "prioritis", "organis", "realis", "digitis", "discretis", "randomis",
    "emphasis", "synthesis", "stabilis", "sterilis", "immunis", "penalis",
    "regularis", "binaris", "modularis", "linearis", "vectoris", "quantis",
    "computeris", "customis", "centralis", "specialis",
]

_ISE_SUFFIXES = ["ation", "ations", "ing", "es", "ed", "e"]


def _ise_family() -> list[tuple[str, str]]:
    pairs = []
    for stem in dict.fromkeys(_ISE_STEMS):          # de-duplicate, keep order
        us_stem = stem[:-2] + "iz"                  # ...is -> ...iz
        for suf in _ISE_SUFFIXES:
            pairs.append((stem + suf, us_stem + suf))
    return pairs


SPELLING = _ise_family() + [
    # -ise verbs kept explicit because the noun form collides
    ("analysed", "analyzed"), ("analysing", "analyzing"),
    ("analyse", "analyze"), ("analyses", "analyses"),  # noun form unchanged

    # -our
    ("behaviour", "behavior"), ("behaviours", "behaviors"),
    ("behavioural", "behavioral"),
    ("colour", "color"), ("colours", "colors"), ("coloured", "colored"),
    ("colouring", "coloring"), ("colourbar", "colorbar"),
    ("colourmap", "colormap"), ("colourful", "colorful"),
    ("tumour", "tumor"), ("tumours", "tumors"),
    ("tumourigenic", "tumorigenic"), ("tumourigenesis", "tumorigenesis"),
    ("favour", "favor"), ("favours", "favors"), ("favoured", "favored"),
    ("favouring", "favoring"), ("favourable", "favorable"),
    ("favourably", "favorably"),
    ("labour", "labor"), ("labours", "labors"), ("laboured", "labored"),
    ("rigour", "rigor"), ("vigour", "vigor"), ("odour", "odor"),
    ("vapour", "vapor"), ("vapours", "vapors"),
    ("endeavour", "endeavor"), ("endeavours", "endeavors"),
    ("honour", "honor"), ("humour", "humor"), ("flavour", "flavor"),
    ("neighbour", "neighbor"), ("neighbours", "neighbors"),
    ("neighbouring", "neighboring"), ("neighbourhood", "neighborhood"),

    # -re
    ("centre", "center"), ("centres", "centers"), ("centred", "centered"),
    ("centring", "centering"),
    ("fibre", "fiber"), ("fibres", "fibers"),
    ("metre", "meter"), ("metres", "meters"),
    ("litre", "liter"), ("litres", "liters"),
    ("nanometre", "nanometer"), ("nanometres", "nanometers"),
    ("micrometre", "micrometer"), ("micrometres", "micrometers"),
    ("millimetre", "millimeter"), ("millimetres", "millimeters"),
    ("centimetre", "centimeter"), ("centimetres", "centimeters"),
    ("kilometre", "kilometer"), ("kilometres", "kilometers"),
    ("millilitre", "milliliter"), ("millilitres", "milliliters"),
    ("microlitre", "microliter"), ("microlitres", "microliters"),
    ("nanolitre", "nanoliter"), ("nanolitres", "nanoliters"),
    ("theatre", "theater"), ("calibre", "caliber"),
    ("manoeuvre", "maneuver"), ("manoeuvres", "maneuvers"),

    # doubled consonant
    ("labelled", "labeled"), ("labelling", "labeling"),
    ("modelling", "modeling"), ("modelled", "modeled"),
    ("signalling", "signaling"), ("signalled", "signaled"),
    ("travelling", "traveling"), ("travelled", "traveled"),
    ("cancelled", "canceled"), ("cancelling", "canceling"),
    ("channelled", "channeled"), ("channelling", "channeling"),
    ("totalled", "totaled"), ("fuelled", "fueled"),
    ("levelled", "leveled"), ("levelling", "leveling"),
    ("focussed", "focused"), ("focusses", "focuses"),
    ("focussing", "focusing"),
    ("fulfil", "fulfill"), ("fulfilment", "fulfillment"),
    ("skilful", "skillful"), ("instalment", "installment"),
    ("enrol", "enroll"),

    # -ae / -oe (biomedical)
    ("haematoxylin", "hematoxylin"), ("haemoglobin", "hemoglobin"),
    ("haematopoietic", "hematopoietic"), ("haematological", "hematological"),
    ("haemorrhage", "hemorrhage"), ("haematoxylin", "hematoxylin"),
    ("leukaemia", "leukemia"), ("leukaemic", "leukemic"),
    ("anaemia", "anemia"), ("anaemic", "anemic"),
    ("oedema", "edema"), ("oesophageal", "esophageal"),
    ("oesophagus", "esophagus"), ("oestrogen", "estrogen"),
    ("oestrogens", "estrogens"), ("paediatric", "pediatric"),
    ("foetal", "fetal"), ("foetus", "fetus"),
    ("aetiology", "etiology"), ("anaesthesia", "anesthesia"),
    ("orthopaedic", "orthopedic"), ("diarrhoea", "diarrhea"),

    # miscellaneous
    ("greyscale", "grayscale"), ("grey", "gray"), ("greys", "grays"),
    ("artefact", "artifact"), ("artefacts", "artifacts"),
    ("artefactual", "artifactual"),
    ("catalogue", "catalog"), ("catalogues", "catalogs"),
    ("analogue", "analog"), ("analogues", "analogs"),
    ("defence", "defense"), ("licence", "license"),
    ("offence", "offense"), ("pretence", "pretense"),
    ("practise", "practice"), ("practised", "practiced"),
    ("practising", "practicing"),
    ("programme", "program"), ("programmes", "programs"),
    ("ageing", "aging"), ("judgement", "judgment"),
    ("sulphur", "sulfur"), ("sulphate", "sulfate"), ("sulphide", "sulfide"),
    ("aluminium", "aluminum"), ("mould", "mold"), ("moulds", "molds"),
]
# NOTE: "acknowledgement" is deliberately absent. Nature journals spell the
# section heading "Acknowledgements" in otherwise American-English papers.

# Drop identity pairs (kept above only as reminders) and de-duplicate,
# longest first so an overlapping shorter form never claims the match.
SPELLING = [(a, b) for a, b in dict.fromkeys(SPELLING) if a != b]
SPELLING.sort(key=lambda p: (-len(p[0]), p[0]))

# Tokens that look British but are identifiers, filenames or fixed strings.
# Anything here is never converted, wherever it appears.
PROTECTED = {
    "default_classical_optimised.yaml",
    "classical_optimised",
    "harmonised_masks",
    "harmonized_masks",
    "optimised.yaml",
    "colour_map",          # add real identifiers here as they turn up
}

MIA_PATTERNS = [
    r"mia_mask_relpath",
    r"original_mia_name",
    r"ecdna[_-]bench\.baselines\.mia",
    r"baselines[/\\]mia(?:\.py)?",
    r"\bmia\.py\b",
    r"mia_mask(?:s)?",
    r"mia_environment",
    r"\bMIA\b",
]

PROSE_EXT = {".md", ".txt", ".rst"}
CODE_EXT = {".py", ".ipynb", ".yaml", ".yml", ".toml", ".cff", ".json", ".sh", ".cfg"}

# FIX 2 (2026-09-08): generated trees, shadow copies and virtualenvs are never
# the source of truth and scanning them double-counts everything.
SKIP_DIRS = {".git", ".ipynb_checkpoints", "__pycache__", "node_modules",
             ".mypy_cache", ".pytest_cache", "reference_data",
             "reference_figures", "reference_configs", "release",
             "outputs", "results", "logs", "build", "dist",
             "env", ".venv", "venv", ".virtual_documents", "repo_inventory"}

# FIX 1 (2026-09-08): the report is full of the patterns the audit searches
# for. This is the first line it writes; any Markdown file starting with it is
# a previous report, wherever it was written and whatever it was named.
REPORT_HEADING = "# Project-file audit"

URL_RE = re.compile(r"https?://\S+|www\.\S+|doi:\S+|10\.\d{4,}/\S+")
PATHY_RE = re.compile(r"[\w./\\-]*[/\\][\w./\\-]+|\b\w+\.(?:yaml|yml|py|csv|json|png|svg|md|toml|cff|ipynb|h5|npz|sh)\b")
INLINE_CODE_RE = re.compile(r"`[^`]*`")
CITATIONY_RE = re.compile(
    r"\bet al\b|\(\s*(?:19|20)\d{2}\s*\)|\bdoi\b|https?://|"
    r"\bpmid\b|\barxiv\b|\bpreprint\b", re.I)
REF_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s*(references|bibliography|"
                            r"works cited|citations)\s*$", re.I)
NEXT_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+\S")

# A panel letter is a lone lower-case letter not followed by another letter, so
# that "Fig. 8e, Extended ..." does not read the E of "Extended" as a panel and
# "Figure 7 and" does not read the a of "and". Panel letters must sit directly
# against the number ("Fig. 6a-i"), which is the manuscript's own style. A
# spaced form ("Fig. 6 a") is therefore not checked - a missed finding is safer
# here than a false one.
_PANEL = r"([a-z](?![A-Za-z])(?:\s*[,–—-]\s*[a-z](?![A-Za-z]))*)?"

ED_FIG_RE = re.compile(
    r"(?i:Extended\s+Data\s+Fig(?:ure)?\.?)\s*(\d+)" + _PANEL)
SUPP_FIG_RE = re.compile(r"(?i:Supplementary\s+Fig(?:ure)?\.?)\s*(\d+)")
MAIN_FIG_RE = re.compile(
    r"(?<!Data\s)(?<!Supplementary\s)\b(?i:Fig(?:ure)?\.?)\s*(\d+)" + _PANEL)

# FIX 3 (2026-09-08): for .ipynb the line index is an index into extracted
# markdown prose, not into the raw file. Keep a human-usable locator.
NB_LABELS: dict[str, dict[int, str]] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def looks_like_own_report(path: Path) -> bool:
    """A previous run's Markdown report, wherever it lives and whatever it is
    called. Cheap: reads only the first line."""
    if path.suffix.lower() != ".md":
        return False
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            return fh.readline().strip() == REPORT_HEADING
    except OSError:
        return False


def looks_like_own_json(path: Path) -> bool:
    """A previous run's findings json. Sniffs the shape, not the name."""
    if path.suffix.lower() != ".json":
        return False
    try:
        if path.stat().st_size > 50_000_000:
            return False
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return False
    if not isinstance(data, list) or not data or not isinstance(data[0], dict):
        return False
    return {"file", "line", "kind", "detail", "text"} <= set(data[0])


def iter_files(root: Path, extra_skip: set[str],
               self_paths: set[Path], self_names: set[str]) -> list[Path]:
    out = []
    skip = SKIP_DIRS | extra_skip
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        if any(part in skip for part in p.parts):
            continue
        if p.suffix.lower() not in PROSE_EXT | CODE_EXT:
            continue
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp == Path(__file__).resolve():
            continue        # this file contains every pattern it searches for
        if rp in self_paths or p.name in self_names:
            continue        # FIX 1: this run's own report / json
        if looks_like_own_report(p) or looks_like_own_json(p):
            continue        # FIX 1: any previous run's output, anywhere
        out.append(p)
    return out


def read_notebook_markdown(path: Path) -> list[str] | None:
    """FIX 3: markdown-cell prose only. Code cells and output cells are
    ignored - a matplotlib repr in an output cell is not a figure reference."""
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, OSError, ValueError):
        return None
    if not isinstance(nb, dict):
        return None
    lines: list[str] = []
    labels: dict[int, str] = {}
    for c, cell in enumerate(nb.get("cells", [])):
        if not isinstance(cell, dict) or cell.get("cell_type") != "markdown":
            continue
        src = cell.get("source", "")
        if isinstance(src, list):
            src = "".join(src)
        for j, ln in enumerate(str(src).splitlines(), start=1):
            lines.append(ln)
            labels[len(lines)] = f"md cell {c}, line {j}"
    NB_LABELS[str(path)] = labels
    return lines


def read_lines(path: Path) -> list[str] | None:
    if path.suffix.lower() == ".ipynb":
        return read_notebook_markdown(path)
    try:
        return path.read_text(encoding="utf-8").splitlines()
    except (UnicodeDecodeError, OSError):
        return None


def line_label(file: str, line: int) -> str:
    return str(NB_LABELS.get(file, {}).get(line, line))


def mask_spans(line: str) -> list[tuple[int, int]]:
    """Character ranges that must not be touched: URLs, paths, inline code."""
    spans = []
    for rx in (URL_RE, INLINE_CODE_RE, PATHY_RE):
        spans.extend((m.start(), m.end()) for m in rx.finditer(line))
    for tok in PROTECTED:
        start = 0
        low, ltok = line.lower(), tok.lower()
        while (i := low.find(ltok, start)) != -1:
            spans.append((i, i + len(tok)))
            start = i + 1
    return spans


def in_masked(pos: int, end: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos and end <= e for s, e in spans)


def annotate_context(lines: list[str]) -> list[tuple[bool, bool]]:
    """For each line return (in_code_fence, in_reference_section)."""
    out, fence, refs = [], False, False
    for ln in lines:
        if ln.lstrip().startswith("```") or ln.lstrip().startswith("~~~"):
            fence = not fence
            out.append((True, refs))
            continue
        if REF_HEADING_RE.match(ln):
            refs = True
        elif refs and NEXT_HEADING_RE.match(ln) and not REF_HEADING_RE.match(ln):
            refs = False
        out.append((fence, refs))
    return out


def expand_panels(raw: str | None) -> list[str]:
    """'a-c' -> [a,b,c];  'd,e' -> [d,e];  None -> []."""
    if not raw:
        return []
    raw = raw.replace("–", "-").replace("—", "-")
    letters: list[str] = []
    for chunk in re.split(r"\s*,\s*", raw.strip()):
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            lo, hi = lo.strip(), hi.strip()
            if len(lo) == 1 and len(hi) == 1 and lo <= hi:
                letters.extend(chr(c) for c in range(ord(lo), ord(hi) + 1))
        elif len(chunk.strip()) == 1:
            letters.append(chunk.strip())
    return letters


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def check_spelling(path: Path, lines: list[str], ctx, apply_it: bool):
    """Replacement is span-aware: only the exact unmasked match positions are
    rewritten. A whole-line re.sub would also hit occurrences inside filenames
    and URLs on the same line, which is how `optimised.yaml` gets broken."""
    findings, new_lines, changed = [], list(lines), False
    prose = path.suffix.lower() in PROSE_EXT
    for i, line in enumerate(lines):
        fence, refs = ctx[i]
        if fence or refs:
            continue
        spans = mask_spans(line)
        edits: list[tuple[int, int, str, str]] = []   # start, end, found, repl
        for brit, amer in SPELLING:
            for m in re.finditer(rf"\b{re.escape(brit)}\b", line, re.I):
                if in_masked(m.start(), m.end(), spans):
                    continue
                if any(s < m.end() and m.start() < e for s, e, _, _ in edits):
                    continue                       # already covered
                found = m.group(0)
                repl = amer.capitalize() if found[0].isupper() else amer
                edits.append((m.start(), m.end(), found, repl))
        if not edits:
            continue
        edits.sort()
        updated, cursor = [], 0
        for s, e, _, repl in edits:
            updated.append(line[cursor:s])
            updated.append(repl)
            cursor = e
        updated.append(line[cursor:])
        findings.append({
            "file": str(path), "line": i + 1, "kind": "spelling",
            "detail": ", ".join(f"{f} -> {r}" for _, _, f, r in edits),
            "text": line.strip()[:110],
            "auto": prose,
        })
        if apply_it and prose:
            new_lines[i], changed = "".join(updated), True
    return findings, (new_lines if changed else None)


def check_journal(path: Path, lines: list[str], ctx):
    findings = []
    for i, line in enumerate(lines):
        fence, refs = ctx[i]
        if refs:
            continue
        if CITATIONY_RE.search(line):
            continue          # a real citation, leave it alone
        seen: list[tuple[int, int]] = []
        for pat, conf in JOURNAL_PATTERNS:
            for m in re.finditer(pat, line):
                # The patterns deliberately overlap ("Nature Computational
                # Science" and "Nature Comput...Sci"); report each stretch of
                # text once, under the first pattern that claims it.
                if any(s < m.end() and m.start() < e for s, e in seen):
                    continue
                seen.append((m.start(), m.end()))
                findings.append({
                    "file": str(path), "line": i + 1, "kind": "journal",
                    "detail": f"{m.group(0)!r} ({conf} confidence)"
                              + ("  [in code block]" if fence else ""),
                    "text": line.strip()[:110], "auto": False,
                })
    return findings


def check_figures(path: Path, lines: list[str], ctx):
    findings = []
    for i, line in enumerate(lines):
        if ctx[i][1]:
            continue
        consumed = []

        for m in ED_FIG_RE.finditer(line):
            consumed.append((m.start(), m.end()))
            n = int(m.group(1))
            if n not in ED_PANELS:
                findings.append(_f(path, i, "figures",
                                   f"Extended Data Fig. {n} does not exist "
                                   f"(1-9 only)", line))
                continue
            bad = [p for p in expand_panels(m.group(2))
                   if p not in ED_PANELS[n]]
            if bad:
                have = ED_PANELS[n] or "no lettered panels"
                findings.append(_f(path, i, "figures",
                                   f"Extended Data Fig. {n} has {have}; "
                                   f"reference names {','.join(bad)}", line))

        for m in SUPP_FIG_RE.finditer(line):
            consumed.append((m.start(), m.end()))
            findings.append(_f(path, i, "figures",
                               f"'Supplementary Figure {m.group(1)}' is stale "
                               f"- supplementary figures became Extended Data",
                               line))

        for m in MAIN_FIG_RE.finditer(line):
            if any(s <= m.start() < e for s, e in consumed):
                continue
            n = int(m.group(1))
            if n not in FIG_PANELS:
                findings.append(_f(path, i, "figures",
                                   f"Figure {n} does not exist (1-6 only); "
                                   f"if an Extended Data figure is meant, say so",
                                   line))
                continue
            bad = [p for p in expand_panels(m.group(2)) if p not in FIG_PANELS[n]]
            if bad:
                findings.append(_f(path, i, "figures",
                                   f"Figure {n} has panels {FIG_PANELS[n]}; "
                                   f"reference names {','.join(bad)}", line))
    return findings


def check_mia(path: Path, lines: list[str], ctx):
    findings = []
    for i, line in enumerate(lines):
        for pat in MIA_PATTERNS:
            for m in re.finditer(pat, line):
                findings.append(_f(path, i, "mia", f"{m.group(0)!r}", line))
                break        # one finding per line per pattern is enough
    return findings


def check_authors(path: Path, lines: list[str], ctx):
    surnames = {s for _, s, _ in AUTHORS}
    findings = []
    for i, line in enumerate(lines):
        hit = [s for s in surnames if re.search(rf"\b{s}\b", line)]
        if len(hit) >= 2:
            findings.append(_f(path, i, "authors",
                               f"names {len(hit)} authors: {', '.join(sorted(hit))}",
                               line))
    return findings


def _f(path, i, kind, detail, line):
    return {"file": str(path), "line": i + 1, "kind": kind,
            "detail": detail, "text": line.strip()[:110], "auto": False}


# ---------------------------------------------------------------------------
# author blocks
# ---------------------------------------------------------------------------

def author_blocks() -> str:
    names = ", ".join(
        f"{g} {s}" + ("[" + ",".join(a) + "]" if a else "")
        for g, s, a in AUTHORS)

    cff = ["authors:"]
    for g, s, _ in AUTHORS:
        cff += [f"  - given-names: {g}", f"    family-names: {s}",
                "    affiliation: University of North Carolina at Chapel Hill"]

    toml = ["authors = ["]
    for g, s, _ in AUTHORS:
        toml.append(f'  {{ name = "{g} {s}" }},')
    toml.append("]")

    md = [f"{i+1}. {g} {s} ({','.join(a)})"
          for i, (g, s, a) in enumerate(AUTHORS)]
    aff = [f"{k}. {v}" for k, v in AFFILIATIONS.items()]

    return "\n".join([
        "### Canonical author line (manuscript order)", "", names, "",
        f"Correspondence: {CORRESPONDING} ({CORRESPONDING_EMAIL})",
        "",
        "> The manuscript also carries \"*These authors contributed equally\" "
        "with no asterisk on any name. Left as-is pending Liz's decision - do "
        "not propagate the asterisk note until it is resolved.",
        "", "### Affiliations", "", *aff,
        "", "### CITATION.cff", "", "```yaml", *cff, "```",
        "", "### pyproject.toml", "", "```toml", *toml, "```",
        "", "### Markdown / README", "", *md,
    ])


# ---------------------------------------------------------------------------

CHECKS = {
    "spelling": None, "journal": None, "figures": None,
    "authors": None, "mia": None,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--checks", default="all",
                    help="comma-separated: " + ",".join(CHECKS) + " (default all)")
    ap.add_argument("--apply", action="store_true",
                    help="write spelling fixes to prose files; every other "
                         "check is report-only, always")
    ap.add_argument("--report", type=Path, default=Path("audit_report.md"))
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--skip-dir", action="append", default=[])
    args = ap.parse_args()

    if not args.root.is_dir():
        print(f"error: --root {args.root} is not a directory", file=sys.stderr)
        return 2
    wanted = set(CHECKS) if args.checks == "all" else {
        c.strip() for c in args.checks.split(",")}
    if bad := wanted - set(CHECKS):
        print(f"error: unknown check(s): {', '.join(sorted(bad))}", file=sys.stderr)
        return 2

    # FIX 1: this run's own outputs, plus their basenames anywhere in the tree
    # (a previous run may have written them somewhere else).
    self_paths = {p.resolve() for p in (args.report, args.json) if p}
    self_names = {p.name for p in (args.report, args.json) if p}

    files = iter_files(args.root, set(args.skip_dir), self_paths, self_names)
    findings: list[dict] = []
    edited: list[Path] = []

    for path in files:
        lines = read_lines(path)
        if lines is None:
            continue
        ctx = annotate_context(lines)

        if "spelling" in wanted:
            f, new = check_spelling(path, lines, ctx, args.apply)
            findings += f
            if new is not None:
                path.write_text("\n".join(new) + "\n", encoding="utf-8")
                edited.append(path)
        if "journal" in wanted:
            findings += check_journal(path, lines, ctx)
        if "figures" in wanted:
            findings += check_figures(path, lines, ctx)
        if "mia" in wanted:
            findings += check_mia(path, lines, ctx)
        if "authors" in wanted:
            findings += check_authors(path, lines, ctx)

    for f in findings:
        f["line_label"] = line_label(f["file"], f["line"])

    by_kind: dict[str, list[dict]] = {}
    for f in findings:
        by_kind.setdefault(f["kind"], []).append(f)

    # ---- report ----------------------------------------------------------
    out = [REPORT_HEADING, "",
           f"Root: `{args.root.resolve()}`  ·  files scanned: {len(files)}  ·  "
           f"mode: {'APPLY (spelling only)' if args.apply else 'dry run'}", ""]
    order = ["figures", "journal", "spelling", "mia", "authors"]
    titles = {
        "figures": "Figure and panel references that do not match the locked "
                   "6 main / 9 Extended Data structure",
        "journal": "Journal-name mentions outside reference lists — remove by "
                   "hand, each one needs the sentence rewritten",
        "spelling": "British spellings in prose",
        "mia": "MIA paths, modules and manifest columns",
        "authors": "Files naming two or more authors — check order against the "
                   "canonical block below",
    }
    out.append("## Summary\n")
    out.append("| Check | Findings |")
    out.append("|---|---|")
    for k in order:
        if k in wanted:
            out.append(f"| {k} | {len(by_kind.get(k, []))} |")
    out.append("")

    for k in order:
        if k not in wanted:
            continue
        rows = by_kind.get(k, [])
        out += [f"## {titles[k]}", ""]
        if not rows:
            out += ["None.", ""]
            continue
        if k == "spelling":
            auto = sum(1 for r in rows if r["auto"])
            out += [f"{auto} of {len(rows)} are in prose files and are "
                    f"converted by `--apply --checks spelling`. The remainder "
                    f"are in code or config and are reported only — check each "
                    f"one is prose and not an identifier before touching it.",
                    ""]
        out += ["| File | Line | Detail | Context |", "|---|---|---|---|"]
        for r in rows[:400]:
            txt = r["text"].replace("|", "\\|")
            out.append(f"| `{r['file']}` | {r['line_label']} | {r['detail']} | {txt} |")
        if len(rows) > 400:
            out.append(f"| … | | {len(rows) - 400} more not listed | |")
        out.append("")

    if "authors" in wanted:
        out += [author_blocks(), ""]

    if edited:
        out += ["## Files written", ""] + [f"- `{p}`" for p in edited] + [""]

    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text("\n".join(out), encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(findings, indent=2), encoding="utf-8")

    print(f"scanned {len(files)} files")
    for k in order:
        if k in wanted:
            print(f"  {k:<9} {len(by_kind.get(k, [])):>5}")
    if edited:
        print(f"  wrote     {len(edited):>5} files")
    print(f"report: {args.report}")

    blocking = len(by_kind.get("figures", [])) + len(by_kind.get("journal", []))
    return 1 if blocking else 0


if __name__ == "__main__":
    raise SystemExit(main())