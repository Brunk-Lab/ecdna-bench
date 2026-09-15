#!/usr/bin/env python3
"""
scripts/rename_gold_standard.py
===============================
Replace the term "ground truth" with "gold standard" (and the abbreviation
"GT" with "GS") in the human-readable text of the ecdna-bench repository:
comments, docstrings, Markdown, notebook Markdown cells, YAML comments,
shell comments and the display strings that end up in figures, logs and
command-line help.

What it deliberately does NOT change
------------------------------------
* Anything already deposited in the BioImage Archive (S-BIAD4097): the
  folder names ``gt_image/``, ``gt_coords/``, ``gt_npz/``, the file lists
  ``filelist_gt.*``, the archive section named "ground truth", and the
  scripts that generated the deposition.
* Machine names: Python identifiers (``gt_mask``, ``ecDNA_gt``,
  ``run_gt_morphology_audit``), snake_case string keys, CSV/TSV column
  headers and cell values, YAML keys, file names and paths.  These stay as
  they are so the code keeps reading the frozen result tables and the
  deposited data.  The README carries one sentence explaining that ``gt``
  in code abbreviates the gold-standard annotation.
* Locked artifacts: ``release/frozen_results``, ``release/figures``,
  ``release/manifests``, ``release/split_files`` and friends.
* Notebook outputs.  Re-run a notebook to refresh its outputs.

A string literal that is not obviously display text (for example a list
item or a dict value) is changed only if its exact value does not appear as
a field in any CSV/TSV/JSON data file under the scanned roots.  If it does,
it is listed for review instead, because code may be using it to look up
data.

Safety
------
* ``scan`` (the default) writes nothing into the repository.  It writes a
  report, a summary and a unified-diff preview into ``--out-dir``.
* ``apply`` refuses to touch a file with uncommitted git changes unless
  ``--allow-dirty`` is given, refuses if any target file is read-only, and
  verifies every change in memory first.  If any file fails verification
  nothing is written, unless ``--skip-errors`` is given (then the failing
  files are left untouched and listed).
* Every edited Python file (and every edited notebook code cell) is parsed
  before and after; the two syntax trees must be identical except for the
  planned edits to string constants.  Comments are not part of the tree, so
  a comment edit can never change behavior.
* Files outside git are backed up to ``--backup-dir`` before they are
  written.
* ``--self-test`` runs the built-in test cases and exits.

Usage (from the repository root)
--------------------------------
    python scripts/rename_gold_standard.py --self-test
    python scripts/rename_gold_standard.py scan  --root . --out-dir ~/gs_rename
    python scripts/rename_gold_standard.py apply --root . --out-dir ~/gs_rename
    python scripts/rename_gold_standard.py scan  --root . --out-dir ~/gs_rename_after

Extra roots (for example the figure notebooks that live outside the
repository) can be added with ``--extra-root PATH`` (repeatable); their CSV
tables are added to the data check automatically.

Only the Python standard library is used, so any Python >= 3.8 works.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import datetime as _dt
import difflib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tokenize
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

__version__ = "1.0.0"

csv.field_size_limit(min(sys.maxsize, 2**31 - 1))

# ============================================================================
# 1.  Scope
# ============================================================================

# Directory names skipped anywhere in the tree.
SKIP_DIR_NAMES: Set[str] = {
    ".git", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", "node_modules", ".venv", "venv", "build", "dist",
    "bioimage_archive", "deposition", "file_lists",
    "reference_data", "reference_figures", "reference_configs",
}

# Directory paths (relative to a root) skipped: locked or generated trees.
SKIP_REL_DIRS: Tuple[str, ...] = (
    "release/frozen_results", "release/figures", "release/manifests",
    "release/split_files", "release/deposition", "release/model_checkpoints",
    "release/harmonized_masks", "release/baseline_predictions",
    "release/river_roi_run", "release/roi_validation", "release/source_data",
    "outputs", "results", "runs", "logs", "env",
)

# Files that produced or verify what is deposited in the BioImage Archive.
# Their wording mirrors the deposited record, so they are left untouched.
PROTECTED_FILE_NAMES: Set[str] = {
    "generate_file_lists.py", "verify_deposition_tree.py", "trace_roi.py",
    "verify_bia_upload.py", "fill_release_values.sh", "bia_ftp_upload.sh",
    "rename_gold_standard.py", "fix_stale_docs.py",
}
PROTECTED_NAME_PATTERNS: Tuple[str, ...] = (
    r"(?i)deposition", r"(?i)^filelist_", r"(?i)bioimage", r"(?i)^bia_",
)

PY_EXT = {".py"}
NB_EXT = {".ipynb"}
MD_EXT = {".md", ".markdown", ".rst", ".txt"}
YAML_EXT = {".yaml", ".yml"}
SH_EXT = {".sh", ".bash", ".slurm", ".sbatch"}
REPORT_ONLY_EXT = {".toml", ".cff", ".cfg", ".ini"}
DATA_EXT = {".csv", ".tsv", ".json"}

# ============================================================================
# 2.  Replacement rules
# ============================================================================

# Nouns after which "ground truth" is an adjective and takes a hyphen.
_NOUNS = (
    "annotation|annotations|mask|masks|count|counts|object|objects|point|points|"
    "coordinate|coordinates|centroid|centroids|label|labels|labelling|labeling|"
    "image|images|data|dataset|datasets|set|sets|burden|value|values|dot|dots|"
    "pixel|pixels|location|locations|component|components|array|arrays|file|"
    "files|folder|folders|directory|directories|path|paths|table|tables|column|"
    "columns|ecDNA|ecDNAs|number|numbers|total|totals|distribution|"
    "distributions|position|positions|spot|spots|signal|signals|matching|"
    "overlay|overlays|diamond|diamonds|footprint|footprints|rendering|disk|"
    "disks|box|boxes|bbox|bboxes|map|maps|effect|effects|call|calls|"
    "reconstruction|triplet|triplets|CC|CCs|instance|instances|side|frame|"
    "representation|representations|source|reference|abundance|density|class|"
    "classes|category|categories|record|records|entry|entries|list|lists|check|"
    "checks|definition|format|formats|shape|shapes|area|areas|bin|bins|"
    "partner|partners|foreground|region|regions|morphology|marker|markers|"
    "loading|load|read|reader|decoder|encoding|comparison|comparisons|fold|"
    "log2|median|medians|mean|means|pair|pairs|target|targets|"
    "conclusion|conclusions|verdict|verdicts|size|sizes"
)

_GT_LEFT = r"(?<![\w./${])(?<!%\()"
_GT_RIGHT = r"(?![\w/}]|\.\w)"

# (compiled pattern, replacement) — applied in order.
CORE_RULES: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"\bground[- ]truths\b"), "gold-standard objects"),
    (re.compile(r"\bGround[- ]truths\b"), "Gold-standard objects"),
    (re.compile(r"\bground-truth\b"), "gold-standard"),
    (re.compile(r"\bGround-truth\b"), "Gold-standard"),
    (re.compile(r"\bGround-Truth\b"), "Gold-Standard"),
    (re.compile(r"\bGROUND-TRUTH\b"), "GOLD-STANDARD"),
    (re.compile(r"\bground truth(?=[ \t]+(?:" + _NOUNS + r")\b)"), "gold-standard"),
    (re.compile(r"\bGround truth(?=[ \t]+(?:" + _NOUNS + r")\b)"), "Gold-standard"),
    (re.compile(r"\bground truth\b"), "gold standard"),
    (re.compile(r"\bGround truth\b"), "Gold standard"),
    (re.compile(r"\bGround Truth\b"), "Gold Standard"),
    (re.compile(r"\bGROUND TRUTH\b"), "GOLD STANDARD"),
    (re.compile(r"\bgroundtruth\b"), "gold standard"),
    (re.compile(r"\bGroundtruth\b"), "Gold standard"),
    (re.compile(r"\btrue (counts?|burden|ecDNA (?:counts?|burden))\b"), r"gold-standard \1"),
    (re.compile(r"\bTrue (counts?|burden|ecDNA (?:counts?|burden))\b"), r"Gold-standard \1"),
    (re.compile(_GT_LEFT + r"GTs" + _GT_RIGHT), "GS objects"),
    (re.compile(_GT_LEFT + r"GT" + _GT_RIGHT), "GS"),
]

# Optional figure-label terms from the co-author review (--figure-terms).
FIGURE_RULES: List[Tuple["re.Pattern[str]", str]] = [
    (re.compile(r"(?<![\w./])LabelEngine(?![\w/]|\.\w)"), "Label Engine"),
    (re.compile(r"\bdensity bins\b"), "count bins"),
    (re.compile(r"\bdensity bin\b"), "count bin"),
    (re.compile(r"\bDensity bins\b"), "Count bins"),
    (re.compile(r"\bDensity bin\b"), "Count bin"),
    (re.compile(r"\bdensity regimes\b"), "count regimes"),
    (re.compile(r"\bPixel Dice\b"), "Pixel-level Dice"),
    (re.compile(r"\bpixel Dice\b"), "pixel-level Dice"),
]

# Anything that looks like a term we care about (used for reporting).
DETECT = re.compile(
    r"(?i:ground[- ]?truths?)|" + _GT_LEFT + r"GTs?" + _GT_RIGHT
    + r"|(?i:\btrue (?:counts?|burden)\b)"
)
DETECT_FIGURE = re.compile(r"LabelEngine|(?i:density bins?)|(?i:\bpixel Dice\b)")
LOWER_GT = re.compile(r"(?<![\w./$])gt(?![\w/]|\.\w)")
STALE_DIAMOND = re.compile(r"7\s*[x×X]\s*7")
ABS_PATH = re.compile(r"(/proj/brunk_ecdna_cv_project|/work/users/|/nas/longleaf/|/users/[a-z]/[a-z]/)")

# Lines that talk about the deposited record keep their wording.
ARCHIVE_LINE = re.compile(
    r"S-BIAD|BioImage Archive|BioStudies|archive's|filelist_|bioimage_archive|EBI\b",
    re.IGNORECASE,
)

# Spans that are never edited inside prose.
PROTECTED_SPAN = re.compile(
    r"``[^`\n]+``"                          # RST inline literal
    r"|`[^`\n]+`"                           # Markdown inline code
    r"|https?://\S+"                        # URLs
    r"|\]\([^)\s]+\)"                       # Markdown link targets
    r"|\"(?:ground[- ]truth|GT)\""          # quoted names
    r"|'(?:ground[- ]truth|GT)'"
    r"|“(?:ground[- ]truth|GT)”"
    r"|\{[^{}\n]*\}"                        # format fields
    r"|%\([^)\n]*\)[#0 +-]*\d*(?:\.\d+)?[a-zA-Z]",  # %-format fields
    re.IGNORECASE,
)


def active_rules(figure_terms: bool) -> List[Tuple["re.Pattern[str]", str]]:
    return CORE_RULES + (FIGURE_RULES if figure_terms else [])


def transform_prose(text: str, rules: Sequence[Tuple["re.Pattern[str]", str]],
                    protect_archive_lines: bool = True) -> Tuple[str, int, int]:
    """
    Apply `rules` to `text`, skipping protected spans and archive lines.

    Returns (new_text, n_replacements, n_protected_hits).
    """
    out_lines: List[str] = []
    n_rep = 0
    n_prot = 0
    for line in text.splitlines(keepends=True):
        if protect_archive_lines and ARCHIVE_LINE.search(line) and _has_term(line, rules):
            n_prot += 1
            out_lines.append(line)
            continue
        if line.lstrip().startswith((">>> ", "... ")):
            if _has_term(line, rules):
                n_prot += 1
            out_lines.append(line)
            continue
        pieces: List[str] = []
        pos = 0
        for m in PROTECTED_SPAN.finditer(line):
            pieces.append(_apply(line[pos:m.start()], rules))
            if _has_term(m.group(0), rules):
                n_prot += 1
            pieces.append((m.group(0), 0))  # type: ignore[arg-type]
            pos = m.end()
        pieces.append(_apply(line[pos:], rules))
        new_line = "".join(p[0] for p in pieces)
        n_rep += sum(p[1] for p in pieces)
        out_lines.append(new_line)
    return "".join(out_lines), n_rep, n_prot


def _apply(segment: str, rules) -> Tuple[str, int]:
    total = 0
    for pat, rep in rules:
        segment, n = pat.subn(rep, segment)
        total += n
    return segment, total


def _has_term(text: str, rules) -> bool:
    return any(p.search(text) for p, _ in rules)


# ============================================================================
# 3.  Records
# ============================================================================

@dataclass
class Finding:
    path: str
    line: int
    kind: str          # comment, docstring, display_string, other_string, markdown, ...
    action: str        # APPLIED, REVIEW, KEPT, INFO
    reason: str
    before: str
    after: str = ""


@dataclass
class FileResult:
    path: Path
    rel: str
    old_text: str
    new_text: str
    findings: List[Finding] = field(default_factory=list)
    error: str = ""

    @property
    def changed(self) -> bool:
        return self.new_text != self.old_text and not self.error


def _snip(s: str, n: int = 160) -> str:
    s = s.replace("\t", " ").replace("\n", "\\n")
    return s if len(s) <= n else s[: n - 3] + "..."


# ============================================================================
# 4.  Data-file check (is a string literal also a data value?)
# ============================================================================

class DataIndex:
    """Distinct CSV/TSV/JSON fields that contain one of the target terms."""

    def __init__(self, max_mb: float = 200.0) -> None:
        self.values: Set[str] = set()
        self.files_scanned = 0
        self.files_skipped: List[str] = []
        self.max_bytes = int(max_mb * 1024 * 1024)

    @staticmethod
    def _interesting(s: str) -> bool:
        return bool(DETECT.search(s) or DETECT_FIGURE.search(s))

    def add_file(self, p: Path) -> None:
        try:
            size = p.stat().st_size
        except OSError:
            return
        if size > self.max_bytes:
            self.files_skipped.append(f"{p} ({size / 1e6:.0f} MB > cap)")
            return
        try:
            if p.suffix.lower() == ".json":
                with open(p, encoding="utf-8", errors="replace") as fh:
                    obj = json.load(fh)
                self._walk_json(obj)
            else:
                delim = "\t" if p.suffix.lower() == ".tsv" else ","
                with open(p, newline="", encoding="utf-8", errors="replace") as fh:
                    for row in csv.reader(fh, delimiter=delim):
                        for cell in row:
                            if cell and self._interesting(cell):
                                self.values.add(cell)
                                self.values.add(cell.strip())
            self.files_scanned += 1
        except Exception as exc:  # malformed data files are reported, not fatal
            self.files_skipped.append(f"{p} (unreadable: {exc.__class__.__name__})")

    def _walk_json(self, obj) -> None:
        stack = [obj]
        while stack:
            o = stack.pop()
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(k, str) and self._interesting(k):
                        self.values.add(k)
                    stack.append(v)
            elif isinstance(o, list):
                stack.extend(o)
            elif isinstance(o, str) and self._interesting(o):
                self.values.add(o)

    def contains(self, s: str) -> bool:
        return s in self.values or s.strip() in self.values


# ============================================================================
# 5.  Python source handling
# ============================================================================

DISPLAY_CALLEES = {
    "print", "pprint", "info", "debug", "warning", "warn", "error", "exception",
    "critical", "fatal", "log", "set_title", "set_xlabel", "set_ylabel",
    "set_zlabel", "suptitle", "supxlabel", "supylabel", "title", "xlabel",
    "ylabel", "zlabel", "text", "annotate", "figtext", "set_text", "legend",
    "set_xticklabels", "set_yticklabels", "set_ticklabels", "xticks", "yticks",
    "set_label", "set_suptitle", "Markdown", "HTML", "display", "echo",
    "secho", "bar_label", "set_axis_labels", "set_titles", "exit",
}
DISPLAY_KEYWORDS = {
    "label", "title", "xlabel", "ylabel", "zlabel", "suptitle", "help",
    "description", "epilog", "desc", "text", "legend_title", "cbar_label",
    "usage", "msg", "message",
}
PLOT_LABELS_CALLEES = {"xticks", "yticks", "set_xticks", "set_yticks", "legend",
                       "boxplot", "pie", "bar_label", "set_ticks", "violinplot"}
EXCEPTION_NAMES = re.compile(r"(Error|Exception|Warning|Exit)$")


def _callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return ""


class _Classifier(ast.NodeVisitor):
    """Map each string-literal node (by source span) to a context label."""

    def __init__(self) -> None:
        self.kinds: Dict[Tuple[int, int, int, int], str] = {}
        self._stack: List[str] = []

    # -- helpers -----------------------------------------------------------
    def _mark(self, node: ast.AST, kind: str) -> None:
        for sub in _display_parts(node):
            key = _span(sub)
            if key is None:
                continue
            prev = self.kinds.get(key)
            if prev is None or prev == "other_string":
                self.kinds[key] = kind

    def _docstring(self, node) -> None:
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and _is_str_node(body[0].value) \
                and isinstance(body[0].value, ast.Constant):
            key = _span(body[0].value)
            if key is not None:
                self.kinds[key] = "docstring"

    # -- visitors -----------------------------------------------------------
    def visit_Module(self, node):
        self._docstring(node)
        self.generic_visit(node)

    def visit_FunctionDef(self, node):
        self._docstring(node)
        self.generic_visit(node)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        self._docstring(node)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        name = _callee_name(node.func)
        if name in DISPLAY_CALLEES or EXCEPTION_NAMES.search(name or "x"):
            for a in node.args:
                self._mark(a, "display_string")
        for kw in node.keywords:
            if kw.arg in DISPLAY_KEYWORDS:
                self._mark(kw.value, "display_string")
            elif kw.arg == "labels" and name in PLOT_LABELS_CALLEES:
                self._mark(kw.value, "display_string")
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise):
        if node.exc is not None:
            self._mark(node.exc, "display_string")
        self.generic_visit(node)

    def visit_Assert(self, node: ast.Assert):
        if node.msg is not None:
            self._mark(node.msg, "display_string")
        self.generic_visit(node)

    def visit_JoinedStr(self, node):
        key = _span(node)
        if key is not None and key not in self.kinds:
            self.kinds[key] = "other_string"
        # Do not descend: literal parts are handled with the whole f-string.

    def visit_Constant(self, node):
        if _is_str_node(node):
            key = _span(node)
            if key is not None and key not in self.kinds:
                self.kinds[key] = "other_string"


def _display_parts(node: ast.AST):
    """
    Yield the string nodes that are rendered as text when `node` is shown.

    Only direct content counts: the literal itself, string operands of `+`
    and `%`, the receiver of `.format()`, both branches of a conditional,
    elements of a list or tuple, and the arguments of an exception
    constructor.  Strings used as subscripts, dict keys or arguments of
    other calls are not display text (``print(df["GT count"])`` prints a
    value looked up with a key).
    """
    if _is_str_node(node):
        yield node
    elif isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mod)):
        yield from _display_parts(node.left)
        if isinstance(node.op, ast.Add):
            yield from _display_parts(node.right)
    elif isinstance(node, ast.IfExp):
        yield from _display_parts(node.body)
        yield from _display_parts(node.orelse)
    elif isinstance(node, (ast.List, ast.Tuple)):
        for elt in node.elts:
            yield from _display_parts(elt)
    elif isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "format":
            yield from _display_parts(func.value)
        elif EXCEPTION_NAMES.search(_callee_name(func) or "x"):
            for a in node.args:
                yield from _display_parts(a)
        elif _callee_name(func) in ("str", "Markdown", "HTML", "dedent", "join"):
            for a in node.args:
                yield from _display_parts(a)
            if isinstance(func, ast.Attribute) and func.attr == "join":
                yield from _display_parts(func.value)


def _is_str_node(node) -> bool:
    if isinstance(node, ast.JoinedStr):
        return True
    return isinstance(node, ast.Constant) and isinstance(node.value, str)


def _span(node) -> Optional[Tuple[int, int, int, int]]:
    if getattr(node, "end_lineno", None) is None:
        return None
    return (node.lineno, node.col_offset, node.end_lineno, node.end_col_offset)


def _byte_to_char_col(line: str, byte_col: int) -> int:
    return len(line.encode("utf-8")[:byte_col].decode("utf-8", errors="replace"))


@dataclass
class _StrTok:
    start: Tuple[int, int]   # (row, char col)
    end: Tuple[int, int]
    text: str


def _string_tokens(source: str) -> Tuple[List[_StrTok], List[_StrTok]]:
    """Return (string literal tokens, comment tokens) with char positions.

    On Python >= 3.12 an f-string is several tokens; they are merged back
    into one span covering the complete literal.
    """
    strings: List[_StrTok] = []
    comments: List[_StrTok] = []
    lines = io.StringIO(source).readlines()
    fstart_type = getattr(tokenize, "FSTRING_START", None)
    fend_type = getattr(tokenize, "FSTRING_END", None)
    depth = 0
    fstart_pos: Optional[Tuple[int, int]] = None
    for tok in tokenize.generate_tokens(io.StringIO(source).readline):
        if fstart_type is not None and tok.type == fstart_type:
            if depth == 0:
                fstart_pos = tok.start
            depth += 1
            continue
        if fend_type is not None and tok.type == fend_type:
            depth -= 1
            if depth == 0 and fstart_pos is not None:
                strings.append(_StrTok(fstart_pos, tok.end,
                                       _slice(lines, fstart_pos, tok.end)))
                fstart_pos = None
            continue
        if depth > 0:
            continue
        if tok.type == tokenize.STRING:
            strings.append(_StrTok(tok.start, tok.end, tok.string))
        elif tok.type == tokenize.COMMENT:
            comments.append(_StrTok(tok.start, tok.end, tok.string))
    return strings, comments


def _slice(lines: List[str], start: Tuple[int, int], end: Tuple[int, int]) -> str:
    (r0, c0), (r1, c1) = start, end
    if r0 == r1:
        return lines[r0 - 1][c0:c1]
    parts = [lines[r0 - 1][c0:]]
    parts.extend(lines[r0:r1 - 1])
    parts.append(lines[r1 - 1][:c1])
    return "".join(parts)


_PREFIX_RE = re.compile(r"^([rRbBuUfF]{0,2})('''|\"\"\"|'|\")", re.S)

# Escape sequences inside a non-raw literal (kept as fixed separators).
_ESCAPE = re.compile(
    r"\\(?:\n|x[0-9a-fA-F]{2}|u[0-9a-fA-F]{4}|U[0-9a-fA-F]{8}|N\{[^}]*\}|[0-7]{1,3}|.)",
    re.S,
)


def _edit_string_token(tok_text: str, rules) -> Tuple[str, int, int, bool]:
    """
    Apply rules to the literal text of one string token.

    Returns (new_token_text, n_replacements, n_protected, is_bytes).
    Only literal characters are edited: f-string replacement fields and
    escape sequences are left alone.
    """
    m = _PREFIX_RE.match(tok_text)
    if not m:
        return tok_text, 0, 0, False
    prefix, quote = m.group(1), m.group(2)
    if "b" in prefix.lower():
        return tok_text, 0, 0, True
    body = tok_text[len(prefix) + len(quote): len(tok_text) - len(quote)]
    is_f = "f" in prefix.lower()
    segments = _split_fstring(body) if is_f else [(body, True)]
    if "r" not in prefix.lower():
        # An escape such as \n is a word boundary in the string's value, so
        # it is kept as a fixed separator while the source text is edited.
        split: List[Tuple[str, bool]] = []
        for seg, editable in segments:
            if not editable:
                split.append((seg, False))
                continue
            pos = 0
            for esc in _ESCAPE.finditer(seg):
                split.append((seg[pos:esc.start()], True))
                split.append((esc.group(0), False))
                pos = esc.end()
            split.append((seg[pos:], True))
        segments = split
    new_parts: List[str] = []
    n_rep = n_prot = 0
    for seg, editable in segments:
        if not editable or not seg:
            new_parts.append(seg)
            continue
        # Never let an edit create or destroy an escape sequence.
        new_seg, r, p = transform_prose(seg, rules)
        n_rep += r
        n_prot += p
        new_parts.append(new_seg)
    return prefix + quote + "".join(new_parts) + quote, n_rep, n_prot, False


def _split_fstring(body: str) -> List[Tuple[str, bool]]:
    """Split an f-string body into (text, editable) runs."""
    out: List[Tuple[str, bool]] = []
    i = 0
    buf: List[str] = []
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "{":
            if i + 1 < n and body[i + 1] == "{":
                buf.append("{{")
                i += 2
                continue
            if buf:
                out.append(("".join(buf), True))
                buf = []
            depth = 0
            j = i
            quote_char = ""
            while j < n:
                c = body[j]
                if quote_char:
                    if c == quote_char:
                        quote_char = ""
                elif c in "'\"":
                    quote_char = c
                elif c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            out.append((body[i:j + 1], False))
            i = j + 1
            continue
        if ch == "}" and i + 1 < n and body[i + 1] == "}":
            buf.append("}}")
            i += 2
            continue
        buf.append(ch)
        i += 1
    if buf:
        out.append(("".join(buf), True))
    return out


def _normalise_tree(tree: ast.AST) -> str:
    """AST dump with every string constant replaced by a placeholder."""
    class _N(ast.NodeTransformer):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                return ast.Constant(value="<S>")
            return node

    t = _N().visit(copy.deepcopy(tree))
    return ast.dump(t, include_attributes=False)


def _string_values(tree: ast.AST) -> List[str]:
    vals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            vals.append(node.value)
    return vals


def process_python_source(source: str, rel: str, rules, data: DataIndex,
                          line_offset: int = 0, kind_prefix: str = "") -> Tuple[str, List[Finding], str]:
    """
    Edit comments, docstrings and eligible string literals in `source`.

    Returns (new_source, findings, error). `error` is non-empty if the
    source could not be parsed or the verification failed; in that case the
    returned source is the original.
    """
    findings: List[Finding] = []
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return source, findings, f"SyntaxError: {exc.msg} (line {exc.lineno})"

    classifier = _Classifier()
    classifier.visit(tree)
    lines = io.StringIO(source).readlines()

    # char-coordinate span -> kind
    kinds_char: Dict[Tuple[int, int, int, int], str] = {}
    for (l0, c0, l1, c1), kind in classifier.kinds.items():
        cc0 = _byte_to_char_col(lines[l0 - 1], c0) if l0 - 1 < len(lines) else c0
        cc1 = _byte_to_char_col(lines[l1 - 1], c1) if l1 - 1 < len(lines) else c1
        kinds_char[(l0, cc0, l1, cc1)] = kind

    try:
        str_toks, com_toks = _string_tokens(source)
    except (tokenize.TokenError, IndentationError, SyntaxError) as exc:
        return source, findings, f"TokenizeError: {exc}"

    def kind_for(tok: _StrTok) -> str:
        best = None
        for (l0, c0, l1, c1), kind in kinds_char.items():
            if (l0, c0) <= tok.start and tok.end <= (l1, c1):
                size = (l1 - l0, c1 - c0)
                if best is None or size < best[0]:
                    best = (size, kind)
        return best[1] if best else "other_string"

    edits: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = []
    planned_values: Dict[str, str] = {}

    for tok in com_toks:
        new, n_rep, n_prot = transform_prose(tok.text, rules)
        line_no = tok.start[0] + line_offset
        if n_rep:
            edits.append((tok.start, tok.end, new))
            findings.append(Finding(rel, line_no, kind_prefix + "comment", "APPLIED",
                                    f"{n_rep} replacement(s)", _snip(tok.text), _snip(new)))
        if n_prot:
            findings.append(Finding(rel, line_no, kind_prefix + "comment", "KEPT",
                                    "archive wording, quoted name or code span", _snip(tok.text)))

    for tok in str_toks:
        if not _has_term(tok.text, rules) and not ARCHIVE_LINE.search(tok.text):
            continue
        new_text, n_rep, n_prot, is_bytes = _edit_string_token(tok.text, rules)
        line_no = tok.start[0] + line_offset
        kind = kind_for(tok)
        if is_bytes:
            continue
        if n_prot and not n_rep:
            findings.append(Finding(rel, line_no, kind_prefix + kind, "KEPT",
                                    "archive wording, quoted name or code span", _snip(tok.text)))
        if not n_rep:
            continue
        try:
            old_val = ast.literal_eval(tok.text) if not _is_fstring(tok.text) else None
            new_val = ast.literal_eval(new_text) if not _is_fstring(new_text) else None
        except Exception:
            old_val = new_val = None

        if kind in ("docstring", "display_string"):
            action, reason = "APPLIED", kind.replace("_", " ")
        else:
            value = old_val if isinstance(old_val, str) else _literal_text(tok.text)
            if data.contains(value):
                action, reason = "REVIEW", "value also occurs in a data file (possible lookup key)"
            elif _is_identifier_like(value):
                action, reason = "REVIEW", "single token; may be a key rather than display text"
            else:
                action, reason = "APPLIED", "prose-like literal, not found in any data file"
        findings.append(Finding(rel, line_no, kind_prefix + kind, action, reason,
                                _snip(tok.text), _snip(new_text)))
        if action == "APPLIED":
            edits.append((tok.start, tok.end, new_text))
            if isinstance(old_val, str) and isinstance(new_val, str):
                planned_values[old_val] = new_val

    if not edits:
        return source, findings, ""

    new_source = _apply_edits(lines, edits)

    # ---- verification ----------------------------------------------------
    try:
        new_tree = ast.parse(new_source)
    except SyntaxError as exc:
        return source, findings, f"verification: edited source does not parse ({exc.msg})"
    if _normalise_tree(tree) != _normalise_tree(new_tree):
        return source, findings, "verification: code structure changed"
    old_vals, new_vals = _string_values(tree), _string_values(new_tree)
    if len(old_vals) != len(new_vals):
        return source, findings, "verification: string constant count changed"
    for a, b in zip(old_vals, new_vals):
        if a == b:
            continue
        expected, _, _ = transform_prose(a, rules)
        if b != expected and b != planned_values.get(a, a):
            return source, findings, "verification: unexpected string change"
    return new_source, findings, ""


def _is_fstring(tok_text: str) -> bool:
    m = _PREFIX_RE.match(tok_text)
    return bool(m and "f" in m.group(1).lower())


def _literal_text(tok_text: str) -> str:
    m = _PREFIX_RE.match(tok_text)
    if not m:
        return tok_text
    prefix, quote = m.group(1), m.group(2)
    return tok_text[len(prefix) + len(quote): len(tok_text) - len(quote)]


def _is_identifier_like(value: str) -> bool:
    v = value.strip()
    return bool(v) and " " not in v and "\n" not in v and len(v) <= 40


def _apply_edits(lines: List[str], edits) -> str:
    """Apply (start, end, text) edits given in (row, char col) coordinates."""
    text = "".join(lines)
    starts = [0]
    for ln in lines:
        starts.append(starts[-1] + len(ln))

    def off(pos):
        r, c = pos
        return starts[r - 1] + c

    for start, end, new in sorted(edits, key=lambda e: off(e[0]), reverse=True):
        a, b = off(start), off(end)
        text = text[:a] + new + text[b:]
    return text


# ============================================================================
# 6.  Other file types
# ============================================================================

_FENCE = re.compile(r"^\s*(```|~~~)")


def process_markdown(text: str, rel: str, rules, kind: str = "markdown",
                     line_offset: int = 0) -> Tuple[str, List[Finding]]:
    findings: List[Finding] = []
    out: List[str] = []
    in_fence = False
    for i, line in enumerate(text.splitlines(keepends=True), start=1):
        if _FENCE.match(line):
            in_fence = not in_fence
            out.append(line)
            continue
        if in_fence or line.lstrip().startswith((">>>", "... ")):
            if _has_term(line, rules):
                findings.append(Finding(rel, i + line_offset, kind + "_code", "REVIEW",
                                        "inside a code block; edit by hand if it is prose",
                                        _snip(line)))
            out.append(line)
            continue
        new, n_rep, n_prot = transform_prose(line, rules)
        if n_rep:
            findings.append(Finding(rel, i + line_offset, kind, "APPLIED",
                                    f"{n_rep} replacement(s)", _snip(line), _snip(new)))
        if n_prot:
            findings.append(Finding(rel, i + line_offset, kind, "KEPT",
                                    "archive wording, quoted name or code span", _snip(line)))
        out.append(new)
    return "".join(out), findings


def process_yaml(text: str, rel: str, rules) -> Tuple[str, List[Finding]]:
    """Edit YAML comments only. Keys and values are data."""
    findings: List[Finding] = []
    out: List[str] = []
    for i, line in enumerate(text.splitlines(keepends=True), start=1):
        idx = _yaml_comment_index(line)
        if idx is None:
            if _has_term(line, rules):
                findings.append(Finding(rel, i, "yaml_value", "REVIEW",
                                        "YAML key or value; left as data", _snip(line)))
            out.append(line)
            continue
        head, comment = line[:idx], line[idx:]
        new_comment, n_rep, n_prot = transform_prose(comment, rules)
        if n_rep:
            findings.append(Finding(rel, i, "yaml_comment", "APPLIED",
                                    f"{n_rep} replacement(s)", _snip(line), _snip(head + new_comment)))
        if _has_term(head, rules):
            findings.append(Finding(rel, i, "yaml_value", "REVIEW",
                                    "YAML key or value; left as data", _snip(head)))
        out.append(head + new_comment)
    return "".join(out), findings


def _yaml_comment_index(line: str) -> Optional[int]:
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
            continue
        if ch in "'\"":
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return i
    return None


_SH_ECHO = re.compile(r"^(\s*(?:echo|printf|warn|fail|ok|step|die|info)\b)(.*)$")


def process_shell(text: str, rel: str, rules) -> Tuple[str, List[Finding]]:
    """Edit full-line comments and the quoted text of echo-like lines."""
    findings: List[Finding] = []
    out: List[str] = []
    for i, line in enumerate(text.splitlines(keepends=True), start=1):
        stripped = line.lstrip()
        if stripped.startswith("#!") or stripped.startswith("#SBATCH"):
            if _has_term(line, rules):
                findings.append(Finding(rel, i, "shell_directive", "REVIEW",
                                        "SLURM directive or shebang", _snip(line)))
            out.append(line)
            continue
        if stripped.startswith("#"):
            new, n_rep, _ = transform_prose(line, rules)
            if n_rep:
                findings.append(Finding(rel, i, "shell_comment", "APPLIED",
                                        f"{n_rep} replacement(s)", _snip(line), _snip(new)))
            out.append(new)
            continue
        body = line.rstrip("\r\n")
        eol = line[len(body):]
        m = _SH_ECHO.match(body)
        if m and _has_term(body, rules):
            new_rest = _edit_quoted(m.group(2), rules)
            new = m.group(1) + new_rest + eol
            if new != line:
                findings.append(Finding(rel, i, "shell_message", "APPLIED",
                                        "message text", _snip(line), _snip(new)))
            out.append(new)
            continue
        if _has_term(line, rules):
            findings.append(Finding(rel, i, "shell_code", "REVIEW",
                                    "shell code; left unchanged", _snip(line)))
        out.append(line)
    return "".join(out), findings


def _edit_quoted(s: str, rules) -> str:
    def repl(m):
        body = m.group(2)
        if "$" in body and re.search(r"\$\{?[A-Za-z_]*GT", body):
            return m.group(0)
        new, _, _ = transform_prose(body, rules)
        return m.group(1) + new + m.group(1)
    return re.sub(r"(['\"])(.*?)\1", repl, s)


# ============================================================================
# 7.  Notebooks
# ============================================================================

_MAGIC = re.compile(r"^\s*[%!]")


def process_notebook(raw: str, rel: str, rules, data: DataIndex) -> Tuple[str, List[Finding], str]:
    findings: List[Finding] = []
    try:
        nb = json.loads(raw, object_pairs_hook=OrderedDict)
    except json.JSONDecodeError as exc:
        return raw, findings, f"invalid JSON: {exc}"
    indent = _detect_indent(raw)
    trailing_nl = raw.endswith("\n")

    def dump(obj) -> str:
        s = json.dumps(obj, indent=indent, ensure_ascii=False, separators=(",", ": "))
        return s + ("\n" if trailing_nl else "")

    roundtrip_exact = dump(nb) == raw
    cells = nb.get("cells", [])
    changed = False
    output_hits = 0
    for ci, cell in enumerate(cells):
        ctype = cell.get("cell_type")
        src = cell.get("source", "")
        was_list = isinstance(src, list)
        text = "".join(src) if was_list else str(src)
        where = f"{rel} [cell {ci}]"
        if ctype == "markdown":
            new_text, f = process_markdown(text, where, rules, kind="nb_markdown")
            findings.extend(f)
        elif ctype == "code":
            if text.lstrip().startswith("%%"):
                if _has_term(text, rules):
                    findings.append(Finding(where, 1, "nb_cell_magic", "REVIEW",
                                            "cell magic; not Python", _snip(text[:120])))
                new_text = text
            else:
                masked = "\n".join(_mask_magic(l) for l in text.split("\n"))
                new_masked, f, err = process_python_source(masked, where, rules, data,
                                                           kind_prefix="nb_")
                findings.extend(f)
                if err:
                    findings.append(Finding(where, 1, "nb_code", "REVIEW",
                                            f"cell skipped: {err}", _snip(text[:120])))
                    new_text = text
                else:
                    new_text = _restore_magics(text, new_masked)
            for out in cell.get("outputs", []) or []:
                blob = json.dumps(out, ensure_ascii=False)
                if _has_term(blob, rules):
                    output_hits += 1
        else:
            new_text = text
        if new_text != text:
            changed = True
            if was_list:
                cell["source"] = new_text.splitlines(keepends=True)
            else:
                cell["source"] = new_text
    if output_hits:
        findings.append(Finding(rel, 0, "nb_outputs", "INFO",
                                f"{output_hits} output(s) still show old wording; re-run the notebook",
                                ""))
    if not changed:
        return raw, findings, ""
    new_raw = dump(nb)
    if not roundtrip_exact:
        findings.append(Finding(rel, 0, "nb_format", "INFO",
                                "notebook JSON formatting was normalized on write "
                                "(the diff is larger than the text change)", ""))
    try:
        json.loads(new_raw)
    except json.JSONDecodeError as exc:
        return raw, findings, f"verification: rewritten notebook is not valid JSON ({exc})"
    return new_raw, findings, ""


def _mask_magic(line: str) -> str:
    m = _MAGIC.match(line)
    if not m:
        return line
    indent = line[: len(line) - len(line.lstrip())]
    return indent + "pass"


def _restore_magics(original: str, edited_masked: str) -> str:
    o_lines = original.split("\n")
    e_lines = edited_masked.split("\n")
    if len(o_lines) != len(e_lines):
        return original
    return "\n".join(o if _MAGIC.match(o) else e for o, e in zip(o_lines, e_lines))


def _detect_indent(raw: str) -> int:
    for line in raw.splitlines()[1:6]:
        stripped = line.lstrip(" ")
        if stripped and len(stripped) != len(line):
            return len(line) - len(stripped)
    return 1


# ============================================================================
# 8.  Walking the tree
# ============================================================================

def _is_protected(path: Path) -> bool:
    if path.name in PROTECTED_FILE_NAMES:
        return True
    return any(re.search(p, path.name) for p in PROTECTED_NAME_PATTERNS)


def iter_files(root: Path, include_hidden: bool = False) -> Iterable[Path]:
    root = root.resolve()
    skip_abs = [(root / d).resolve() for d in SKIP_REL_DIRS]
    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        keep = []
        for d in sorted(dirnames):
            full = (dp / d).resolve()
            if d in SKIP_DIR_NAMES or any(full == s for s in skip_abs):
                continue
            if d.startswith(".") and not include_hidden and d not in (".github",):
                continue
            keep.append(d)
        dirnames[:] = keep
        for fn in sorted(filenames):
            yield dp / fn


def iter_data_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "__pycache__",
                                                         ".ipynb_checkpoints", "node_modules"}]
        for fn in filenames:
            p = Path(dirpath) / fn
            if p.suffix.lower() in DATA_EXT and p.suffix.lower() != ".ipynb":
                yield p


def git_dirty_files(root: Path) -> Optional[Set[Path]]:
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        out = subprocess.run(["git", "-C", top, "status", "--porcelain=v1", "-uno"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    dirty = set()
    for line in out.splitlines():
        if len(line) > 3:
            dirty.add((Path(top) / line[3:].split(" -> ")[-1]).resolve())
    return dirty


def git_tracked(root: Path) -> Optional[Set[Path]]:
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=True).stdout.strip()
        out = subprocess.run(["git", "-C", top, "ls-files", "-z"],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    return {(Path(top) / p).resolve() for p in out.split("\0") if p}


def process_file(path: Path, root: Path, rules, data: DataIndex) -> Optional[FileResult]:
    ext = path.suffix.lower()
    handled = PY_EXT | NB_EXT | MD_EXT | YAML_EXT | SH_EXT | REPORT_ONLY_EXT
    if ext not in handled:
        return None
    try:
        raw = path.read_bytes()
    except OSError as exc:
        return FileResult(path, _rel(path, root), "", "", [], f"unreadable: {exc}")
    if b"\x00" in raw[:4096]:
        return None
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return FileResult(path, _rel(path, root), "", "", [], "not UTF-8; skipped")
    rel = _rel(path, root)
    res = FileResult(path, rel, text, text)
    if not (_has_term(text, rules) or ARCHIVE_LINE.search(text)):
        return res
    if _is_protected(path):
        if _has_term(text, rules):
            res.findings.append(Finding(rel, 0, "protected_file", "KEPT",
                                        "produces or verifies the BioImage Archive deposition", ""))
        return res
    if "\r\n" in text:
        res.findings.append(Finding(rel, 0, "line_endings", "INFO",
                                    "file uses CRLF line endings; they are preserved", ""))
    if ext in PY_EXT:
        new, f, err = process_python_source(text, rel, rules, data)
        res.new_text, res.findings, res.error = new, f, err
    elif ext in NB_EXT:
        new, f, err = process_notebook(text, rel, rules, data)
        res.new_text, res.findings, res.error = new, f, err
    elif ext in MD_EXT:
        res.new_text, res.findings = process_markdown(text, rel, rules)
    elif ext in YAML_EXT:
        res.new_text, res.findings = process_yaml(text, rel, rules)
    elif ext in SH_EXT:
        res.new_text, res.findings = process_shell(text, rel, rules)
    else:
        for i, line in enumerate(text.splitlines(), start=1):
            if _has_term(line, rules):
                res.findings.append(Finding(rel, i, "config_text", "REVIEW",
                                            "metadata file; edit by hand if needed", _snip(line)))
    if res.error:
        res.new_text = res.old_text
    return res


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


# ============================================================================
# 9.  Informational audits
# ============================================================================

def extra_audits(results: List[FileResult]) -> Dict[str, List[str]]:
    audits: Dict[str, List[str]] = defaultdict(list)
    for r in results:
        text = r.new_text if r.changed else r.old_text
        if not text:
            continue
        for i, line in enumerate(text.splitlines(), start=1):
            if STALE_DIAMOND.search(line) and re.search(r"(?i)diamond", line):
                audits["stale '7 x 7 diamond' wording (manuscript says 5 x 5-pixel diamonds)"].append(
                    f"{r.rel}:{i}: {_snip(line.strip(), 120)}")
            if ABS_PATH.search(line) and r.path.suffix.lower() in (".py", ".ipynb"):
                audits["absolute Longleaf paths in code (will not work elsewhere)"].append(
                    f"{r.rel}:{i}: {_snip(line.strip(), 120)}")
            if LOWER_GT.search(line) and re.search(r"#|\"\"\"|'''", line) and \
                    r.path.suffix.lower() == ".py":
                audits["lowercase 'gt' in comments/docstrings (left as identifier-style)"].append(
                    f"{r.rel}:{i}")
            if DETECT_FIGURE.search(line) and r.path.suffix.lower() in (".py", ".ipynb", ".md"):
                audits["figure-label terms (LabelEngine / density bin / pixel Dice)"].append(
                    f"{r.rel}:{i}: {_snip(line.strip(), 120)}")
    return audits


# ============================================================================
# 10.  Reporting
# ============================================================================

def write_reports(results: List[FileResult], out_dir: Path, mode: str, roots: List[Path],
                  data: DataIndex, rules_desc: str) -> Tuple[int, int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    findings = [f for r in results for f in r.findings]
    with open(out_dir / "rename_report.tsv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["file", "line", "kind", "action", "reason", "before", "after"])
        for f in findings:
            w.writerow([f.path, f.line, f.kind, f.action, f.reason, f.before, f.after])

    patch_chunks = []
    for r in results:
        if r.changed:
            patch_chunks.extend(difflib.unified_diff(
                r.old_text.splitlines(keepends=True), r.new_text.splitlines(keepends=True),
                fromfile=f"a/{r.rel}", tofile=f"b/{r.rel}"))
    patch = "".join(patch_chunks)
    (out_dir / "rename_gold_standard.patch").write_text(patch, encoding="utf-8")

    changed = [r for r in results if r.changed]
    errors = [r for r in results if r.error]
    by_action = Counter(f.action for f in findings)
    by_kind = Counter((f.kind, f.action) for f in findings)
    per_file = Counter(f.path.split(" [cell")[0] for f in findings if f.action == "APPLIED")
    review = [f for f in findings if f.action == "REVIEW"]
    audits = extra_audits(results)

    lines = []
    lines.append(f"rename_gold_standard.py v{__version__} | {mode.upper()} | "
                 f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("roots: " + ", ".join(str(r) for r in roots))
    lines.append(f"rules: {rules_desc}")
    lines.append(f"python: {sys.version.split()[0]}")
    lines.append("")
    lines.append(f"files examined      : {len(results)}")
    lines.append(f"files with edits    : {len(changed)}"
                 + (" (written)" if mode == "apply" else " (preview only; nothing written)"))
    lines.append(f"files with errors   : {len(errors)}")
    lines.append(f"edits applied       : {by_action.get('APPLIED', 0)}")
    lines.append(f"items to review     : {by_action.get('REVIEW', 0)}")
    lines.append(f"kept on purpose     : {by_action.get('KEPT', 0)}")
    lines.append(f"data files checked  : {data.files_scanned} "
                 f"({len(data.values)} distinct term-bearing values)")
    for s in data.files_skipped[:10]:
        lines.append(f"  data file skipped : {s}")
    lines.append("")
    lines.append("EDITS BY KIND")
    for (kind, action), n in sorted(by_kind.items()):
        lines.append(f"  {kind:<28} {action:<8} {n}")
    lines.append("")
    lines.append("FILES WITH EDITS")
    for path, n in sorted(per_file.items()):
        lines.append(f"  {n:>4}  {path}")
    if errors:
        lines.append("")
        lines.append("ERRORS (these files were NOT changed)")
        for r in errors:
            lines.append(f"  {r.rel}: {r.error}")
    lines.append("")
    lines.append("REVIEW BY HAND (not changed)")
    if not review:
        lines.append("  none")
    for f in review[:400]:
        lines.append(f"  {f.path}:{f.line}  [{f.kind}] {f.reason}")
        lines.append(f"      {f.before}")
    if len(review) > 400:
        lines.append(f"  ... {len(review) - 400} more in rename_report.tsv")
    info = [f for f in findings if f.action == "INFO"]
    if info:
        lines.append("")
        lines.append("NOTES")
        for f in info:
            lines.append(f"  {f.path}: {f.reason}")
    for title, items in audits.items():
        lines.append("")
        lines.append(f"AUDIT: {title}: {len(items)}")
        for it in items[:60]:
            lines.append(f"  {it}")
        if len(items) > 60:
            lines.append(f"  ... {len(items) - 60} more")
    lines.append("")
    if mode == "apply":
        verdict = ("APPLIED CLEANLY" if not errors else
                   f"APPLIED WITH {len(errors)} FILE(S) LEFT UNTOUCHED (see ERRORS)")
    else:
        verdict = ("READY TO APPLY" if not errors else
                   f"{len(errors)} FILE(S) NEED ATTENTION BEFORE APPLY (see ERRORS)")
    lines.append(f"VERDICT: {verdict}; {len(review)} item(s) to review by hand")
    lines.append("")
    lines.append("OUTPUT FILES")
    lines.append(f"  {out_dir / 'rename_summary.txt'}")
    lines.append(f"  {out_dir / 'rename_report.tsv'}")
    lines.append(f"  {out_dir / 'rename_gold_standard.patch'}  ({len(patch.splitlines())} lines)")
    summary = "\n".join(lines) + "\n"
    (out_dir / "rename_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    return len(changed), len(errors), by_action.get("REVIEW", 0)


# ============================================================================
# 11.  Main
# ============================================================================

def run(mode: str, roots: List[Path], out_dir: Path, figure_terms: bool,
        allow_dirty: bool, backup_dir: Optional[Path], data_roots: List[Path],
        max_data_mb: float, skip_errors: bool = False) -> int:
    rules = active_rules(figure_terms)
    rules_desc = "core (ground truth -> gold standard, GT -> GS, true count -> gold-standard count)"
    if figure_terms:
        rules_desc += " + figure terms (LabelEngine, density bin, pixel Dice)"

    data = DataIndex(max_mb=max_data_mb)
    for dr in list(roots) + list(data_roots):
        for p in iter_data_files(dr):
            data.add_file(p)

    results: List[FileResult] = []
    for root in roots:
        for p in iter_files(root):
            r = process_file(p, root, rules, data)
            if r is not None:
                if len(roots) > 1:
                    r.rel = f"{root.name}/{r.rel}"
                results.append(r)

    if mode == "apply":
        errored = [r for r in results if r.error]
        if errored and not skip_errors:
            print("REFUSING TO APPLY: some files could not be processed safely "
                  "(nothing was changed). Fix them, or pass --skip-errors to apply "
                  "the other files and leave these untouched:")
            for r in errored:
                print(f"  {r.rel}: {r.error}")
            write_reports(results, out_dir, "scan", roots, data, rules_desc)
            return 3
        changed = [r for r in results if r.changed]
        dirty = None
        tracked = None
        for root in roots:
            d = git_dirty_files(root)
            t = git_tracked(root)
            if d is not None:
                dirty = (dirty or set()) | d
            if t is not None:
                tracked = (tracked or set()) | t
        blocked = [r for r in changed if dirty and r.path.resolve() in dirty]
        if blocked and not allow_dirty:
            print("REFUSING TO APPLY: these files have uncommitted changes. Commit or stash "
                  "them first (or pass --allow-dirty):")
            for r in blocked:
                print(f"  {r.rel}")
            write_reports(results, out_dir, "scan", roots, data, rules_desc)
            return 3
        not_writable = [r for r in changed if not os.access(r.path, os.W_OK)]
        if not_writable:
            print("REFUSING TO APPLY: these files are not writable (nothing was changed):")
            for r in not_writable:
                print(f"  {r.rel}")
            write_reports(results, out_dir, "scan", roots, data, rules_desc)
            return 3
        untracked = [r for r in changed if not tracked or r.path.resolve() not in tracked]
        if untracked:
            bdir = backup_dir or (out_dir / "backup")
            for r in untracked:
                dest = bdir / r.rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(r.path, dest)
            print(f"backed up {len(untracked)} file(s) that git does not track to {bdir}")
        for r in changed:
            mode_bits = r.path.stat().st_mode
            with open(r.path, "w", encoding="utf-8", newline="") as fh:
                fh.write(r.new_text)
            os.chmod(r.path, mode_bits)
        # Re-read and re-verify what is now on disk.
        bad = []
        for r in changed:
            on_disk = r.path.read_text(encoding="utf-8")
            if on_disk != r.new_text:
                bad.append(r.rel)
            if r.path.suffix == ".py":
                try:
                    ast.parse(on_disk)
                except SyntaxError:
                    bad.append(r.rel)
            if r.path.suffix == ".ipynb":
                try:
                    json.loads(on_disk)
                except json.JSONDecodeError:
                    bad.append(r.rel)
        if bad:
            print("POST-WRITE CHECK FAILED for: " + ", ".join(sorted(set(bad))))
            write_reports(results, out_dir, mode, roots, data, rules_desc)
            return 2

    n_changed, n_err, n_review = write_reports(results, out_dir, mode, roots, data, rules_desc)
    if n_err:
        return 1
    return 0


# ============================================================================
# 12.  Self-test
# ============================================================================

def self_test() -> int:
    rules = active_rules(False)
    frules = active_rules(True)
    data = DataIndex()
    data.values.update({"GT (points)", "gt_count"})
    failures = []

    def check(name, got, want):
        if got != want:
            failures.append(f"{name}\n    got:  {got!r}\n    want: {want!r}")

    t = lambda s, r=rules: transform_prose(s, r)[0]
    check("noun", t("the ground truth is noisy"), "the gold standard is noisy")
    check("adjective", t("the ground truth mask"), "the gold-standard mask")
    check("hyphen", t("ground-truth objects"), "gold-standard objects")
    check("capital", t("Ground truth count"), "Gold-standard count")
    check("plural", t("predictions and ground truths"), "predictions and gold-standard objects")
    check("GT word", t("GT mask (GT = 116)"), "GS mask (GS = 116)")
    check("GTs", t("GTs that nobody matched"), "GS objects that nobody matched")
    check("GT sentence end", t("compare with GT."), "compare with GS.")
    check("GT hyphen", t("any GT-positive pixel, non-GT"), "any GS-positive pixel, non-GS")
    check("identifier kept", t("gt_mask GT_TYPE ecDNA_gt"), "gt_mask GT_TYPE ecDNA_gt")
    check("path kept", t("see gt/GT.png and /GT/ dir"), "see gt/GT.png and /GT/ dir")
    check("format kept", t("{GT} %(GT)s $GT ${GT}"), "{GT} %(GT)s $GT ${GT}")
    check("code span kept", t("column `GT count` and ``ground truth``"),
          "column `GT count` and ``ground truth``")
    check("quoted name kept", t('the "ground truth" section'), 'the "ground truth" section')
    check("archive line kept", t("BioImage Archive section: ground truth"),
          "BioImage Archive section: ground truth")
    check("true count", t("True count vs true counts; true positives"),
          "Gold-standard count vs gold-standard counts; true positives")
    check("url kept", t("https://x.org/ground-truth ground-truth"),
          "https://x.org/ground-truth gold-standard")
    check("figure terms off", t("LabelEngine density bin"), "LabelEngine density bin")
    check("figure terms on", t("LabelEngine density bins", frules), "Label Engine count bins")
    check("GTX kept", t("NVIDIA GTX and SGT"), "NVIDIA GTX and SGT")

    src = '''"""Module about ground truth masks."""
import logging
logger = logging.getLogger(__name__)
LABELS = {"gt": "Ground truth"}           # ground truth label map
KEY = "GT (points)"
TOKEN = "GT"
def f(gt_mask, x):
    """Load the GT mask."""
    ax = None
    print(f"GT count = {len(x)} for {x!r} {{GT}}")
    logger.info("ground-truth objects: %d", 3)
    if gt_mask is None:
        raise ValueError("GT mask missing: " + str(x))
    y = x.gt(0)
    return {"gt_count": 1, "label": "Ground truth diamonds"}
'''
    new, findings, err = process_python_source(src, "t.py", rules, data)
    check("py error", err, "")
    check("py module docstring", '"""Module about gold-standard masks."""' in new, True)
    check("py function docstring", '"""Load the GS mask."""' in new, True)
    check("py comment", "# gold-standard label map" in new, True)
    check("py dict value applied", '{"gt": "Gold standard"}' in new, True)
    check("py data-bound kept", 'KEY = "GT (points)"' in new, True)
    check("py single token kept", 'TOKEN = "GT"' in new, True)
    check("py fstring", 'print(f"GS count = {len(x)} for {x!r} {{GS}}")' in new
          or 'print(f"GS count = {len(x)} for {x!r} {{GT}}")' in new, True)
    check("py log", '"gold-standard objects: %d"' in new, True)
    check("py raise", '"GS mask missing: "' in new, True)
    check("py method kept", "x.gt(0)" in new, True)
    check("py identifiers kept", "def f(gt_mask, x):" in new and '"gt_count"' in new, True)
    check("py dict prose value", '"Gold-standard diamonds"' in new, True)
    try:
        compile(new, "t.py", "exec")
    except SyntaxError as exc:
        failures.append(f"edited source does not compile: {exc}")

    nb = OrderedDict([
        ("cells", [
            OrderedDict([("cell_type", "markdown"), ("metadata", OrderedDict()),
                         ("source", ["# Ground truth\n", "Compare with `gt_mask` and GT.\n"])]),
            OrderedDict([("cell_type", "code"), ("execution_count", None),
                         ("metadata", OrderedDict()), ("outputs", []),
                         ("source", ["%matplotlib inline\n",
                                     "import matplotlib.pyplot as plt\n",
                                     "plt.xlabel('GT count')  # ground truth\n",
                                     "!echo ground truth"])]),
        ]),
        ("metadata", OrderedDict()), ("nbformat", 4), ("nbformat_minor", 5),
    ])
    raw = json.dumps(nb, indent=1, ensure_ascii=False, separators=(",", ": ")) + "\n"
    new_raw, nf, nerr = process_notebook(raw, "t.ipynb", rules, data)
    check("nb error", nerr, "")
    nb2 = json.loads(new_raw)
    check("nb markdown", nb2["cells"][0]["source"],
          ["# Gold standard\n", "Compare with `gt_mask` and GS.\n"])
    check("nb code", nb2["cells"][1]["source"],
          ["%matplotlib inline\n", "import matplotlib.pyplot as plt\n",
           "plt.xlabel('GS count')  # gold standard\n", "!echo ground truth"])

    y = "paths:\n  gt_mask_dir: /x/gt_image   # rendered ground truth masks\n  label: \"GT\"\n"
    ny, _ = process_yaml(y, "t.yaml", rules)
    check("yaml", ny, "paths:\n  gt_mask_dir: /x/gt_image   # rendered gold-standard masks\n"
                      "  label: \"GT\"\n")

    sh = '#!/bin/bash\n#SBATCH -J verify_gt\n# check ground truth\necho "Scoring GT masks"\nGT_DIR=$1\n'
    nsh, _ = process_shell(sh, "t.sh", rules)
    check("shell", nsh, '#!/bin/bash\n#SBATCH -J verify_gt\n# check gold standard\n'
                        'echo "Scoring GS masks"\nGT_DIR=$1\n')

    md = "Use the ground truth.\n```bash\n# ground truth here\n```\nSee S-BIAD4097 ground truth.\n"
    nmd, _ = process_markdown(md, "t.md", rules)
    check("markdown", nmd, "Use the gold standard.\n```bash\n# ground truth here\n```\n"
                           "See S-BIAD4097 ground truth.\n")

    if failures:
        print("SELF-TEST FAILED")
        for f in failures:
            print("  - " + f)
        return 1
    print(f"SELF-TEST PASSED (python {sys.version.split()[0]})")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["scan", "apply"], default="scan")
    ap.add_argument("--root", type=Path, default=Path("."),
                    help="repository root to process (default: current directory)")
    ap.add_argument("--extra-root", type=Path, action="append", default=[],
                    help="additional folder to process, e.g. notebooks outside the repo")
    ap.add_argument("--data-root", type=Path, action="append", default=[],
                    help="additional folder whose CSV/TSV/JSON files are checked")
    ap.add_argument("--out-dir", type=Path, default=Path("gold_standard_rename"),
                    help="where the report, summary and patch are written")
    ap.add_argument("--figure-terms", action="store_true",
                    help="also apply LabelEngine -> Label Engine, density bin -> count bin, "
                         "pixel Dice -> pixel-level Dice")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="apply even to files with uncommitted git changes")
    ap.add_argument("--backup-dir", type=Path, default=None,
                    help="backup location for files not tracked by git")
    ap.add_argument("--skip-errors", action="store_true",
                    help="apply to the files that verified cleanly even if others failed")
    ap.add_argument("--max-data-mb", type=float, default=200.0,
                    help="skip data files larger than this in the data check")
    ap.add_argument("--self-test", action="store_true", help="run built-in tests and exit")
    ap.add_argument("--version", action="version", version=__version__)
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()

    roots = [args.root.resolve()] + [p.resolve() for p in args.extra_root]
    for r in roots:
        if not r.is_dir():
            print(f"not a directory: {r}", file=sys.stderr)
            return 2
    rc = self_test()
    if rc:
        print("Built-in self-test failed; nothing was changed.", file=sys.stderr)
        return rc
    return run(args.mode, roots, args.out_dir.expanduser().resolve(), args.figure_terms,
               args.allow_dirty, args.backup_dir, [p.resolve() for p in args.data_root],
               args.max_data_mb, args.skip_errors)


if __name__ == "__main__":
    sys.exit(main())
