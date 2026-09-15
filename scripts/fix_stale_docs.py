#!/usr/bin/env python3
"""
scripts/fix_stale_docs.py
=========================
Correct statements in comments, docstrings and documentation that no longer
match the paper, without changing what any code does.

Run AFTER scripts/rename_gold_standard.py (the patterns accept both "GT" and
"GS" spellings, so the order is not critical).

What it fixes
-------------
  diamonds   Gold-standard points are rendered as 5 x 5 diamonds (13 px), not
             7 x 7 (metadata.py, samples.py, postprocess.py, docs).
  count      ``ecDNA_gt`` is the number of mask components, not the number of
             annotated points (samples.py).
  checkpoint The released ecCount checkpoint is from epoch 49, validation loss
             0.6421 (train.py said epoch 58, 0.6415).
  disks      ecCount peaks are drawn as diamonds, not disks (run_eccount.py,
             configs/default.yaml).
  config     Comment fixes in configs/default.yaml: resource size 2,986;
             template name paths.example.yaml; no "novel"; a note that the
             benchmark model labels differ from the command-line keys.

  --selection-metric (optional, changes one VALUE in configs/default.yaml)
             classical_opt.selection_metric: test_f1 -> val_f1, with a comment
             that matches the paper (the code selects by validation F1).
             Refused if any Python file other than config.py reads the key.

Safety
------
* Python files: the syntax tree with docstrings removed must be identical
  before and after, so only comments and docstrings can change.
* YAML files: the parsed content must be identical (except the one key that
  --selection-metric changes, which is checked explicitly).
* Notebooks: only markdown cells and comment lines in code cells are edited,
  and a notebook is written only if the file round-trips byte for byte.
* Files under release/, reference_*/ and the deposition scripts are never
  touched. ``apply`` refuses files with uncommitted changes.

Usage
-----
    python scripts/fix_stale_docs.py scan  --root .            # show what would change
    python scripts/fix_stale_docs.py apply --root .            # write the changes
    python scripts/fix_stale_docs.py apply --root . --selection-metric
    python scripts/fix_stale_docs.py --self-test

Exit status: 0 success, 1 verification failure, 2 usage error,
3 refused (uncommitted changes or a reader of selection_metric exists).
Standard library only, plus PyYAML for configuration files.
"""
from __future__ import annotations

import argparse
import ast
import difflib
import io
import json
import re
import subprocess
import sys
import tempfile
import tokenize
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

EXTENSIONS = {".py", ".md", ".yaml", ".yml", ".ipynb", ".rst", ".txt"}
SKIP_DIRS = {".git", "release", "reference_data", "reference_figures", "reference_configs",
             "__pycache__", ".ipynb_checkpoints", "outputs", "runs", "logs", "node_modules",
             ".venv", "venv", "build", "dist"}
SKIP_FILES = {"generate_file_lists.py", "verify_deposition_tree.py", "trace_roi.py",
              "verify_bia_upload.py", "fill_release_values.sh", "bia_ftp_upload.sh",
              "rename_gold_standard.py", "fix_stale_docs.py", "CHANGELOG.md"}
SKIP_NAME_PARTS = ("deposition", "filelist_", "bioimage", "bia_")
G = r"G[TS]"  # GT before the terminology change, GS after


@dataclass
class Rule:
    name: str
    pattern: re.Pattern
    repl: str
    applies: Callable[[Path], bool]


def _any(_p: Path) -> bool:
    return True


def _named(*names: str) -> Callable[[Path], bool]:
    return lambda p: p.name in names


RULES: List[Rule] = [
    Rule("diamonds", re.compile(
        r"Uses an L1-norm \(Manhattan distance\) criterion so the rendered shape\n"
        r"(?P<i1>[ \t]*)matches the 7×7 diamond used for " + G + r" annotation \(point_disk_radius=3\n"
        r"(?P<i2>[ \t]*)in " + G + r"; use point_disk_radius=2 here for a 5×5 diamond\)\.\n"
        r"\n"
        r"(?P<i3>[ \t]*)A diamond of radius r has tip-to-tip span \(2r\+1\) in both axes:\n"
        r"(?P<i4>[ \t]*)r=1 → 3×3 diamond  \(13 pixels\)  — very small\n"
        r"(?P=i4)r=2 → 5×5 diamond  \(13 pixels\)  — matches " + G + r" 7×7 style at half size\n"
        r"(?P=i4)r=3 → 7×7 diamond  \(25 pixels\)  — identical to " + G + r" diamond shape\n"),
        "Uses an L1-norm (Manhattan distance) criterion, the shape used to render\n"
        "\\g<i1>the gold-standard points: a 5×5 diamond (13 pixels) for the published\n"
        "\\g<i2>point_disk_radius=2.\n"
        "\n"
        "\\g<i3>A diamond of radius r has tip-to-tip span (2r+1) in both axes:\n"
        "\\g<i4>r=1 → 3×3 diamond  (5 pixels)\n"
        "\\g<i4>r=2 → 5×5 diamond  (13 pixels)  — published setting, gold-standard footprint\n"
        "\\g<i4>r=3 → 7×7 diamond  (25 pixels)\n",
        _named("postprocess.py")),
    Rule("diamonds", re.compile(r"\((7 × 7|7×7) diamonds at(\s+)each annotated centroid\)"),
         r"(5 × 5 diamonds, 13 px, at\2each annotated point)", _any),
    Rule("diamonds", re.compile(r"(?<!r=3 → )\b7(\s?)([×x])(\s?)7([- ])diamond"),
         r"5\1\2\g<3>5\4diamond", _any),
    Rule("count", re.compile(r"((?:Ground-truth|Gold-standard) ecDNA count) \(number of annotated centroids\)\."),
         r"\1: 8-connected components of at least 3 px in the rendered mask (at most the"
         r" number of annotated points).", _named("samples.py")),
    Rule("checkpoint", re.compile(r"Best checkpoint: epoch 58, val_loss 0\.6415"),
         "Best checkpoint of the released model: epoch 49, val_loss 0.6421", _named("train.py")),
    Rule("disks", re.compile(r"(# For peaks mask, scale COORDINATES then re-render) disks at"),
         r"\1 diamonds at", _named("run_eccount.py")),
    Rule("disks", re.compile(r'# Fixed-radius disks rendered at peak coordinates for the "peaks"'),
         '# Fixed-radius diamonds (5 x 5, 13 px, for radius 2) drawn at peak coordinates for the "peaks"',
         _named("default.yaml")),
    Rule("config", re.compile(r"(#.*)\(n = 2,984\)"), r"\1(n = 2,986)", _named("default.yaml")),
    Rule("config", re.compile(r"(#.*)paths_example\.yaml"), r"\1paths.example.yaml", _any),
    Rule("config", re.compile(r"# eccount — novel deep-learning model"),
         "# eccount — probabilistic-localization model", _named("default.yaml")),
    Rule("config", re.compile(
        r"(?m)^(?P<ind>[ \t]*)# Models in the main benchmark \(Figure 6\)\. Order matches paper figure\.\n"
        r"(?!(?P=ind)# These labels are descriptive)"),
        "\\g<ind># Models in the main benchmark (Figure 6). Order matches paper figure.\n"
        "\\g<ind># These labels are descriptive. The command-line tools use the registry keys\n"
        "\\g<ind># classical, classical_before_opt, label_engine, ecseg, mia, eccount_mask and\n"
        "\\g<ind># eccount_peaks (for example --models eccount_peaks eccount_mask).\n",
        _named("default.yaml")),
]

SELECTION_RULE = Rule("selection-metric", re.compile(
    r"(?m)^(?P<ind>[ \t]*)# Final selection policy: highest F1 on the held-out test split\.\n"
    r"(?P=ind)selection_metric: test_f1\b"),
    "\\g<ind># Final selection policy: highest F1 on the validation split. The test F1 of\n"
    "\\g<ind># the selected configuration is recorded for reporting only.\n"
    "\\g<ind>selection_metric: val_f1", _named("default.yaml"))


# ----------------------------------------------------------------------------
# Verification
# ----------------------------------------------------------------------------

class _StripDocstrings(ast.NodeTransformer):
    def _strip(self, node):
        self.generic_visit(node)
        body = getattr(node, "body", None)
        if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            body[0] = ast.Expr(value=ast.Constant(value="<doc>"))
        return node

    visit_Module = visit_FunctionDef = visit_AsyncFunctionDef = visit_ClassDef = _strip


def python_equivalent(old: str, new: str) -> bool:
    try:
        a = _StripDocstrings().visit(ast.parse(old))
        b = _StripDocstrings().visit(ast.parse(new))
    except SyntaxError:
        return False
    return ast.dump(a) == ast.dump(b)


def yaml_equivalent(old: str, new: str, allow_selection_change: bool) -> Tuple[bool, str]:
    try:
        import yaml
    except ImportError:
        return False, "PyYAML is not installed"
    try:
        a, b = yaml.safe_load(old), yaml.safe_load(new)
    except yaml.YAMLError as exc:
        return False, f"YAML parse error: {exc}"
    if a == b:
        return True, ""
    if allow_selection_change and isinstance(a, dict) and isinstance(b, dict):
        try:
            if a["classical_opt"]["selection_metric"] == "test_f1" and \
                    b["classical_opt"]["selection_metric"] == "val_f1":
                a["classical_opt"]["selection_metric"] = "val_f1"
                if a == b:
                    return True, ""
        except (KeyError, TypeError):
            pass
    return False, "parsed content changed"


# ----------------------------------------------------------------------------
# Editing
# ----------------------------------------------------------------------------

def _apply_rules(text: str, path: Path, rules: List[Rule], comment_only: bool) -> Tuple[str, List[str]]:
    hits: List[str] = []
    if comment_only:
        lines = text.splitlines(keepends=True)
        out = []
        for line in lines:
            if line.lstrip().startswith("#"):
                new = line
                for r in rules:
                    if r.applies(path):
                        new, n = r.pattern.subn(r.repl, new)
                        hits += [r.name] * n
                out.append(new)
            else:
                out.append(line)
        return "".join(out), hits
    for r in rules:
        if r.applies(path):
            text, n = r.pattern.subn(r.repl, text)
            hits += [r.name] * n
    return text, hits


def edit_notebook(raw: str, path: Path, rules: List[Rule]) -> Tuple[Optional[str], List[str], str]:
    try:
        nb = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, [], f"not valid JSON ({exc})"
    indent = 1
    for candidate in (1, 2, 4):
        if json.dumps(nb, indent=candidate, ensure_ascii=False) + "\n" == raw:
            indent = candidate
            break
    else:
        return None, [], "does not round-trip; edit by hand if the scan lists changes"
    hits: List[str] = []
    for cell in nb.get("cells", []):
        src = cell.get("source", "")
        joined = "".join(src) if isinstance(src, list) else src
        new, h = _apply_rules(joined, path, rules, comment_only=cell.get("cell_type") == "code")
        if h:
            hits += h
            cell["source"] = new.splitlines(keepends=True) if isinstance(src, list) else new
    return json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", hits, ""


def _python_spans(src: str) -> List[Tuple[int, int]]:
    """Character spans of docstrings and comments in Python source."""
    lines = src.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def at(lineno: int, col: int, byte_col: bool) -> int:
        if byte_col:
            col = len(lines[lineno - 1].encode("utf-8")[:col].decode("utf-8", errors="ignore"))
        return starts[lineno - 1] + col

    spans: List[Tuple[int, int]] = []
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body:
            first = body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) \
                    and isinstance(first.value.value, str):
                v = first.value
                spans.append((at(v.lineno, v.col_offset, True), at(v.end_lineno, v.end_col_offset, True)))
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            spans.append((at(*tok.start, False), at(*tok.end, False)))
    return sorted(spans)


def _edit_python(raw: str, path: Path, rules: List[Rule]) -> Tuple[str, List[str]]:
    try:
        spans = _python_spans(raw)
    except (SyntaxError, tokenize.TokenError, ValueError):
        return raw, []
    out, hits, pos = [], [], 0
    for a, b in spans:
        if a < pos:
            continue
        out.append(raw[pos:a])
        piece, h = _apply_rules(raw[a:b], path, rules, comment_only=False)
        out.append(piece)
        hits += h
        pos = b
    out.append(raw[pos:])
    return "".join(out), hits


def process_file(path: Path, rules: List[Rule], allow_selection: bool) -> Tuple[Optional[str], List[str], str]:
    raw = path.read_text(encoding="utf-8")
    if path.suffix == ".ipynb":
        new, hits, err = edit_notebook(raw, path, rules)
        if err or not hits:
            return None, hits, err
        return new, hits, ""
    if path.suffix == ".py":
        new, hits = _edit_python(raw, path, rules)
    else:
        new, hits = _apply_rules(raw, path, rules, comment_only=False)
    if not hits:
        return None, [], ""
    if path.suffix == ".py" and not python_equivalent(raw, new):
        return None, hits, "REFUSED: the change would alter code, not only comments or docstrings"
    if path.suffix in (".yaml", ".yml"):
        ok, why = yaml_equivalent(raw, new, allow_selection)
        if not ok:
            return None, hits, f"REFUSED: {why}"
    return new, hits, ""


def iter_files(root: Path):
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix not in EXTENSIONS:
            continue
        rel = p.relative_to(root)
        if any(part in SKIP_DIRS or part.startswith("reference_") for part in rel.parts[:-1]):
            continue
        if p.name in SKIP_FILES or any(s in p.name.lower() for s in SKIP_NAME_PARTS):
            continue
        yield p


def selection_readers(root: Path) -> List[str]:
    out = []
    for p in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in p.relative_to(root).parts) or p.name in ("config.py", "fix_stale_docs.py"):
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for i, line in enumerate(text.splitlines(), 1):
            if "selection_metric" in line and not line.lstrip().startswith("#"):
                out.append(f"{p.relative_to(root)}:{i}: {line.strip()}")
    return out


def dirty_files(root: Path, files: List[Path]) -> List[str]:
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--"] + [str(f) for f in files],
                             capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    return [line[3:] for line in out.splitlines() if line.strip()]


def run(mode: str, root: Path, selection: bool, force: bool) -> int:
    rules = list(RULES)
    if selection:
        readers = selection_readers(root)
        if readers and not force:
            print("REFUSING --selection-metric: these lines read the key; check them first "
                  "(or pass --force if they only document it):")
            for r in readers:
                print("   ", r)
            return 3
        rules.append(SELECTION_RULE)
    changes, problems = [], []
    for p in iter_files(root):
        new, hits, err = process_file(p, rules, selection)
        if err:
            problems.append((p, err, hits))
        if new is not None:
            changes.append((p, new, hits))
    for p, new, hits in changes:
        old = p.read_text(encoding="utf-8")
        rel = p.relative_to(root)
        print(f"--- {rel}  ({', '.join(sorted(set(hits)))}; {len(hits)} edit(s))")
        for line in difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0):
            if line.startswith(("---", "+++")):
                continue
            print("   ", line)
    for p, err, hits in problems:
        print(f"!!! {p.relative_to(root)}: {err}")
    print(f"\nfiles to change: {len(changes)}; problems: {len(problems)}")
    if mode == "scan":
        return 1 if any(e.startswith("REFUSED") for _, e, _ in problems) else 0
    dirty = dirty_files(root, [p for p, _, _ in changes])
    if dirty:
        print("REFUSING to write: uncommitted changes in", ", ".join(dirty))
        return 3
    for p, new, _ in changes:
        p.write_text(new, encoding="utf-8")
    print(f"written: {len(changes)} file(s). Review with 'git diff', then run the tests.")
    return 1 if any(e.startswith("REFUSED") for _, e, _ in problems) else 0


# ----------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------

SAMPLE_POST = '''def peaks_to_mask(peaks, shape, disk_radius=2):
    """Render peaks as filled diamonds on a binary mask.

    Uses an L1-norm (Manhattan distance) criterion so the rendered shape
    matches the 7×7 diamond used for GS annotation (point_disk_radius=3
    in GS; use point_disk_radius=2 here for a 5×5 diamond).

    A diamond of radius r has tip-to-tip span (2r+1) in both axes:
        r=1 → 3×3 diamond  (13 pixels)  — very small
        r=2 → 5×5 diamond  (13 pixels)  — matches GS 7×7 style at half size
        r=3 → 7×7 diamond  (25 pixels)  — identical to GS diamond shape
    """
    # For peaks mask, scale COORDINATES then re-render disks at
    return "7×7 diamond"   # a string literal that must not change
'''
SAMPLE_META = '''PREFERRED = (
    # rendered 7×7-diamond GS mask under 8-connectivity
    # annotation points are close enough that their 7×7 diamonds merge into
    "ecDNA_gt",
)
'''
SAMPLE_SAMPLES = '''class Sample:
    """
    gt_mask_path
        Path to the rendered gold-standard binary mask (7 × 7 diamonds at
        each annotated centroid).
    ecdna_gt
        Gold-standard ecDNA count (number of annotated centroids).
    """
'''
SAMPLE_TRAIN = '''"""
* Best checkpoint: epoch 58, val_loss 0.6415
"""
EPOCH = 58
'''
SAMPLE_YAML = '''# users copy `configs/paths_example.yaml` to `configs/paths.local.yaml`
dataset:
  # Five cell lines are present in the full resource (n = 2,984). Four are in
  cell_lines_full: [A, B]
classical_opt:
  # Final selection policy: highest F1 on the held-out test split.
  selection_metric: test_f1
# eccount — novel deep-learning model (paper §5, Supp §10).
eccount:
  postprocess:
    # Fixed-radius disks rendered at peak coordinates for the "peaks"
    point_disk_radius: 2
benchmark:
  # Models in the main benchmark (Figure 6). Order matches paper figure.
  models: [classic_post_opt, eccount_threshold]
'''


def self_test() -> int:
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src/pkg").mkdir(parents=True)
        (root / "configs").mkdir()
        (root / "release").mkdir()
        files = {
            "src/pkg/postprocess.py": SAMPLE_POST, "src/pkg/metadata.py": SAMPLE_META,
            "src/pkg/samples.py": SAMPLE_SAMPLES, "src/pkg/train.py": SAMPLE_TRAIN,
            "configs/default.yaml": SAMPLE_YAML, "release/notes.md": "7×7 diamond\n",
        }
        for rel, text in files.items():
            (root / rel).write_text(text, encoding="utf-8")
        (root / "src/pkg/run_eccount.py").write_text(
            "x = 1\n# For peaks mask, scale COORDINATES then re-render disks at\n", encoding="utf-8")
        nb = {"cells": [{"cell_type": "markdown", "metadata": {}, "source": ["Each point is a 7×7 diamond.\n"]},
                        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                         "source": ["# a 7 × 7 diamond\n", "label = '7 × 7 diamond'\n"]}],
              "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        (root / "tutorial.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = run("apply", root, selection=True, force=False)
        post = (root / "src/pkg/postprocess.py").read_text()
        checks = [
            ("exit code 0", code == 0),
            ("postprocess docstring rewritten", "(5 pixels)" in post and "published setting" in post),
            ("r=3 line kept", "r=3 → 7×7 diamond  (25 pixels)" in post),
            ("string literal untouched", 'return "7×7 diamond"' in post),
            ("run_eccount comment", "re-render diamonds at" in (root / "src/pkg/run_eccount.py").read_text()),
            ("metadata comments", "5×5-diamond GS mask" in (root / "src/pkg/metadata.py").read_text()
             and "5×5 diamonds merge" in (root / "src/pkg/metadata.py").read_text()),
            ("samples docstring", "5 × 5 diamonds, 13 px, at" in (root / "src/pkg/samples.py").read_text()
             and "at most the number of annotated points" in (root / "src/pkg/samples.py").read_text()),
            ("train docstring", "epoch 49, val_loss 0.6421" in (root / "src/pkg/train.py").read_text()
             and "EPOCH = 58" in (root / "src/pkg/train.py").read_text()),
            ("yaml fixes", all(s in (root / "configs/default.yaml").read_text() for s in (
                "paths.example.yaml", "(n = 2,986)", "selection_metric: val_f1",
                "probabilistic-localization model", "Fixed-radius diamonds", "registry keys"))),
            ("release untouched", (root / "release/notes.md").read_text() == "7×7 diamond\n"),
            ("notebook markdown and comment edited, code string kept",
             "5×5 diamond" in (root / "tutorial.ipynb").read_text()
             and "# a 5 × 5 diamond" in (root / "tutorial.ipynb").read_text()
             and "label = '7 × 7 diamond'" in (root / "tutorial.ipynb").read_text()),
        ]
        with contextlib.redirect_stdout(io.StringIO()):
            again = run("scan", root, selection=True, force=False)
        checks.append(("idempotent", "files to change: 0" in _capture(run, "scan", root, True, False) and again == 0))
        bad = root / "src/pkg/bad.py"
        bad.write_text("# 7×7 diamond\nx = 1\n", encoding="utf-8")
        checks.append(("comment-only python edit accepted",
                       process_file(bad, RULES, False)[0] == "# 5×5 diamond\nx = 1\n"))
        (root / "src/pkg/reader.py").write_text("m = cfg['classical_opt']['selection_metric']\n", encoding="utf-8")
        checks.append(("selection reader blocks the value change",
                       _capture_code(run, "scan", root, True, False) == 3))
    for name, passed in checks:
        print(f"  {'ok  ' if passed else 'FAIL'} {name}")
        ok &= passed
    if not ok:
        print(buf.getvalue())
    print("SELF-TEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def _capture(fn, *args) -> str:
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(*args)
    return buf.getvalue()


def _capture_code(fn, *args) -> int:
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args)


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", nargs="?", choices=["scan", "apply"])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--selection-metric", action="store_true",
                    help="also change classical_opt.selection_metric to val_f1 in configs/default.yaml")
    ap.add_argument("--force", action="store_true", help="ignore readers of selection_metric")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args(argv)
    if args.self_test:
        return self_test()
    if not args.mode:
        ap.print_usage()
        return 2
    root = args.root.resolve()
    if not (root / "configs").is_dir():
        print(f"{root} does not look like the repository root (no configs/)", file=sys.stderr)
        return 2
    return run(args.mode, root, args.selection_metric, args.force)


if __name__ == "__main__":
    sys.exit(main())
