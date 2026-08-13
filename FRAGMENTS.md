# Small edits — apply these three by hand

## 1. `Makefile` — line in the help block

REPLACE:
	@echo "  test               run pytest (112 tests)"
WITH:
	@echo "  test               run pytest (126 tests)"

## 2. `LONGLEAF_INSTALL.md` — section 6

REPLACE:
```
Expected (last line):

```
112 passed in ~4s
```
```
WITH:
```
Expected (last line):

```
124 passed, 2 skipped in ~3s
```

The two skips are I/O tests that need sample images from the released dataset;
they are expected on a code-only clone.
```

ALSO REPLACE (section 8, troubleshooting heading):
### `pytest` reports fewer than 112 passed
WITH:
### `pytest` reports fewer than 124 passed

## 3. `pyproject.toml` — two edits

(a) REPLACE:
```
# FILL IN: add the real authors below (the lab entry can stay as a fallback).
authors = [
    { name = "<<FILL: First Last>>", email = "<<FILL: email>>" },
    { name = "Brunk Lab", email = "brunk@unc.edu" },
]
```
WITH:
```
authors = [
    { name = "Poorya Behnamie" },
    { name = "Elizabeth Brunk", email = "brunk@unc.edu" },
]
```

(b) REPLACE the commented Dataset URL under [project.urls]:
```
# Dataset = "https://www.ebi.ac.uk/biostudies/studies/<<FILL: S-BIAD#####>>"
```
WITH:
```
Dataset = "https://www.ebi.ac.uk/biostudies/studies/{{ACCESSION}}"
```

(c) DELETE this block entirely — see note in chat:
```
[tool.setuptools.package-data]
# Ship the frozen classical-pipeline parameter JSON inside the installed package
# so users can import + run the post-opt classical pipeline without checking
# out the source tree.
"ecdna_bench" = [
    "../configs/classical/stage3_frozen_params.json",
]
```

## 4. `src/ecdna_bench/cli/__init__.py` — delete line 21

DELETE:
make_figures        Generate paper figures from frozen CSVs.

(Check the surrounding docstring lists nine commands afterwards, not ten.)

## 5. `docs/REPRODUCTION.md` — line 4 (optional)

The phrase "There is no `make_figures` CLI" is correct prose but keeps tripping
the marker sweep. Optionally reword to:

    generates it and lists the output files produced. Figures are not produced
    by any CLI command; all five notebooks under `notebooks/` generate them.
