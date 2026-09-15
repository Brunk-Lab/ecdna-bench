# ecdna-bench on Longleaf: a complete guide for Brunk Lab members

### From never having used a computing cluster, to reproducing every number in the paper

---

> **What changed in this version (15 September 2026)**
>
> 1. **Setup no longer changes your other conda environments.** The first
>    version of the setup (script or manual steps) added lines to `~/.bashrc`
>    that moved the conda and pip package caches and defined general names such
>    as `$DATA` and `$REPO` for every terminal. Those lines affected every
>    environment you use, not only this project. The new setup writes all
>    settings into one file in your project folder and applies them only when you
>    type `ecdna`; `ecdna_off` removes them again.
>    **If you set up before 15 September, run once:**
>    ```bash
>    bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh --migrate
>    ```
>    It removes the old lines (a backup of `~/.bashrc` is kept), keeps your
>    clone and your results, and checks everything. Then open a new terminal.
> 2. **"Ground truth" is now "gold standard"**, as in the paper. File and folder
>    names that are part of the BioImage Archive record keep `gt`
>    (`gt_image/`, `gt_count`, ...); they mean the same annotation.
> 3. **The leave-one-cell-line-out recipe (Part 6.1) is replaced.** The old
>    recipe put its settings file in `configs/`, where `paths.local.yaml` silently
>    overrides it, so the job trained on the original split. Use
>    `scripts/train_eccount_loco.py` and `slurm/submit_eccount_loco.sh`.
> 4. **The model key for the threshold output is `eccount_mask`**, not
>    `eccount_threshold`. With the wrong key the benchmark scores nothing.
> 5. The repository is `https://github.com/Brunk-Lab/ecdna-bench`, and the image
>    data are BioImage Archive accession **S-BIAD4097**.

---

**What this is.** A hands-on tutorial for the ecdna-bench project: a benchmark
of six computational methods for detecting and counting extrachromosomal DNA
(ecDNA) in fluorescence microscopy images, and the resource of images and
annotations behind it.

**Who it is for.** Lab members with a Longleaf account. No experience with
Linux, clusters, Python or image analysis is assumed. Users outside UNC follow
[`TUTORIAL_EXTERNAL.md`](TUTORIAL_EXTERNAL.md) instead.

**How to use it.** Read Parts 1 and 2 once. Then run the setup script in
Part 3. **You will only ever type one thing yourself: your ONYEN**, and the
script suggests it.

**How long.** About fifteen minutes for the setup. After that, any part of the
analysis starts with one word.

---

## Contents

**Part 1: The tools** *(read once)*
1.1 The terminal · 1.2 Commands · 1.3 Files and paths · 1.4 What a cluster is ·
1.5 SLURM · 1.6 Environments · 1.7 Git · 1.8 Notebooks

**Part 2: The science** *(read once)*
2.1 ecDNA · 2.2 FISH imaging · 2.3 The dataset · 2.4 The six methods ·
2.5 The metrics · 2.6 The pipeline

**Part 3: Setup** *(one script)*

**Part 4: Running the analysis**

**Part 5: Notebooks and figures**

**Part 6: Worked examples** *(the two questions reviewers ask)*

**Part 7: When something goes wrong**

**Part 8: Reference**

---

# Part 1: The tools

Six ideas. Once these click, everything else is detail.

## 1.1 The terminal, the shell, and bash

When you connect to a computing cluster you do not get a desktop. You get a
window with text in it and a blinking cursor.

That window is a **terminal**. Behind it runs a program that reads what you
type and does it: the **shell**. The one almost everyone uses is **bash**.

- **Terminal** = the window
- **Bash** = the program inside it that listens to you
- **Command** = one instruction you type and then press Enter

Scientists use typed commands because a command can be saved, shared, repeated
exactly and put in a document like this one. A sequence of mouse clicks cannot.

**Bash forgets everything between sessions.** Close the terminal, open it
again, and it starts fresh. Part 3 uses this on purpose: project settings exist
only inside a session you start with `ecdna`, so nothing else you do on
Longleaf is affected.

## 1.2 What a command looks like

```
program    options    what to act on
```

For example `ls -l /proj`: `ls` lists a folder, `-l` asks for the long format
(sizes and dates), `/proj` is the folder.

| Command | Meaning | Example |
|---|---|---|
| `pwd` | Where am I? | `pwd` |
| `ls` | List what is here | `ls` |
| `cd` | Go somewhere | `cd /proj` |
| `mkdir` | Create a folder | `mkdir myfolder` |
| `cat` | Show a text file | `cat notes.txt` |
| `nano` | A simple text editor | `nano notes.txt` |
| `cp` | Copy a file | `cp a.txt b.txt` |
| `echo` | Print something | `echo hello` |
| `grep` | Search inside files | `grep error log.txt` |

Two conventions: the code blocks below contain no `$` prompt, so you can copy
whole blocks; and `#` starts a comment, which bash ignores.

**If you get stuck mid-command**, press `Ctrl-C` to get back to a clean prompt.

## 1.3 Files, folders and paths

Folders live inside folders, up to a single starting point called `/`
("root"). A **path** is an address:
`/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench` means: from root, into
`proj`, then `brunk_ecdna_cv_project`, then `Poorya`, then `ecdna-bench`.

- **Absolute** paths start with `/` and work from anywhere.
- **Relative** paths are read from the folder you are in.
- `.` is "here", `..` is "the folder above", `~` is "my home folder".

Files whose names start with a dot are hidden; `ls -a` shows them. The
configuration file `~/.bashrc` is one of them.

## 1.4 What a computing cluster is

Longleaf is several hundred computers sharing one filesystem.

**Login nodes** are where you land. They are for editing files, looking at
results and submitting work, **not for doing real work**.

**Compute nodes** are the large machines where real work runs. You ask for one,
and a scheduler gives you one when it is free.

## 1.5 SLURM: asking for a compute node

**SLURM** is the scheduler. You describe a job (processors, memory, time, what
to run) in a small file and hand it over with `sbatch`. The job runs whether or
not you stay logged in.

| Command | Meaning |
|---|---|
| `sbatch job.sh` | Submit this job; prints a job number |
| `squeue -u $USER` | What am I running or waiting for? |
| `scancel 12345678` | Cancel job 12345678 |
| `sacct -j 12345678` | Did job 12345678 finish, and how? |
| `seff 12345678` | How much memory and CPU did it use? |

**Partitions** are groups of nodes: `general` has ordinary processors;
`a100-gpu` and `l40-gpu` have the graphics cards needed to train networks.

## 1.6 Environments

The analysis uses about forty Python libraries, and their results can change
between versions. A **conda environment** is a folder with its own Python and
its own libraries at fixed versions. The project's recipe is
`env/environment.yml`; the lab's copy of that environment is already built at
`/proj/brunk_ecdna_cv_project/Poorya/envs/ecdna-bench` and is readable by the
whole group. **You never need to install anything into it**, and you should not
try (Part 7 explains why).

> **Two environments you may hear about.** `ecDNA_Hybrid` was used to draw the
> original figures; `env/environment.yml` was built from it, and the shared
> `ecdna-bench` environment reproduces its numbers (Part 8.3). Use only
> `ecdna-bench`.

## 1.7 Git: the undo button

**Git** records the state of a folder as **commits**. You can return to any of
them. **GitHub** keeps a copy online: <https://github.com/Brunk-Lab/ecdna-bench>.

| Command | Meaning |
|---|---|
| `git status` | What has changed since the last commit? |
| `git log --oneline -5` | The last five commits |
| `git restore <file>` | Undo my changes to this file |
| `git pull` | Fetch the latest version |

## 1.8 Jupyter notebooks and kernels

A **notebook** mixes text, code and results. A **kernel** is the engine that
runs the code, and the kernel you pick decides which environment (and which
library versions) the code uses. Part 3 registers the right kernel for you.

---

# Part 2: The science

## 2.1 What ecDNA is, and why counting it matters

In many cancers, fragments of DNA break away from the chromosomes and form
small circles in the nucleus: **extrachromosomal DNA (ecDNA)**. They often carry
the oncogenes that drive the cancer, they are shared unevenly when a cell
divides, and the resulting cell-to-cell variation helps tumors survive
treatment. The quantity that matters is **how many ecDNA copies each cell
has**, which means counting them, cell by cell, in images.

## 2.2 How the images are made

**FISH** (fluorescence in situ hybridization) attaches a fluorescent dye to a
short DNA probe that binds only the amplified sequence, so every copy lights
up as a dot. Cells are arrested in **metaphase** and burst onto a slide, giving
a **metaphase spread**: one cell's chromosomes and ecDNA, laid flat.

Each image set in the resource has:

- an **RGB** image of the probe channel: chromosomes appear as large shapes,
  ecDNA as small dots;
- a **DAPI** image, a stain for all DNA, for context.

Images are 2,448 × 2,048 pixels, acquired at ×60. A single image holds from none
to more than 1,600 ecDNA. The dots are small, vary in brightness, overlap and
share the field with debris and neighboring cells, which is why counting them
automatically is hard and why methods must be tested against careful manual
annotation.

## 2.3 What is in the dataset

**The full resource: 2,986 image sets from five cancer cell lines**:
NCI-H2170, SUM159PT, SNU16, COLO320DM and NCI-H716. Every image set carries a
manual annotation of every ecDNA signal: the **gold standard**.

**The benchmark: 1,145 of those image sets.** These also carry a hand-drawn
**ROI mask** (region of interest), an outline of the one metaphase spread to
analyze. SUM159PT has no ROI masks, so it is not in the benchmark.

**The partitions: 800 training / 170 validation / 175 test.**

- **Training**: what a learning method may learn from.
- **Validation**: used during development to choose settings and checkpoints.
- **Test**: held out and used once, at the end; the honest estimate of
  performance on new data.

**How the gold standard is stored.** Annotators marked each ecDNA with a point.
Each point is drawn as a small diamond (5 × 5 pixels, 13 pixels in area) to give
the **gold-standard mask**, and the objects the benchmark counts are the
8-connected components of that mask with an area of at least 3 pixels. Points
closer than a few pixels merge into one object, so an image's object count is at
most its point count.

Everything is deposited in the BioImage Archive under **S-BIAD4097**. The
archive record was written before the terminology change and calls the
annotation "ground truth"; its folders are `images/gt_image/` (masks),
`images/gt_coords/` (points) and `images/gt_npz/` (sparse masks).

## 2.4 The six methods being compared

| Method | What it is |
|---|---|
| **Classic (after opt)** | A rule-based image-processing pipeline (contrast enhancement, thresholding, connected components, a color rule) whose parameters were tuned per cell line by Bayesian optimization |
| **Label Engine** | A U-Net segmentation network developed in the lab (<https://github.com/Brunk-Lab/Label-Engine>), trained on the benchmark training partition and used in its automatic mode |
| **ecSeg** | A published deep-learning tool for ecDNA, applied with its released weights, without retraining |
| **MIA** | A published image-analysis model, evaluated from the archived predictions of the original study |
| **ecCount (threshold mask)** | The model developed in this project, read out by thresholding its probability map |
| **ecCount (peaks)** | The same model, read out by finding local peaks: the headline method |

The last two are two readouts of one network. Its output is a probability map;
keeping everything above a cut-off gives the threshold mask, and finding the
local high points gives the peaks. Peaks separate touching dots better.

`Classic (before opt)`, the classical pipeline with default parameters, also
appears in the results, as the starting point of the optimization.

## 2.5 How performance is measured

**Object-level F1: were the right dots found?** Every predicted object is
paired with at most one gold-standard object that is close enough (centroid
distance at most 20 pixels) or overlaps enough (IoU at least 0.1). Then
precision is the fraction of reported objects that are real, recall the fraction
of real objects that were found, and F1 combines the two. True positives, false
positives and false negatives are **summed over all images first** and F1 is
computed once (pooled).

**Count MAE: how far off is the number?** The mean, over images, of the absolute
difference between the predicted and the gold-standard count.

**Signed bias: does it lean one way?** The same differences, keeping the sign.
Negative means undercounting.

**Why bias matters for biology.** A method with a bias of −40 does not simply
report smaller numbers; it loses more objects the more crowded an image is,
which distorts comparisons between conditions.

**The results**, all 1,145 benchmark images, OR matching:

| Method | Object F1 | Count MAE | Signed bias |
|---|---|---|---|
| ecCount (peaks) | **0.942** | **13.1** | **+0.4** |
| ecCount (threshold mask) | 0.917 | 18.1 | −11.4 |
| Label Engine | 0.825 | 36.0 | −28.8 |
| MIA | 0.800 | 47.0 | −40.6 |
| Classic (after opt) | 0.777 | 44.9 | −24.3 |
| ecSeg | 0.464 | 121.5 | −118.5 |

> **OR matching.** A predicted and a gold-standard object may be paired if they
> are close **or** overlap (OR); the stricter AND requires both. Every published
> number uses OR, from the `or_matching/` folder. Results with AND sit in a
> sibling folder, `and_matching/`, **with identical file names**. Always check
> which folder you read.

## 2.6 The pipeline

```
   raw images
        │
   build_metadata      catalog every image and its files
        │
      run_qc           check every file opens and is consistent
        │
        ├──────────────────┬───────────────────────┐
        ▼                  ▼                       ▼
 optimize_classical   train_eccount           run_baseline
   (tune settings)    (train the network)     (convert ecSeg / MIA /
        │                  │                   Label Engine outputs)
  run_classical       run_eccount                  │
        └──────────────────┴────────────┬──────────┘
                                        ▼
                                   benchmark        score everything
                                        │
                                   notebooks        draw the figures
```

Each stage reads files and writes files, so any stage can run on its own once
its inputs exist. The ROI model, which predicts metaphase outlines for images
without a manual mask, is a separate step (Part 6.2).

---

# Part 3: Setup

## 3.1 Before you start

You need a Longleaf account and membership of the `brunk_ecdna_cv_project`
group; request both at <https://help.rc.unc.edu>.

**Log in.** On macOS or Linux open Terminal; on Windows open PowerShell. Then,
with your UNC username in place of `YOUR_ONYEN`:

```bash
ssh YOUR_ONYEN@longleaf.unc.edu
```

Type your password (nothing appears as you type), approve the Duo prompt, and
you are on a login node: the prompt looks like `[yourname@longleaf-login2 ~]$`.

## 3.2 Run the setup script

```bash
bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh
```

It asks for your ONYEN and suggests your login name; press Enter to accept.
Everything else is automatic, and it is safe to run again at any time.

If you used the old setup before 15 September, add `--migrate` (see the box at
the top).

When it finishes you see `Setup complete.` and a short summary. **Open a new
terminal** (or log out and in again). From then on, every working session
starts with

```bash
ecdna
```

and ends, if you want to go back to a plain terminal, with

```bash
ecdna_off
```

Closing the terminal also ends the session.

## 3.3 What the script did

| Step | What | Where |
|---|---|---|
| 1 | Asked for your ONYEN | |
| 2 | Checked that the lab storage, the shared environment, the image data and the shared repository are reachable | `/proj/brunk_ecdna_cv_project/Poorya/...` |
| 3 | Looked for settings left by the old setup, and removed them if you agreed (backup kept as `~/.bashrc.backup-DATE`) | `~/.bashrc` |
| 4 | Created your workspace | `/proj/brunk_ecdna_cv_project/<ONYEN>/{repos,runs,logs}` |
| 5 | Made your own copy of the code, with its full history | `<your folder>/repos/ecdna-bench` |
| 6 | Wrote your local paths file: inputs from the shared data, outputs to your folder | `<your copy>/configs/paths.local.yaml` |
| 7 | Wrote your session file, and added the one-line `ecdna` shortcut | `<your folder>/ecdna-bench.env`, `~/.bashrc` |
| 8 | Registered the Jupyter kernel `ecdna-bench (canonical)`, with its settings stored inside the kernel | `~/.local/share/jupyter/kernels/ecdna-bench` |
| 9 | Checked everything in a clean shell, and ran the test suite | log in `<your folder>/logs/` |

**What it did not do.** It did not set anything globally: no cache locations,
no general variable names, no changes to other kernels, and it installed nothing
into the shared environment or into `~/.local`.

**What `ecdna` does.** It reads your session file, which:

- switches on the shared environment (`module load anaconda`, `conda activate`);
- makes Python import the code from **your** copy (`PYTHONPATH`);
- makes Python ignore packages in `~/.local` (`PYTHONNOUSERSITE=1`), so that
  nothing installed there can change the results;
- sets `ECDNA_PYTHON` and `ECDNA_PROJECT_ROOT`, which the cluster job scripts
  read, plus shortcuts to your folders: `$ECDNA_MYDIR`, `$ECDNA_REPO`,
  `$ECDNA_DATA`, `$ECDNA_SOURCE_REPO`;
- moves you into your copy of the repository.

`ecdna_off` undoes all of it. Every name starts with `ECDNA_`, so nothing
collides with other projects.

## 3.4 The checks, and what they mean

The script's last section prints one line per check:

| Line | Meaning |
|---|---|
| `session uses the shared environment` | `python` inside a session is the shared one |
| `~/.local packages are ignored in the session` | nothing in `~/.local` can shadow the environment |
| `code is imported from your copy` | your edits, not the shared repository, are what runs |
| `ecCount builds with 7,849,601 parameters` | the network architecture matches the paper |
| `every output path is inside <your folder>` | nothing you run writes into someone else's folder |
| `benchmark table readable` and `image files ... are readable` | the benchmark can read its inputs |
| `Jupyter kernel carries its own settings` | notebooks run like a session |
| `test suite: ... passed` | the code's internal logic works (no data needed) |

To run only the checks later:

```bash
bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh --check
```

Other options: `--no-bashrc` (do not add the shortcut; start sessions with
`source <your folder>/ecdna-bench.env`), `--uninstall` (remove the shortcut and
the kernel; your folder is kept), `--yes` (no questions).

## 3.5 Where things live

| Area | Path | Use it for |
|---|---|---|
| Home | `/nas/longleaf/home/<ONYEN>` (also `/users/<o>/<n>/<ONYEN>`) | Settings and small files only (about 50 GB) |
| Scratch | `/work/users/<o>/<n>/<ONYEN>` | Temporary files. **Deleted automatically.** |
| Lab project | `/proj/brunk_ecdna_cv_project` | Everything real: data, environments, results |
| Your folder | `/proj/brunk_ecdna_cv_project/<ONYEN>` | Your copy of the code and all your outputs |
| Shared, read only | `/proj/brunk_ecdna_cv_project/Poorya/` | The shared environment, the data and the reference copy of the repository |

**The golden rule: read from the shared folders, write only into your own.**
Three layers make mistakes recoverable: git (every committed file can be
restored), separate copies (your settings affect only your copy), and write
protection on the published results.

## 3.6 Your copy and GitHub

Your copy was cloned from the shared repository, so `git pull` fetches updates
from it. To follow GitHub instead:

```bash
cd $ECDNA_REPO
git remote set-url origin https://github.com/Brunk-Lab/ecdna-bench.git
git pull
```

---

# Part 4: Running the analysis

## 4.1 Two ways to run

**Directly on the login node**: only for commands that take seconds and little
memory. **As a cluster job**: for everything else. The `slurm/` folder has a job
file for every stage.

## 4.2 How the job scripts work

1. **Start a session first** (`ecdna`). `sbatch` passes the session's
   `ECDNA_PYTHON` and `ECDNA_PROJECT_ROOT` to the job; without them a job uses the
   wrong Python or the wrong folder.
2. **Submit from the repository root**, never from inside `slurm/`. `ecdna`
   already puts you there; `pwd` must show your own copy.
3. **`logs/` must exist.** The setup script created it; if you deleted it,
   `mkdir -p logs`.

```bash
ecdna
pwd                    # /proj/brunk_ecdna_cv_project/<ONYEN>/repos/ecdna-bench
sbatch slurm/submit_benchmark.sh
```

## 4.3 Where the output goes

Your `configs/paths.local.yaml` sends results, logs and training output to
`/proj/brunk_ecdna_cv_project/<ONYEN>/runs/`. Some job scripts also write inside
the project root; because the project root is your copy, those files land in
your copy too. If a job rewrites a committed file in your copy (for example under
`release/frozen_results/`), `git status` shows it and `git restore <path>` puts
the published version back.

When running a command by hand, add `--output-dir` with a folder of your own.

## 4.4 The job scripts

| Script | Runs on | Roughly | What it does |
|---|---|---|---|
| `submit_eccount_train.sh` | `a100-gpu,l40-gpu` | about 1.5 h of training (70 epochs × ~80 s); the job requests more | Train the network from scratch |
| `submit_eccount_infer.sh` | `a100-gpu,l40-gpu` | 1–4 h | Run the trained network on images |
| `submit_eccount_loco.sh` | `a100-gpu,l40-gpu` | ~1 h 40 m per task (~35 m for NCI-H2170) | Leave-one-cell-line-out: train, predict, score (Part 6.1) |
| `submit_benchmark.sh` | `general` | 1–4 h | Score every method against the gold standard |
| `submit_sensitivity.sh` | `general` | 2–8 h | Sweep the matching settings |
| `submit_optimize_classical.sh` | `general` | ~6 h | Tune the classical pipeline, one task per cell line |
| `submit_classical_default.sh` | `general` | ~1 h | Classical pipeline, default settings |
| `submit_classical_optimized.sh` | `general` | ~1 h | Classical pipeline, tuned settings |
| `submit_benchmark_classical_before_after.sh` | `general` | ~1 h | Classical before and after tuning |

`slurm/README.md` has the same table with more detail.

## 4.5 Your first job

Scoring is the safest first job: it reads predictions that already exist and
computes every metric in the paper.

```bash
ecdna
sbatch slurm/submit_benchmark.sh
squeue -u $USER
```

SLURM prints a job number, for example `Submitted batch job 51234567`.

| `ST` | Meaning |
|---|---|
| `PD` | Waiting for a node |
| `R` | Running |
| *(not listed)* | Finished |

Follow the output with `tail -f logs/benchmark_51234567.out` (`Ctrl-C` stops
watching; the job continues). When it has finished:

```bash
sacct -j 51234567 --format=JobID,JobName,State,Elapsed,ExitCode
```

`COMPLETED` with `0:0` is success. Then compare your results with the paper:

```bash
python scripts/verify_headline_numbers.py --results $ECDNA_MYDIR/runs/benchmark
```

Every row should say `MATCH`, and the last line
`VERDICT: all 12 model rows MATCH the published values`. If your job wrote its
tables elsewhere, the log names the folder in its last line
(`benchmark complete — results in ...`); pass that folder instead.

## 4.6 Changing how a job runs

Settings can be passed in front of `sbatch`:

```bash
MODEL=eccount_peaks SPLIT=test sbatch slurm/submit_sensitivity.sh
N_WORKERS=8 SEED=1234 sbatch slurm/submit_optimize_classical.sh
```

| Variable | Used by | Meaning |
|---|---|---|
| `ECDNA_PYTHON` | all | Which Python (set by `ecdna`) |
| `ECDNA_PROJECT_ROOT` | all | Which copy of the project (set by `ecdna`) |
| `MODEL` | sensitivity | Which method, or `all` |
| `SPLIT` | sensitivity | `train`, `val`, `test`, `all` |
| `N_WORKERS` | optimization, benchmark | Parallel workers |
| `SEED` | optimization | Random seed |
| `LOG_LEVEL` | most | `INFO` or `DEBUG` |
| `FORCE` | sensitivity | `1` recomputes instead of reusing |

For time, memory or partition, copy the script and edit the copy's `#SBATCH`
lines (`-t` time, `--mem` memory, `-p` partition).

## 4.7 If a job fails

Read the `.err` file first; the real error is almost always there.

```bash
ls -lt logs | head
tail -50 logs/<the newest .err file>
```

| `State` | Meaning | What to do |
|---|---|---|
| `COMPLETED` `0:0` | Success | |
| `OUT_OF_MEMORY` | Ran out of memory | Copy the script, raise `--mem` |
| `TIMEOUT` | Hit the time limit | Copy the script, raise `-t` |
| `FAILED` | The program stopped with an error | Read the `.err` file |
| Rejected at submission | Unknown partition or QOS | `sinfo -s`, then fix `-p` in your copy |

## 4.8 Quick commands on the login node

```bash
ecdna
python -m ecdna_bench.cli.run_qc --config configs/default.yaml
python -m ecdna_bench.cli.benchmark --config configs/default.yaml \
    --output-dir $ECDNA_MYDIR/runs/benchmark_$(date +%Y%m%d)
```

Always pass `--config configs/default.yaml`: your `paths.local.yaml` is merged
over it automatically. Anything longer than about ten minutes belongs in a job.

## 4.9 About training

Training reads the benchmark table and uses its `split` column: it learns from
`train`, selects the checkpoint on `val`, and never touches `test`.

```bash
ecdna
sbatch slurm/submit_eccount_train.sh
tail -f $ECDNA_MYDIR/runs/eccount_training/train_history.csv
```

`train_history.csv` gets one row per epoch. The published run trained for 70
epochs; its best validation loss, 0.6421, was at epoch 49.

**Model keys** for `--models`: `eccount_peaks`, `eccount_mask` (the threshold
mask), `label_engine`, `mia`, `classical`, `classical_before_opt`, `ecseg`.

---

# Part 5: Notebooks and figures

## 5.1 Opening a notebook

1. Go to <https://ondemand.rc.unc.edu> and sign in.
2. **Interactive Apps → Jupyter Notebook**; partition `general`, 4 hours,
   4 CPUs, 32 GB.
3. **Launch**, then **Connect to Jupyter** when the panel turns green.
4. Open your copy: `/proj/brunk_ecdna_cv_project/<ONYEN>/repos/ecdna-bench/notebooks`.
5. **Kernel → Change kernel → ecdna-bench (canonical)**.

Check the first cell of any notebook with:

```python
import sys, ecdna_bench
print(sys.executable)
print(ecdna_bench.__file__)
```

The first line must contain `envs/ecdna-bench`, the second your own copy.

`Shift-Enter` runs a cell; **Kernel → Restart & Run All** runs the whole
notebook from a clean start.

## 5.2 What each notebook produces

| Notebook | Produces |
|---|---|
| `01` | Figure 1: the dataset and a representative image |
| `02` | Figures 2 and 3: matching, and the classical pipeline |
| `03` | Figures 5 and 6: ecCount, and the six-method comparison |
| `04` | Supplementary: sensitivity analysis, OR versus AND |
| `05` | Statistical tests, heatmaps and the qualitative gallery |

Each figure panel comes with a `source_*.csv` holding exactly the numbers drawn.

## 5.3 Re-running a figure notebook safely

**Before running, point the output folder in the setup cell at your own
space**, for example:

```python
OUTDIR = "/proj/brunk_ecdna_cv_project/<ONYEN>/runs/figures/notebook03"
```

Otherwise the notebook rewrites the figure files in place. Expect small layout
differences (the plotting library moved from 3.8 to 3.10); the numbers do not
change.

## 5.4 The tutorial notebooks

`notebooks/tutorials/` has four step-by-step notebooks: images and the gold
standard; running ecCount; scoring against the gold standard; retraining. They
read the data in the BioImage Archive layout. On Longleaf, add this to the top
of the first code cell before running a notebook:

```python
import os
os.environ["ECDNA_DATA_ROOT"] = "/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/bia_view"
os.environ["ECCOUNT_WEIGHTS"] = "/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/release/model_checkpoints/eccount_best.pt"
```

`bia_view` is a read-only view of the lab's copy of the resource arranged like
the archive (`images/`, `predictions/`). The notebooks never write into it;
their outputs go to `runs/` in your copy.

---

# Part 6: Worked examples

## 6.1 "How do you know it hasn't just memorized these cell lines?"

Train on three cell lines, test on the fourth. `scripts/train_eccount_loco.py`
builds the split from the benchmark table, refuses to run if the held-out line
leaks into training, and writes a self-contained run folder.

**Step 1: check the splits** (two seconds, login node):

```bash
ecdna
python scripts/train_eccount_loco.py --hold-out all --dry-run
```

It prints, for each held-out line, the training and validation sizes and a
composition table. The held-out line must not appear under `train` or `val`.
Read the table; it is the evidence for the claim.

**Step 2: submit the four runs** (one array task per line):

```bash
sbatch slurm/submit_eccount_loco.sh
```

or, with the size-matched control for NCI-H2170 as a fifth task:

```bash
sbatch --array=0-4 slurm/submit_eccount_loco.sh
```

NCI-H2170 is 888 of the 1,145 benchmark images, so holding it out shrinks the
training set from 800 to 179 images. The control trains on 179 images drawn from
all four lines, which separates the effect of the unseen line from the effect of
less data.

Each task trains, predicts the held-out images and scores them, writing to
`$ECDNA_REPO/outputs/eccount_loco/holdout_<line>/`:
`best_model.pt`, `train_history.csv`, `split_composition.csv`,
`eval_metadata.csv`, `run_config.yaml`, `run_manifest.json` and `results/`.

**Step 3: compare seen and unseen**, on the held-out line's test images:

```bash
python - <<'EOF'
import os
import pandas as pd

LINE = "SNU16"
repo = os.environ["ECDNA_REPO"]
slug = LINE.lower().replace("-", "_")

def pooled_f1(path):
    d = pd.read_csv(path)
    d = d[(d.model == "ecCount (peaks)") & (d.cell_line == LINE) & (d.split == "test")]
    tp, fp, fn = d.obj_tp.sum(), d.obj_fp.sum(), d.obj_fn.sum()
    return 2 * tp / (2 * tp + fp + fn), len(d)

seen = pooled_f1(f"{repo}/release/frozen_results/or_matching/per_image_metrics.csv")
unseen = pooled_f1(f"{repo}/outputs/eccount_loco/holdout_{slug}/results/or_matching/per_image_metrics.csv")
print(f"{LINE} seen in training : F1 {seen[0]:.3f} on {seen[1]} test images")
print(f"{LINE} never seen       : F1 {unseen[0]:.3f} on {unseen[1]} test images")
EOF
```

What the paper reports (ecCount peaks, test images of each line):

| Held-out line | Test images | Seen in training | Never seen |
|---|---|---|---|
| COLO320DM | 11 | 0.886 | 0.860 |
| NCI-H2170 | 134 | 0.941 | 0.833 |
| NCI-H716 | 11 | 0.949 | 0.945 |
| SNU16 | 19 | 0.931 | 0.907 |

**Why not the old recipe.** It wrote its settings to
`configs/paths.loco_snu16.yaml`. The tools merge `configs/paths.local.yaml`
over any configuration file in `configs/`, and the local file wins, so training
silently used the original split and the original output folder. The LOCO
script writes `run_config.yaml` into the run folder and checks that no
`paths.local.yaml` sits beside it.

## 6.2 "Does it work on a cell line you never touched?"

SUM159PT is in the resource but not in the benchmark, because it has no manual
ROI masks. A second model, the **ROI model**, predicts metaphase outlines; its
masks for all 2,986 image sets are in the archive (`predicted_roi/`).

The ROI model is a residual U-Net that reads the RGB and DAPI images (four input
channels, 35,923,337 parameters), trained on the 1,145 manual masks with the
same partitions as ecCount. Its probability map is upsampled to the native
frame, smoothed (Gaussian, σ = 10 px), thresholded at 0.4, hole-filled, and the
component containing the image center is kept.

With predicted instead of manual outlines, ecCount (peaks) reached an
object-level F1 of 0.913 on the 1,145 benchmark images (0.942 with manual
outlines) and 0.896 on the 345 validation and test images. Two points must be
stated whenever such numbers are used:

1. The ROI model was trained on the 800 training images, so agreement measured
   on all 1,145 images is mostly on its training data; the 345-image figure is the
   fairer one.
2. Detections inside the predicted outline but outside the manual one are not
   annotated, so they count as false positives; the values are lower bounds.

Running the ROI model on new images: see the ROI section of the README (the
command-line entry point is added with the ROI module).

---

# Part 7: When something goes wrong

Error messages put the useful part at the **bottom**. Read the last lines first.

### `command not found` (for `pytest`, `python`, ...)

The session is not active. Type `ecdna`; the prompt shows `(ecdna-bench)`.

### `ModuleNotFoundError: No module named 'ecdna_bench'`

Start a session (`ecdna`) and check:

```bash
python -c "import ecdna_bench; print(ecdna_bench.__file__)"
```

If it still fails, re-run the setup script with `--check`.
**Do not run `pip install`**: the shared environment is read-only for you, and
pip then installs into `~/.local`, which every Python 3.10 on your account
reads, in every environment. That is exactly the kind of change the new setup
avoids.

### I installed something with pip by mistake, and other environments behave differently

Packages in `~/.local` are picked up by every environment with the same Python
version. See what is there, then switch it off in one reversible step:

```bash
ls ~/.local/lib/
ls ~/.local/lib/python3.10/site-packages | head -30
mv ~/.local/lib/python3.10 ~/.local/lib/python3.10.disabled
```

Move it back (`mv ~/.local/lib/python3.10.disabled ~/.local/lib/python3.10`) if
something you need was there. Sessions started with `ecdna` ignore `~/.local`
either way.

### My other conda environments changed after the first setup

The old setup moved conda's and pip's cache folders for every terminal. Run the
setup script with `--migrate`, then open a new terminal. To check nothing is
left:

```bash
grep -n "ecdna\|PIP_CACHE_DIR\|XDG_CACHE_HOME\|CONDA_PKGS_DIRS" ~/.bashrc
```

Only the block between `# >>> ecdna-bench shortcut (setup v2) >>>` and
`# <<< ecdna-bench shortcut (setup v2) <<<` should be listed.

### `No such file or directory: configs/benchmark.yaml`

Pass the configuration explicitly: `--config configs/default.yaml`.

### `Permission denied` when writing

You are writing into shared or published files. This is the protection working.
Use `--output-dir` with a folder of your own.

### Jupyter will not start, or the kernel dies at once

Usually a full home folder:

```bash
quota -s
du -sh ~/.cache ~/.local ~/.conda 2>/dev/null
```

Clearing download caches is safe: `rm -rf ~/.cache/pip`, and
`conda clean --tarballs --packages` (with `module load anaconda`).

### The kernel is listed but never connects

The shared environment's `ipykernel` may be newer than the OnDemand Jupyter
server expects. Ask the lab member who maintains the shared environment to
install a compatible version there, once, for everyone. Do not install it
yourself (see above).

### A job fails at once with `ModuleNotFoundError`, or runs from the wrong folder

The job was submitted without a session. `ecdna`, check `echo $ECDNA_PYTHON`
and `pwd`, then submit again.

### The setup check says image files are not readable

The benchmark table lists the location of every image, and your account must be
able to read them. If the check suggests a `consistency_csv` line, put it under
`paths:` in `$ECDNA_REPO/configs/paths.local.yaml` and run the check again.
Otherwise ask the lab member who looks after the shared data.

### A job fails at once with nothing in the log

`logs/` does not exist in the folder you submitted from: `mkdir -p logs`.

### `sbatch: error: invalid partition specified`

Partition names change after upgrades: `sinfo -s`, then fix `-p` in a copy of
the script.

### `Unknown model key 'eccount_threshold' — skipping.`

The key is `eccount_mask`.

### The numbers do not match the paper

1. **Wrong matching folder**: `or_matching/` is the published one.
2. **Wrong subset**: the paper quotes all 1,145 images and the 175 test images
   separately.
3. **Wrong averaging**: pooled F1 (0.942 for ecCount peaks) is published, not the
   mean of per-image F1 (0.931).

`python scripts/verify_headline_numbers.py --results <folder>` checks all three.

### A job crashed while processing many images

Code that sizes its worker pool from the machine's processor count can start far
more workers than SLURM allocated and run out of memory. Set `N_WORKERS` (or
`--n-workers`) to the number of CPUs requested.

### I have broken my `~/.bashrc`

Restore a backup (the setup script makes one before every change):

```bash
ls -lt ~/.bashrc.backup-*
cp ~/.bashrc.backup-YYYYMMDD-HHMMSS ~/.bashrc
```

If nothing works, `bash --norc` gives a shell without it, and
`nano ~/.bashrc` lets you fix it.

### I have overwritten a file in my copy

```bash
cd $ECDNA_REPO
git status
git restore path/to/the/file
```

---

# Part 8: Reference

## 8.1 Glossary

| Term | Meaning |
|---|---|
| **ONYEN** | Your UNC username |
| **Session** | A terminal after `ecdna`: project settings on, until `ecdna_off` |
| **Login node / compute node** | Where you land / where real work runs |
| **SLURM, partition** | The scheduler / a group of compute nodes |
| **Conda environment** | A folder with Python and libraries at fixed versions |
| **Kernel** | The engine a notebook uses to run code |
| **Git commit** | A permanent snapshot of the project |
| **YAML** | A settings file format (indent with spaces, never tabs) |
| **ecDNA** | Circles of DNA outside the chromosomes, often carrying oncogenes |
| **FISH** | Fluorescence in situ hybridization |
| **Metaphase spread** | One cell's DNA, arrested mid-division and spread flat |
| **ROI mask** | An outline of the spread to analyze |
| **Gold standard (GS)** | The manual annotation of every ecDNA; `gt` in archive file names |
| **Object F1** | Detection accuracy after one-to-one matching |
| **Count MAE / signed bias** | Average size / average direction of the counting error |
| **Pooled** | Sum TP, FP and FN over images, then compute once. **Published** |
| **OR / AND matching** | Pairing needs distance *or* overlap / both. **OR is published** |
| **Anchor image** | `ncih2170_facs_fish_0723_low_her2_52`, 116 gold-standard ecDNA: the running example |

## 8.2 The published numbers

Pooled, OR matching, all 1,145 benchmark images:

| Method | Object F1 | Count MAE | Signed bias |
|---|---|---|---|
| ecCount (peaks) | 0.942 | 13.1 | +0.4 |
| ecCount (threshold mask) | 0.917 | 18.1 | −11.4 |
| Label Engine | 0.825 | 36.0 | −28.8 |
| MIA | 0.800 | 47.0 | −40.6 |
| Classic (after opt) | 0.777 | 44.9 | −24.3 |
| ecSeg | 0.464 | 121.5 | −118.5 |

Test split only (175 images):

| Method | Object F1 | Count MAE |
|---|---|---|
| ecCount (peaks) | 0.939 | 13.3 |
| ecCount (threshold mask) | 0.916 | 17.6 |
| Label Engine | 0.825 | 34.4 |
| MIA | 0.813 | 40.4 |
| Classic (after opt) | 0.775 | 40.9 |
| ecSeg | 0.508 | 110.3 |

**Resource:** 2,986 image sets, five cell lines.
**Benchmark:** 1,145 image sets, four cell lines, split 800 / 170 / 175.
**ecCount:** 7,849,601 parameters; input 1,024 × 1,224 px; best at epoch 49,
validation loss 0.6421.
**Matching:** `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, OR policy,
8-connectivity, minimum object area 3 px.

Method names, spelled exactly this way everywhere:
`ecCount (peaks)` · `ecCount (threshold mask)` · `Label Engine` · `MIA` ·
`Classic (after opt)` · `ecSeg`

## 8.3 Environment provenance

| Artifact | Produced by | Environment |
|---|---|---|
| Every published metric | Cluster jobs | **`ecdna-bench`** |
| Trained model, training history | Cluster job | **`ecdna-bench`** |
| Figure files, notebooks 01–05 | Jupyter | `ecDNA_Hybrid` |

Measured difference between the two environments (30 August 2026):

| Package | `ecdna-bench` | `ecDNA_Hybrid` | Consequence |
|---|---|---|---|
| python, numpy, pandas, scikit-learn, Pillow, torch, seaborn | identical | identical | none |
| scikit-image, imageio, tifffile | 0.25.2 / 2.37.x / 2025.5.10 | effectively the same | none |
| scipy | 1.15.3 | 1.13.1 | the only gap that could move a value |
| statsmodels | 0.14.6 | 0.14.4 | patch level only |
| OpenCV | 4.13.0 | 4.10.0 | classical pipeline only, which ran in `ecdna-bench` |
| matplotlib | 3.10.0 | 3.8.4 | figure appearance only |

The paired Wilcoxon tests reported in the paper were recomputed from the released
per-image data under both SciPy versions, with identical results:

| Subset | n | W (two-sided) | p (two-sided) | W (greater) | p (greater) |
|---|---|---|---|---|---|
| All images | 1,145 | 118.0 | 1.50 × 10⁻¹⁸⁸ | 654,822.0 | 7.49 × 10⁻¹⁸⁹ |
| Test only | 175 | 3.0 | 1.90 × 10⁻³⁰ | 15,397.0 | 9.52 × 10⁻³¹ |

## 8.4 Where things are in the repository

```
ecdna-bench/
├── configs/
│   ├── default.yaml            every published setting; never edit
│   ├── paths.example.yaml      template for local paths
│   └── paths.local.yaml        yours; written by the setup script; not shared
├── env/                        environment recipes (GPU, CPU, pip)
├── src/ecdna_bench/
│   ├── cli/                    every command you can run
│   ├── data/                   catalog and file handling
│   ├── classical/              rule-based pipeline and its tuning
│   ├── eccount/                the ecCount network
│   ├── baselines/              converters for the external tools
│   ├── benchmark/              model registry, harmonization, scoring
│   └── evaluation/             objects, matching, metrics, statistics
├── slurm/                      cluster job files
├── scripts/                    helper tools (setup, LOCO, downloads, checks)
├── notebooks/                  01–05 figure notebooks; tutorials/
├── release/
│   ├── frozen_results/         or_matching/ (published) and and_matching/
│   ├── figures/notebookNN/     figure files and their source numbers
│   ├── split_files/            which image is in which split
│   ├── manifests/              the benchmark table and checksums
│   └── model_checkpoints/      the published ecCount weights
├── docs/                       tutorials and further documentation
└── tests/                      automated checks
```

## 8.5 Command card

```bash
# Start and end a session
ecdna
ecdna_off

# Where am I, is anything broken?
pwd
git status
squeue -u $USER

# Cluster jobs (inside a session, from the repository root)
sbatch slurm/submit_benchmark.sh
sbatch slurm/submit_eccount_infer.sh
sbatch slurm/submit_eccount_train.sh
python scripts/train_eccount_loco.py --hold-out all --dry-run
sbatch slurm/submit_eccount_loco.sh

# Score by hand, into a folder of your own
python -m ecdna_bench.cli.benchmark --config configs/default.yaml \
    --output-dir $ECDNA_MYDIR/runs/benchmark_$(date +%Y%m%d)

# Compare a results folder with the paper
python scripts/verify_headline_numbers.py --results <folder>

# Check a job
sacct -j JOBID --format=JobID,State,Elapsed,ExitCode
seff JOBID

# Re-check the setup
bash /proj/brunk_ecdna_cv_project/Poorya/ecdna-bench/scripts/setup_new_user.sh --check
```

**The three rules:** read from shared folders, write only to your own ·
`or_matching/` is the published one · pooled F1, not the per-image mean.

## 8.6 Getting help

**Cluster problems** (accounts, quotas, jobs, storage): <https://help.rc.unc.edu>.

**Project questions**: the Brunk Lab.

When asking about an error, include the exact command, the last twenty lines of
the error, and the output of `squeue -u $USER` for a cluster job.
