"""Build the three ecdna-bench tutorial notebooks (no outputs).

    python scripts/build_tutorial_notebooks.py [OUT_DIR]     # default: notebooks/tutorials

The notebooks need only the installed ``ecdna_bench`` package and an
internet connection. They never read the repository folders and never
write a share link to disk. Files go to ``tutorial_data/`` next to the
notebook (or to ``$ECDNA_TUTORIAL_DATA``).
"""
import sys
from pathlib import Path

import nbformat as nbf

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/tutorials")

# SHA-256 of eccount_best.pt in release v1.0.0 (SHA256SUMS of the release).
ECCOUNT_SHA256 = "7f11c52ccc12d50178a839cbfe584681e72fc00c3efb7acf816f92e09e228e32"
ECCOUNT_URL = "https://github.com/Brunk-Lab/ecdna-bench/releases/download/v1.0.0/eccount_best.pt"

# Twelve held-out test image sets: for each cell line the lowest and highest
# gold-standard count, and the median slot (NCI-H2170: the paper's example image).
SAMPLE = [
    ("COLO320DM", "lowest", "colo320dm_qpcr_jq1_24h_dmso_111"),
    ("COLO320DM", "median", "colo320dm_qpcr_jq1_24h_ctrl_39"),
    ("COLO320DM", "highest", "colo320dm_qpcr_jq1_24h_ctrl_114"),
    ("NCI-H2170", "lowest", "ncih2170_facs_fish_0223_low_her2_6"),
    ("NCI-H2170", "example", "ncih2170_facs_fish_0723_low_her2_52"),
    ("NCI-H2170", "highest", "ncih2170_antibiotics_ps_g_36"),
    ("NCI-H716", "lowest", "ncih716_jc_ctrl_2_30"),
    ("NCI-H716", "median", "ncih716_jc_alo_8nm_24h_13"),
    ("NCI-H716", "highest", "ncih716_jc_ctrl_1_66"),
    ("SNU16", "lowest", "snu16_jc_ctrl_45"),
    ("SNU16", "median", "snu16_jc_jq1_ic50_24h_67"),
    ("SNU16", "highest", "snu16_jc_ctrl_114"),
]

SETUP = r'''
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import ecdna_bench

# All files of the tutorials live here (next to the notebook by default).
DATA = Path(os.environ.get("ECDNA_TUTORIAL_DATA", "tutorial_data")).expanduser().resolve()
print("ecdna_bench :", getattr(ecdna_bench, "__version__", "?"), "from", Path(ecdna_bench.__file__).parent)
print("data folder :", DATA)
'''.strip()

READ_HELPERS = r'''
import cv2
cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)   # hide notes about extra TIFF tags

def read_rgb(path):
    """RGB image as uint8 (H, W, 3), read the same way as the benchmark."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def read_gray(path):
    """Single-channel image (TIFF or PNG); colour images are reduced by their maximum."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        import tifffile
        img = tifffile.imread(str(path))
    img = np.asarray(img)
    return img.max(axis=2) if img.ndim == 3 else img

def load_sample():
    """The table written by notebook 1."""
    table = DATA / "sample.csv"
    if not table.is_file():
        raise FileNotFoundError(f"{table} not found. Run notebook 1 first "
                                "(or set ECDNA_TUTORIAL_DATA to its data folder).")
    return pd.read_csv(table)
'''.strip()


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (ipykernel)",
                                 "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    return nb


# ---------------------------------------------------------------------------
# Notebook 1
# ---------------------------------------------------------------------------
sample_literal = "SAMPLE = [\n" + "".join(
    f"    ({cl!r}, {slot!r}, {uid!r}),\n" for cl, slot, uid in SAMPLE) + "]"

nb1 = notebook([
    md("""
# 1. Get the sample and look at it

This notebook downloads 12 image sets from the ecdna-bench imaging resource
(BioImage Archive, accession S-BIAD4097) and shows what an image set contains:
the RGB FISH image, the manual region-of-interest (ROI) mask and the
gold-standard ecDNA mask. It then counts ecDNA in the gold standard the way the
benchmark does.

The 12 image sets are held-out test images, three per cell line: the lowest and
the highest gold-standard count, and one from the middle of the range (for
NCI-H2170 this is the example image used throughout the paper).

**Needs:** the installed `ecdna_bench` package and an internet connection.
Everything is written to `tutorial_data/` next to this notebook; notebooks 2
and 3 read from there. Downloads resume if interrupted: just run the cells again.
"""),
    code(SETUP + "\n\n" + sample_literal + "\n\nUIDS = [uid for _, _, uid in SAMPLE]\n"
         "EXAMPLE_UID = \"ncih2170_facs_fish_0723_low_her2_52\"\n"
         "DATA.mkdir(parents=True, exist_ok=True)"),
    md("""
## Where the files come from

After the record is published, the files are public and nothing needs to be
entered. While the record is private (during peer review), the next cell asks
for the **share link** from the reviewer instructions
(`https://www.ebi.ac.uk/biostudies/bioimages/studies/S-BIAD4097?key=...`).
The link works like a password: the prompt hides it, and the notebook never
prints it or writes it to disk. You can also set it in the shell before
starting Jupyter: `export ECDNA_BIA_BASE_URL='<share link>'`.
"""),
    code(r'''
import getpass, json, re, shutil, time
import urllib.error, urllib.parse, urllib.request

ACCESSION = "S-BIAD4097"
INFO_API = f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{ACCESSION}/info"
PUBLIC_FILES = f"https://ftp.ebi.ac.uk/pub/databases/biostudies/S-BIAD/097/{ACCESSION}/Files"
HEADERS = {"User-Agent": "ecdna-bench-tutorial/1.0"}

def _get(url, timeout=60):
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=timeout) as r:
        return r.read()

def hide_key(url):
    """The address with any access key replaced by <key>."""
    url = re.sub(r"(/\.private/\d+/)[^/]+", r"\1<key>", url)
    return re.sub(r"([?&](?:key|accessKey|token)=)[^&#/]+", r"\1<key>", url, flags=re.I)

def files_location(link=""):
    """URL of the study's Files folder, from a share link, a Files URL, or nothing (public)."""
    link = (link or "").strip()
    if link and not link.startswith(("http://", "https://")):
        raise ValueError("That is not a web address; paste the whole share link (it starts with https://).")
    key = ""
    if link:
        parts = urllib.parse.urlsplit(link)
        if "/studies/" in parts.path:
            key = urllib.parse.parse_qs(parts.query).get("key", [""])[0]
        if not key:
            return link.rstrip("/")            # already a Files address
    info_url = f"{INFO_API}?key={urllib.parse.quote(key)}" if key else INFO_API
    try:
        info = json.loads(_get(info_url, timeout=30).decode("utf-8"))
        loc = info.get("httpLink") or info.get("ftpHttp_link")
    except urllib.error.HTTPError as exc:
        if key:
            raise RuntimeError(f"The BioStudies API refused the share link (HTTP {exc.code}). "
                               "Check that the whole link was copied.") from None
        loc = None
    except (urllib.error.URLError, OSError, ValueError):
        if key:
            raise RuntimeError("Could not reach the BioStudies API to resolve the share link.") from None
        loc = None
    if not loc:
        if key:
            raise RuntimeError("The BioStudies API returned no file location for this share link.")
        return PUBLIC_FILES
    loc = str(loc).rstrip("/")
    return loc if loc.endswith("/Files") else loc + "/Files"

def download(rel, retries=3):
    """Download one archive file to DATA/<rel> (kept if already complete)."""
    dest = DATA / rel
    if dest.is_file() and dest.stat().st_size > 0:
        return "kept"
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = FILES + "/" + urllib.parse.quote(rel)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=120) as r, \
                 open(part, "wb") as fh:
                expected = r.headers.get("Content-Length")
                shutil.copyfileobj(r, fh, 1 << 20)
            if expected is not None and part.stat().st_size != int(expected):
                raise IOError("incomplete download")
            part.replace(dest)
            return "downloaded"
        except (urllib.error.URLError, OSError) as exc:
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"could not download {rel} ({last.__class__.__name__}: "
                       f"{getattr(last, 'code', '') or getattr(last, 'reason', '')})")
'''),
    code(r'''
link = os.environ.get("ECDNA_BIA_BASE_URL", "").strip()
if not link:
    try:
        link = getpass.getpass("Share link (press Enter if the record is public): ").strip()
    except Exception:          # no interactive input available (e.g. a headless run)
        link = ""
FILES = files_location(link)
del link
print("files from:", hide_key(FILES))
'''),
    md("""
## Select the 12 image sets

The archive publishes one file list per section. The next cell reads four of
them (images, gold standard, ROI, predictions) and picks the files of the 12
image sets: the RGB image, the gold-standard mask, the manual ROI mask, and the
masks predicted by the six benchmarked methods (used in notebook 3).
"""),
    code(r'''
LISTS = ["filelist_images.tsv", "filelist_gt.tsv", "filelist_roi.tsv", "filelist_predictions.tsv"]
PRED_FOLDERS = {                        # archive folder -> method name used in the paper
    "eccount_peaks": "ecCount (peaks)",
    "eccount_threshold": "ecCount (threshold mask)",
    "label_engine": "Label Engine",
    "mia": "MIA",
    "classical_optimised": "Classic (after opt)",
    "ecseg": "ecSeg",
}

tables = {}
for name in LISTS:
    dest = DATA / "filelists" / name
    if not (dest.is_file() and dest.stat().st_size > 0):
        try:
            data = _get(FILES + "/" + name)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(f"Could not read {name} (HTTP {exc.code}). If the record is not public "
                               "yet, run the previous cell again and paste the share link.") from None
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Could not reach the archive to read {name} "
                               f"({getattr(exc, 'reason', exc.__class__.__name__)}). "
                               "Check the internet connection and run this cell again.") from None
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
    t = pd.read_csv(dest, sep="\t", dtype=str)
    t["uid"] = t["Files"].map(lambda p: Path(p).stem)
    t["folder"] = t["Files"].map(lambda p: "/".join(p.split("/")[:-1]))
    tables[name] = t[t["uid"].isin(UIDS)]

images = tables["filelist_images.tsv"]
rgb = images[images["folder"] == "images/rgb"].set_index("uid")
gt = tables["filelist_gt.tsv"].query("folder == 'images/gt_image'").set_index("uid")
roi = tables["filelist_roi.tsv"].query("folder == 'images/roi_mask'").set_index("uid")
preds = tables["filelist_predictions.tsv"].copy()
preds["method_folder"] = preds["folder"].str.split("/").str[1]
preds = preds[preds["method_folder"].isin(PRED_FOLDERS)]

wanted = list(rgb.loc[UIDS, "Files"]) + list(gt.loc[UIDS, "Files"]) \
       + list(roi.loc[UIDS, "Files"]) + list(preds["Files"])
missing = [u for u in UIDS if u not in rgb.index or u not in gt.index or u not in roi.index]
assert not missing, f"not in the file lists: {missing}"
assert len(preds) == len(UIDS) * len(PRED_FOLDERS), f"expected {len(UIDS) * len(PRED_FOLDERS)} prediction files, found {len(preds)}"
print(f"{len(wanted)} files to fetch for {len(UIDS)} image sets")
'''),
    code(r'''
t0, status = time.time(), {"downloaded": 0, "kept": 0}
for i, rel in enumerate(wanted, 1):
    status[download(rel)] += 1
    if i % 20 == 0 or i == len(wanted):
        print(f"  {i}/{len(wanted)} files", flush=True)
size_mb = sum((DATA / rel).stat().st_size for rel in wanted) / 1e6
print(f"done in {time.time() - t0:.0f} s: {status['downloaded']} downloaded, {status['kept']} already here, "
      f"{size_mb:.0f} MB in {DATA}")

sample = pd.DataFrame([{
    "uid": uid, "cell_line": cl, "slot": slot,
    "split": rgb.loc[uid, "Split"],
    "gt_count_archive": int(rgb.loc[uid, "ecDNA Count"]),
    "rgb": rgb.loc[uid, "Files"], "gt": gt.loc[uid, "Files"], "roi": roi.loc[uid, "Files"],
} for cl, slot, uid in SAMPLE])
sample.to_csv(DATA / "sample.csv", index=False)
preds.assign(method=preds["method_folder"].map(PRED_FOLDERS))[["uid", "method", "Files"]] \
     .rename(columns={"Files": "path"}).to_csv(DATA / "predictions.csv", index=False)
sample[["cell_line", "slot", "uid", "split", "gt_count_archive"]]
'''),
    md("""
## One image set

The RGB image is a FISH image of a metaphase spread: DNA (DAPI) in blue and the
amplified locus in the FISH colour. The manual ROI mask marks the metaphase
spread; everything outside it is ignored. The gold-standard mask marks each
ecDNA as a small object. Below: the paper's example image, with the ROI outline
and a zoomed region.
"""),
    code(READ_HELPERS + r'''

row = load_sample().set_index("uid").loc[EXAMPLE_UID]
img, roi_mask, gt_mask = read_rgb(DATA / row["rgb"]), read_gray(DATA / row["roi"]), read_gray(DATA / row["gt"])
print("image", img.shape, img.dtype, "| ROI", roi_mask.shape, "| gold standard", gt_mask.shape, np.unique(gt_mask))

x0, y0, x1, y1 = 1247, 1225, 1458, 1353        # zoom window used in the paper's figures
fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(img); ax[0].contour(roi_mask > 0, levels=[0.5], colors="w", linewidths=1)
ax[0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1.5))
ax[0].set_title("RGB image, ROI outline (white), zoom (yellow)")
ax[1].imshow(img[y0:y1, x0:x1]); ax[1].set_title("zoom")
ax[2].imshow(img[y0:y1, x0:x1])
ax[2].contour(gt_mask[y0:y1, x0:x1] > 0, levels=[0.5], colors="lime", linewidths=1)
ax[2].set_title("zoom with gold-standard objects (green)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()
'''),
    md("""
## Count ecDNA the way the benchmark does

The benchmark counts objects as 8-connected components of at least 3 px in a
mask. The same rule is applied to every method's prediction and to the gold
standard, so counts are comparable. `ecdna_bench.evaluation.objects_from_mask`
implements it; the counts below should equal the count recorded in the archive.
"""),
    code(r'''
from ecdna_bench.evaluation import objects_from_mask

sample = load_sample()
sample["gt_count"] = [len(objects_from_mask(read_gray(DATA / p), min_area=3, connectivity=8,
                                            attach_mask=False)) for p in sample["gt"]]
sample["matches_archive"] = sample["gt_count"] == sample["gt_count_archive"]
print(f"counts equal to the archive for {int(sample['matches_archive'].sum())} of {len(sample)} image sets")
sample[["cell_line", "slot", "gt_count", "gt_count_archive", "matches_archive"]]
'''),
    code(r'''
fig, axes = plt.subplots(4, 3, figsize=(12, 15))
for ax, r in zip(axes.ravel(), sample.itertuples()):
    im, m = read_rgb(DATA / r.rgb), read_gray(DATA / r.roi) > 0
    ys, xs = np.where(m)
    ax.imshow(im[ys.min():ys.max() + 1, xs.min():xs.max() + 1])
    ax.set_title(f"{r.cell_line}, {r.slot}: {r.gt_count} ecDNA", fontsize=10)
    ax.axis("off")
plt.suptitle("The 12 image sets, cropped to the ROI", y=1.0)
plt.tight_layout(); plt.show()
'''),
    md("""
## Next

* **Notebook 2** runs ecCount on these 12 images on the CPU.
* **Notebook 3** scores ecCount and the other benchmarked methods against the
  gold standard on the same images.

To download more (a whole split, one cell line, or the full resource), use
`scripts/fetch_bia_subset.py` from the repository; see `docs/TUTORIAL_EXTERNAL.md`.
"""),
])

# ---------------------------------------------------------------------------
# Notebook 2
# ---------------------------------------------------------------------------
nb2 = notebook([
    md("""
# 2. Count ecDNA with ecCount

This notebook downloads the released ecCount weights and runs the model on the
12 image sets from notebook 1, on the CPU (a few minutes in total).

The steps are the ones used for the paper: the image is resized to
1224 × 1024 px, pixels outside the ROI are set to zero, the network gives a
probability map, and local maxima of the smoothed map (above 0.35) are the
detected ecDNA. Each detection is drawn as a small diamond at full resolution
(the *peaks* mask); the probability map thresholded at 0.5 gives the
*threshold* mask.

**Needs:** notebook 1 run first (it provides `tutorial_data/`).
"""),
    code(SETUP + "\n\n" + READ_HELPERS + r'''

sample = load_sample()
for col in ("rgb", "roi", "gt"):
    missing = [p for p in sample[col] if not (DATA / p).is_file()]
    assert not missing, f"{len(missing)} {col} file(s) missing; run notebook 1 again"
print(len(sample), "image sets ready")
'''),
    md("""
## The weights

The weights (`eccount_best.pt`) are an asset of the repository's release
v1.0.0. The cell downloads them once into `tutorial_data/` and checks the
SHA-256 checksum published with the release. If you already have the file, set
`ECCOUNT_WEIGHTS=/path/to/eccount_best.pt` before starting Jupyter.
"""),
    code(f'''
import hashlib, urllib.request

ECCOUNT_URL = "{ECCOUNT_URL}"
ECCOUNT_SHA256 = "{ECCOUNT_SHA256}"

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()

WEIGHTS = Path(os.environ.get("ECCOUNT_WEIGHTS", DATA / "eccount_best.pt")).expanduser()
if not WEIGHTS.is_file():
    print("downloading", ECCOUNT_URL)
    part = WEIGHTS.with_name(WEIGHTS.name + ".part")
    req = urllib.request.Request(ECCOUNT_URL, headers={{"User-Agent": "ecdna-bench-tutorial/1.0"}})
    with urllib.request.urlopen(req, timeout=300) as r, open(part, "wb") as fh:
        while block := r.read(1 << 20):
            fh.write(block)
    part.replace(WEIGHTS)
digest = sha256(WEIGHTS)
if digest != ECCOUNT_SHA256:
    raise ValueError(f"{{WEIGHTS}} is not the released eccount_best.pt (SHA-256 {{digest[:12]}}...). "
                     "Delete it and run this cell again.")
print("weights OK:", WEIGHTS, f"({{WEIGHTS.stat().st_size / 1e6:.0f}} MB)")
'''),
    code(r'''
import torch
from ecdna_bench.eccount.model import ModelConfig, build_model
from ecdna_bench.eccount.infer import InferConfig, infer_one, load_checkpoint
from ecdna_bench.eccount.postprocess import PostprocessConfig, peaks_to_mask

DEVICE = os.environ.get("ECCOUNT_DEVICE", "cpu")
model = build_model(ModelConfig())            # the published architecture
try:
    ckpt = load_checkpoint(str(WEIGHTS), model, device=DEVICE)
except Exception as exc:                        # older checkpoints need the full unpickler
    if "weights_only" not in str(exc):
        raise
    ckpt = torch.load(str(WEIGHTS), map_location=DEVICE, weights_only=False)   # checksum verified above
    model.load_state_dict(ckpt["model_state_dict"])
model.eval().to(DEVICE)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"ecCount U-Net: {n_params:,} parameters; checkpoint from epoch {ckpt.get('best_epoch', '?')}")

PP = PostprocessConfig()                         # frozen paper values
CFG = InferConfig(postprocess=PP, threshold_mask_cutoff=0.5, peaks_disk_radius=PP.point_disk_radius)
INPUT_H, INPUT_W = 1024, 1224
print(PP)
'''),
    md("""
## Run ecCount

`run_eccount` reproduces the benchmark's inference for one image. The predicted
count is the number of objects in the peaks mask, counted with the same rule as
the gold standard (8-connected components of at least 3 px).
"""),
    code(r'''
import time
from ecdna_bench.evaluation import objects_from_mask

def count_objects(mask):
    return len(objects_from_mask(mask, min_area=3, connectivity=8, attach_mask=False))

def run_eccount(rgb_path, roi_path):
    rgb = read_rgb(rgb_path)
    orig_h, orig_w = rgb.shape[:2]
    small = cv2.resize(rgb, (INPUT_W, INPUT_H), interpolation=cv2.INTER_AREA)
    roi = (cv2.resize(read_gray(roi_path).astype(np.uint8), (INPUT_W, INPUT_H),
                      interpolation=cv2.INTER_NEAREST) > 0).astype(np.uint8)
    small = small * roi[:, :, None]
    x = torch.from_numpy(small.astype(np.float32).transpose(2, 0, 1) / 255.0).unsqueeze(0).to(DEVICE)
    res = infer_one(x, roi, model, CFG)
    sx, sy = orig_w / INPUT_W, orig_h / INPUT_H
    peaks = [(int(round(px * sx)), int(round(py * sy)), s) for px, py, s in res.peaks]
    peaks_mask = peaks_to_mask(peaks, shape=(orig_h, orig_w), disk_radius=CFG.peaks_disk_radius)
    thr_mask = cv2.resize(res.threshold_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return res.prob_map, peaks, peaks_mask, thr_mask

OUT = DATA / "my_eccount"
(OUT / "peaks").mkdir(parents=True, exist_ok=True)
(OUT / "threshold").mkdir(parents=True, exist_ok=True)
released = pd.read_csv(DATA / "predictions.csv").query("method == 'ecCount (peaks)'").set_index("uid")["path"]

rows, t0 = [], time.time()
for r in sample.itertuples():
    t = time.time()
    prob, peaks, peaks_mask, thr_mask = run_eccount(DATA / r.rgb, DATA / r.roi)
    cv2.imwrite(str(OUT / "peaks" / f"{r.uid}.png"), peaks_mask)
    cv2.imwrite(str(OUT / "threshold" / f"{r.uid}.png"), thr_mask)
    rows.append({"cell_line": r.cell_line, "slot": r.slot, "uid": r.uid,
                 "gold_standard": count_objects(read_gray(DATA / r.gt)),
                 "ecCount_here": count_objects(peaks_mask),
                 "ecCount_released": count_objects(read_gray(DATA / released[r.uid])),
                 "seconds": round(time.time() - t, 1)})
    print(f"  {r.uid}: {rows[-1]['ecCount_here']} ecDNA ({rows[-1]['seconds']} s)", flush=True)
counts = pd.DataFrame(rows)
counts.to_csv(OUT / "counts.csv", index=False)
print(f"12 images in {time.time() - t0:.0f} s; masks in {OUT}")
counts.drop(columns="seconds")
'''),
    md("""
`ecCount_released` is the count from the deposited ecCount (peaks) predictions,
which were computed on a GPU. CPU and GPU arithmetic differ slightly, so a few
peaks near the 0.35 cut-off can appear or disappear; the counts should agree
closely but not always exactly.
"""),
    code(r'''
ex = sample.set_index("uid").loc["ncih2170_facs_fish_0723_low_her2_52"]
prob, peaks, peaks_mask, _ = run_eccount(DATA / ex["rgb"], DATA / ex["roi"])
img, gt_mask = read_rgb(DATA / ex["rgb"]), read_gray(DATA / ex["gt"])
x0, y0, x1, y1 = 1247, 1225, 1458, 1353
prob_full = cv2.resize(prob, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
pk = np.array([(px, py) for px, py, _ in peaks if x0 <= px < x1 and y0 <= py < y1])

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
ax[0].imshow(img[y0:y1, x0:x1]); ax[0].set_title("zoom of the example image")
ax[1].imshow(prob_full[y0:y1, x0:x1], vmin=0, vmax=1, cmap="magma"); ax[1].set_title("ecCount probability map")
ax[2].imshow(img[y0:y1, x0:x1])
ax[2].contour(gt_mask[y0:y1, x0:x1] > 0, levels=[0.5], colors="lime", linewidths=1)
if len(pk):
    ax[2].scatter(pk[:, 0] - x0, pk[:, 1] - y0, s=18, facecolors="none", edgecolors="magenta")
ax[2].set_title("gold standard (green) and ecCount peaks (magenta)")
for a in ax: a.axis("off")
plt.tight_layout(); plt.show()

fig, ax = plt.subplots(figsize=(5, 5))
for cl, g in counts.groupby("cell_line"):
    ax.scatter(g["gold_standard"], g["ecCount_here"], label=cl, s=40)
lim = [0, max(counts["gold_standard"].max(), counts["ecCount_here"].max()) * 1.05]
ax.plot(lim, lim, "k--", lw=0.8); ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("gold-standard count"); ax.set_ylabel("ecCount count (this run)"); ax.legend()
plt.tight_layout(); plt.show()
'''),
    md("""
## Next

**Notebook 3** scores these predictions, and the deposited predictions of all
benchmarked methods, against the gold standard.

To run ecCount on your own images, see route D in `docs/TUTORIAL_EXTERNAL.md`.
"""),
])

# ---------------------------------------------------------------------------
# Notebook 3
# ---------------------------------------------------------------------------
nb3 = notebook([
    md("""
# 3. Score and compare methods

This notebook scores the predictions of the six benchmarked methods (as
deposited in the archive) and, if notebook 2 was run, your own ecCount run,
against the gold standard on the 12 image sets. It uses the benchmark's
object-matching rules, from `ecdna_bench.evaluation`.

**These numbers are illustrative.** Twelve images chosen to span the count
range are not a representative sample; the paper's values come from all 175
test images (or all 1,145 benchmark images). To reproduce those, follow route B
in `docs/TUTORIAL_EXTERNAL.md`.

**Needs:** notebook 1 run first.
"""),
    code(SETUP + "\n\n" + READ_HELPERS + r'''

sample = load_sample()
pred_table = pd.read_csv(DATA / "predictions.csv")
METHODS = ["ecCount (peaks)", "ecCount (threshold mask)", "Label Engine", "MIA",
           "Classic (after opt)", "ecSeg"]
paths = {m: pred_table[pred_table["method"] == m].set_index("uid")["path"].map(lambda p: DATA / p)
         for m in METHODS}
own = DATA / "my_eccount" / "peaks"
if own.is_dir() and all((own / f"{u}.png").is_file() for u in sample["uid"]):
    METHODS.append("ecCount (peaks), this run")
    paths[METHODS[-1]] = pd.Series({u: own / f"{u}.png" for u in sample["uid"]})
print("methods:", ", ".join(METHODS))
'''),
    md("""
## From deposited files to binary masks

Each method's output is turned into a binary mask with the benchmark's rules:
any non-zero pixel is foreground, except for Label Engine, whose deposited
output is its raw RGB rendering; it is converted to grey, kept above 0.5, and
8-connected components smaller than 3 px are removed.
"""),
    code(r'''
def prediction_mask(method, path):
    if method == "Label Engine":
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.shape[2] == 3 else img[:, :, 0]
        fg = (img.astype(np.float32) > 0.5).astype(np.uint8)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        keep = np.zeros(n, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 3
        return keep[lab]
    return read_gray(path) > 0
'''),
    md("""
## Matching predictions to the gold standard

Objects are 8-connected components of at least 3 px. A predicted object and a
gold-standard object form a candidate pair if their centroids are at most
20 px apart **or** their intersection over union is at least 0.1 (the OR
policy). The Hungarian algorithm then pairs objects one to one, minimising
`0.5 × (1 − IoU) + 0.5 × distance / 20`. Paired objects are true positives,
unpaired gold-standard objects false negatives, and unpaired predictions false
positives, except predictions that had a candidate but lost it to another
prediction: those are *ignored* and do not count as false positives.
"""),
    code(r'''
from ecdna_bench.evaluation import (objects_from_mask, precompute_pairwise,
                                    resolve_matching_from_pairwise, object_metrics_from_counts)

from skimage.measure import label, regionprops

D_MAX, IOU_MIN, ALPHA = 20.0, 0.1, 0.5

class ObjectMask:
    """One object's mask, read from the label image on demand. It gives the same
    pixels as a full-frame mask per object but needs far less memory on images
    with hundreds of ecDNA."""
    def __init__(self, labels, value):
        self.labels, self.value, self.shape = labels, value, labels.shape
    def __getitem__(self, index):
        return (self.labels[index] == self.value).astype(np.uint8)

def objects(mask):
    """8-connected components of at least 3 px, as the benchmark extracts them."""
    mask = np.asarray(mask) > 0
    objs = objects_from_mask(mask, min_area=3, connectivity=8, attach_mask=False)
    labels = label(mask, connectivity=2)
    values = [p.label for p in regionprops(labels) if p.area >= 3]
    assert len(values) == len(objs)
    for obj, value in zip(objs, values):
        obj["mask"] = ObjectMask(labels, value)
    return objs

def score(gt_mask, pred_mask):
    gt_objs = objects(gt_mask)
    pred_objs = objects(pred_mask)
    pw = precompute_pairwise(pred_objs, gt_objs, max_precompute_dist=D_MAX)
    m = resolve_matching_from_pairwise(pw, d_max=D_MAX, min_iou=IOU_MIN, alpha=ALPHA, policy="OR")
    return {"tp": m.tp, "fp": m.fp, "fn": m.fn, "ignored": m.ignored,
            "gt_count": len(gt_objs), "pred_count": len(pred_objs),
            "f1": object_metrics_from_counts(m.tp, m.fp, m.fn)["f1"]}

rows = []
for r in sample.itertuples():
    gt_mask = read_gray(DATA / r.gt)
    for method in METHODS:
        pred = prediction_mask(method, paths[method][r.uid])
        assert pred.shape == gt_mask.shape, (method, r.uid, pred.shape, gt_mask.shape)
        rows.append({"uid": r.uid, "cell_line": r.cell_line, "method": method, **score(gt_mask, pred)})
    print(f"  {r.uid} scored", flush=True)
per_image = pd.DataFrame(rows)
per_image.to_csv(DATA / "scores_per_image.csv", index=False)
per_image.query("uid == 'ncih2170_facs_fish_0723_low_her2_52'").drop(columns=["uid", "cell_line"])
'''),
    md("""
## Pooled over the 12 images

As in the paper, object F1 is computed from the summed true positives, false
positives and false negatives of all images, and count error is the predicted
minus the gold-standard count per image (MAE: mean absolute error; bias: mean
signed error).
"""),
    code(r'''
per_image["count_error"] = per_image["pred_count"] - per_image["gt_count"]
pooled = per_image.groupby("method", sort=False).agg(
    tp=("tp", "sum"), fp=("fp", "sum"), fn=("fn", "sum"),
    count_mae=("count_error", lambda e: e.abs().mean()), count_bias=("count_error", "mean"))
pooled["object_f1"] = 2 * pooled.tp / (2 * pooled.tp + pooled.fp + pooled.fn)
pooled = pooled.loc[METHODS, ["tp", "fp", "fn", "object_f1", "count_mae", "count_bias"]]
pooled.to_csv(DATA / "scores_pooled.csv")
pooled.round({"object_f1": 3, "count_mae": 1, "count_bias": 1})
'''),
    code(r'''
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
ax[0].barh(pooled.index[::-1], pooled["object_f1"][::-1], color="steelblue")
ax[0].set_xlim(0, 1); ax[0].set_xlabel("object F1 (12 images, pooled)")
for y, v in enumerate(pooled["object_f1"][::-1]):
    ax[0].text(v + 0.01, y, f"{v:.3f}", va="center", fontsize=9)
for method in METHODS:
    g = per_image[per_image["method"] == method]
    ax[1].scatter(g["gt_count"], g["pred_count"], s=25, label=method)
lim = [0, per_image[["gt_count", "pred_count"]].to_numpy().max() * 1.05]
ax[1].plot(lim, lim, "k--", lw=0.8); ax[1].set_xlim(lim); ax[1].set_ylim(lim)
ax[1].set_xlabel("gold-standard count"); ax[1].set_ylabel("predicted count"); ax[1].legend(fontsize=8)
plt.suptitle("Illustrative comparison on 12 test images (not the paper's values)")
plt.tight_layout(); plt.show()
'''),
    md("""
## Next

* Reproduce the paper's tables on the 175 test images or all 1,145 benchmark
  images: route B in `docs/TUTORIAL_EXTERNAL.md` (download with
  `scripts/fetch_bia_subset.py`, prepare with `scripts/prepare_local_run.py`,
  score with `python -m ecdna_bench.cli.benchmark`).
* Score your own method: write one binary mask per image (same size as the
  gold standard) and pass it through `prediction_mask` and `score` above.
"""),
])

OUT.mkdir(parents=True, exist_ok=True)
for name, nb in [("01_get_the_sample.ipynb", nb1), ("02_count_with_eccount.ipynb", nb2),
                 ("03_score_and_compare.ipynb", nb3)]:
    nbf.validate(nb)
    nbf.write(nb, OUT / name)
    print("wrote", OUT / name)
