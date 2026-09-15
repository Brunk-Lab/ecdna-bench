#!/usr/bin/env python
"""
diagnose_river_roi.py — repair the subset labels and diagnose the ROI benchmark.

Run this after benchmark_river_roi.py has finished. It does five things:

  A. Repair the `subset` column, which was wrong in the first run.
  B. Explain the 13 images that have no ecCount prediction.
  C. Test whether `gt_image` is the right comparator, by checking it against
     the frozen benchmark ground-truth counts on the 1,145 benchmark images.
  D. Compare predicted ROIs against manual ROIs geometrically.
  E. Report the ground-truth burden distribution per cell line.

C and D are the ones that matter. The run produced positive count bias
everywhere (+9.4 overall, +26.1 on COLO320DM) where the benchmark reports
+0.4, and COLO320DM object precision of 0.419. Two explanations are on the
table and these checks separate them:

  (1) The predicted ROI covers regions the manual ROI excluded, so ecCount
      detects real ecDNA that the ground truth does not contain. Those become
      false positives and drive bias positive.
  (2) COLO320DM and SUM159PT have very low burden (implied means of 27 and 5
      ecDNA per image), so relative bias is unstable there and a handful of
      spurious objects reads as +98 %.

Both can be true at once. Nothing from this run should be reported until this
script has been read.

Usage:
    python scripts/diagnose_river_roi.py                 # all sections
    python scripts/diagnose_river_roi.py --skip-repair   # diagnosis only
    python scripts/diagnose_river_roi.py --sample 300    # faster C and D
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".PNG", ".TIF", ".TIFF")
MIN_AREA = 3
CONNECTIVITY = 8


def index_masks(folder: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    if not folder.exists():
        return out
    for p in sorted(folder.iterdir()):
        if p.is_file() and p.suffix in IMAGE_SUFFIXES:
            out.setdefault(p.stem, p)
    return out


def rule(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root",
                    default="/proj/brunk_ecdna_cv_project/Poorya/ecdna-bench")
    ap.add_argument("--gt-full-dir",
                    default="/proj/brunk_ecdna_cv_project/Poorya/ecDNA_Data/"
                            "bioimage_archive/gt_image")
    ap.add_argument("--roi-pred-dir",
                    default="/proj/brunk_ecdna_cv_project/Poorya/River/output/all")
    ap.add_argument("--sample", type=int, default=400,
                    help="Benchmark images to inspect in sections C and D. 0 = all.")
    ap.add_argument("--skip-repair", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root)
    river = repo / "release" / "river_roi_run"
    out_dir = river / "benchmark"

    src = repo / "src"
    sys.path.insert(0, str(src if src.exists() else repo))
    from ecdna_bench.data.io import load_mask
    from ecdna_bench.evaluation.objects import objects_from_mask

    def n_objects(path: Path) -> int:
        return len(objects_from_mask(load_mask(path), min_area=MIN_AREA,
                                     connectivity=CONNECTIVITY))

    bench_meta = pd.read_csv(repo / "release/manifests/metadata.csv")
    uid_col = "unique_id" if "unique_id" in bench_meta.columns else bench_meta.columns[0]
    bench_uids = set(bench_meta[uid_col].astype(str))
    train_uids = set(pd.read_csv(repo / "release/split_files/train_ids.csv")
                     .iloc[:, 0].astype(str))

    # ── A. Repair the subset labels ────────────────────────────────────────
    rule("A. Repairing the subset column")
    print("The first run set in_benchmark from a metadata dict populated from")
    print("BOTH full_counts_master.csv (2,986 rows) and metadata.csv (1,145),")
    print("so every image matched and everything was labelled 'benchmark'.")
    print("roi_model_saw_image was unaffected: 621+44+49+86 = 800, correct.\n")

    if args.skip_repair:
        print("--skip-repair: leaving files untouched.")
    else:
        targets = [
            out_dir / "per_image_metrics_river.partial.csv",
            out_dir / "per_image_metrics_river.csv",
            out_dir / "roi_retention_per_image.csv",
            out_dir / "inventory.csv",
        ]
        for path in targets:
            if not path.exists():
                print(f"  not present, skipping: {path.name}")
                continue
            df = pd.read_csv(path)
            if "uid" in df.columns:
                uid = df["uid"].astype(str)
            elif "unique_id" in df.columns:
                uid = df["unique_id"].astype(str)
            else:
                print(f"  no uid column, skipping: {path.name}")
                continue
            in_b = uid.isin(bench_uids)
            saw = uid.isin(train_uids)
            df["in_benchmark"] = in_b
            df["roi_model_saw_image"] = saw
            df["subset"] = np.where(
                ~in_b, "extension (not in benchmark)",
                np.where(saw, "benchmark train (SEEN by ROI model)",
                         "benchmark val+test (held out)"))
            df.to_csv(path, index=False)
            counts = df.groupby("subset")[uid.name].nunique()
            print(f"  repaired {path.name}")
            for k, v in counts.items():
                print(f"      {k:38s} {v:5d}")

        print("\nNow re-aggregate and redraw without re-scoring:")
        print("  python scripts/benchmark_river_roi.py --workers 1 --resume")

    # ── B. The 13 images with no prediction ────────────────────────────────
    rule("B. The 13 images with no ecCount prediction")
    gt_river_idx = index_masks(river / "gt_mask_river")
    pred_idx = index_masks(river / "masks" / "eccount_peaks")
    roi_idx = index_masks(Path(args.roi_pred_dir))
    missing = sorted(set(gt_river_idx) - set(pred_idx))
    print(f"Missing predictions: {len(missing)}")

    rows = []
    for uid in missing:
        roi_path = roi_idx.get(uid)
        if roi_path is None:
            rows.append({"uid": uid, "roi_pred_found": False,
                         "roi_fg_px": np.nan, "roi_n_components": np.nan,
                         "gt_river_objects": np.nan})
            continue
        roi = load_mask(roi_path)
        fg = int((roi > 0).sum())
        rows.append({
            "uid": uid, "roi_pred_found": True, "roi_fg_px": fg,
            "roi_n_components": len(objects_from_mask(roi, min_area=MIN_AREA,
                                                      connectivity=CONNECTIVITY)),
            "gt_river_objects": n_objects(gt_river_idx[uid]),
        })
    miss_df = pd.DataFrame(rows)
    miss_df["cell_line"] = miss_df["uid"].str.split("_").str[0]
    print(miss_df.to_string(index=False))
    miss_df.to_csv(out_dir / "missing_predictions.csv", index=False)

    empty = miss_df[(miss_df["roi_fg_px"] == 0) | (~miss_df["roi_pred_found"])]
    print(f"\nOf the {len(miss_df)}, {len(empty)} have an empty or absent predicted ROI.")
    if len(empty) == len(miss_df):
        print("VERDICT: every missing prediction has no ROI to run inside.")
        print("This is a reportable property of the ROI model, not a pipeline bug.")
    else:
        print("VERDICT: some have a non-empty ROI, so ecCount inference skipped them")
        print("for another reason. Check the inference log for these uids.")
    print("\nBy cell line:")
    print(miss_df.groupby("cell_line").size().to_string())

    # ── C. Is gt_image the right comparator? ───────────────────────────────
    rule("C. Is gt_image the same ground truth the benchmark used?")
    print("If gt_image holds the UNMASKED annotation while the benchmark scored")
    print("against an ROI-MASKED ground truth, then this run compared ecCount")
    print("against a different target and the numbers are not comparable.\n")

    frozen = pd.read_csv(repo / "release/frozen_results/or_matching/per_image_metrics.csv")
    bench_gt = (frozen[frozen["model"].astype(str).str.contains("peaks", case=False)]
                .set_index("uid")["gt_count"].astype(int))
    if bench_gt.empty:
        bench_gt = frozen.groupby("uid")["gt_count"].max().astype(int)

    gt_full_idx = index_masks(Path(args.gt_full_dir))
    common = sorted(set(bench_gt.index.astype(str)) & set(gt_full_idx))
    if args.sample and len(common) > args.sample:
        common = list(pd.Series(common).sample(args.sample, random_state=0))
    print(f"Comparing {len(common)} benchmark images.")

    rows = []
    for i, uid in enumerate(common, start=1):
        rows.append({
            "uid": uid,
            "gt_count_benchmark": int(bench_gt.loc[uid]),
            "gt_count_gt_image": n_objects(gt_full_idx[uid]),
            "gt_count_gt_river": (n_objects(gt_river_idx[uid])
                                  if uid in gt_river_idx else np.nan),
        })
        if i % 100 == 0:
            print(f"  {i}/{len(common)}", flush=True)

    cmp = pd.DataFrame(rows)
    cmp["delta_image_minus_bench"] = cmp["gt_count_gt_image"] - cmp["gt_count_benchmark"]
    cmp["ratio"] = cmp["gt_count_gt_image"] / cmp["gt_count_benchmark"].replace(0, np.nan)
    cmp.to_csv(out_dir / "gt_comparator_check.csv", index=False)

    n_eq = int((cmp["delta_image_minus_bench"] == 0).sum())
    print(f"\nExact matches           : {n_eq} / {len(cmp)} "
          f"({100 * n_eq / len(cmp):.1f} %)")
    print(f"median delta            : {cmp['delta_image_minus_bench'].median():+.1f}")
    print(f"median ratio            : {cmp['ratio'].median():.3f}")
    print(f"gt_image larger in      : {int((cmp['delta_image_minus_bench'] > 0).sum())}")
    print(f"gt_image smaller in     : {int((cmp['delta_image_minus_bench'] < 0).sum())}")

    if n_eq / max(len(cmp), 1) > 0.95:
        print("\nVERDICT: gt_image matches the benchmark ground truth. The comparator")
        print("is correct and the positive bias is a real property of running ecCount")
        print("inside predicted ROIs.")
    else:
        print("\nVERDICT: gt_image does NOT match the benchmark ground truth.")
        print("The ROI run scored against a different target. Every number from")
        print("benchmark_river_roi.py is uninterpretable until this is resolved.")
        print("Most likely gt_image is the unmasked annotation and the benchmark")
        print("applied the manual ROI on top of it.")

    # ── D. Predicted ROI versus manual ROI ─────────────────────────────────
    rule("D. Predicted ROI versus manual ROI")
    roi_col = next((c for c in ("roi_fullpath", "roi_mask_relpath")
                    if c in bench_meta.columns), None)
    if roi_col is None:
        print("No ROI path column in metadata.csv; skipping.")
    else:
        meta_idx = bench_meta.set_index(bench_meta[uid_col].astype(str))
        sub = [u for u in common if u in meta_idx.index and u in roi_idx][:args.sample or None]
        print(f"Comparing {len(sub)} images.\n")
        rows = []
        for uid in sub:
            rel = str(meta_idx.loc[uid, roi_col])
            man_path = Path(rel) if Path(rel).is_absolute() else repo / rel
            if not man_path.exists():
                continue
            man = load_mask(man_path) > 0
            pred = load_mask(roi_idx[uid]) > 0
            if man.shape != pred.shape:
                continue
            inter = int((man & pred).sum())
            union = int((man | pred).sum())
            rows.append({
                "uid": uid,
                "manual_px": int(man.sum()), "pred_px": int(pred.sum()),
                "area_ratio": pred.sum() / max(int(man.sum()), 1),
                "iou": inter / max(union, 1),
                "frac_manual_covered": inter / max(int(man.sum()), 1),
                "pred_outside_manual_px": int((pred & ~man).sum()),
                "pred_n_components": len(objects_from_mask(
                    pred.astype(np.uint8) * 255, min_area=50, connectivity=CONNECTIVITY)),
                "manual_n_components": len(objects_from_mask(
                    man.astype(np.uint8) * 255, min_area=50, connectivity=CONNECTIVITY)),
            })
        roi_df = pd.DataFrame(rows)
        if roi_df.empty:
            print("No comparable ROI pairs found.")
        else:
            roi_df.to_csv(out_dir / "roi_geometry_check.csv", index=False)
            print(roi_df[["area_ratio", "iou", "frac_manual_covered",
                          "pred_n_components", "manual_n_components"]]
                  .describe().round(3).to_string())
            bigger = float((roi_df["area_ratio"] > 1.2).mean())
            multi = float((roi_df["pred_n_components"] >
                           roi_df["manual_n_components"]).mean())
            print(f"\nPredicted ROI >20 % larger than manual : {100 * bigger:.1f} % of images")
            print(f"Predicted ROI has more components      : {100 * multi:.1f} % of images")
            print(f"Median IoU with manual ROI             : {roi_df['iou'].median():.3f}")
            if bigger > 0.3 or multi > 0.3:
                print("\nVERDICT: the predicted ROI routinely covers more of the frame than")
                print("the manual ROI. ecCount therefore sees metaphase spreads the manual")
                print("annotation never covered, and those detections score as false")
                print("positives. That is the mechanism behind the positive bias, and it")
                print("must be stated as a property of the comparison, not of ecCount.")

    # ── E. Burden distribution ─────────────────────────────────────────────
    rule("E. Ground-truth burden per cell line")
    disc = repo / "release/figures/notebook05/source_figS_gt_count_disclosure_long.csv"
    if disc.exists():
        d = pd.read_csv(disc)
        summ = (d.groupby("cell_line")["ecDNA_gt"]
                .agg(n="size", mean="mean", median="median",
                     frac_under_10=lambda s: float((s < 10).mean()))
                .reset_index().round(2))
        print(summ.to_string(index=False))
        print("\nRelative bias is unstable wherever burden is low. A cell line with a")
        print("mean of 5 objects per image cannot support a percentage-of-burden claim")
        print("the way one with a mean of 300 can. Report absolute counts there.")
    else:
        print(f"Not found: {disc}")

    print("\nDone. Read sections C and D before quoting anything from this run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
