import pandas as pd

r = pd.read_csv("release/river_roi_run/preparation_report.csv")

def band(d, name):
    return {"subset": name, "n": len(d),
            "median": d.kept_frac.median(),
            "p10": d.kept_frac.quantile(.10),
            "p01": d.kept_frac.quantile(.01),
            "empty": int((d.kept_frac == 0).sum())}

ext = r[r.split == "extension"]
print(pd.DataFrame([
    band(r[r.split == "train"],            "benchmark train (ROI model saw these)"),
    band(r[r.split.isin(["val","test"])],  "benchmark val+test (held out, seen lines)"),
    band(ext[ext.cell_line != "SUM159PT"], "extension, seen cell lines"),
    band(ext[ext.cell_line == "SUM159PT"], "extension, SUM159PT (unseen line)"),
]).set_index("subset").round(4).to_string())

print()
print(r[r.status == "empty_after_mask"].cell_line.value_counts().to_string())
