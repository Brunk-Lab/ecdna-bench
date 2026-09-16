# Figure captions — test-split panels, 2026-08-31

Draft captions for the panels produced by Notebook 06. Every panel below is
computed on the **held-out 175-image test split**. The corresponding
full-benchmark panels in `release/figures/notebook03/` and
`release/figures/notebook05/` are unchanged and move to supplementary.

Matching policy for all panels: OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px.

---

- `fig6_test_f1_overall.svg` — **Object-level detection accuracy on the held-out test split.** Pooled object-level F1 for six ecDNA detection methods, evaluated on the 175-image held-out test split. Bars are micro-averaged: true positives, false positives and false negatives are summed across images before F1 is computed. ecCount (peaks) achieved the highest object-level F1 at 0.939, compared with 0.508 for ecSeg. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_f1_overall.csv`.

- `fig6_test_f1_by_cell_line.svg` — **Detection accuracy by cell line on the held-out test split.** Pooled object-level F1 for each method within each of the four benchmark cell lines. The number of held-out test images contributing to each cell line is given beneath the axis label. The ranking of methods was preserved across all four cell lines. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_f1_by_cell_line.csv`.

- `fig6_test_f1_by_cell_line_heatmap.svg` — **Detection accuracy by cell line, compact view.** The same pooled object-level F1 values as the grouped-bar panel, presented as a method × cell-line matrix with values annotated. No method exceeded ecCount (peaks) in any cell line. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_f1_by_cell_line.csv`.

- `fig6_test_f1_by_density.svg` — **Detection accuracy as a function of ecDNA density.** Pooled object-level F1 for each method across ground-truth count bins, on the held-out test split. Bins are defined on the ground-truth count per image. The number of test images per bin is given beneath the axis label. Method separation increased with ecDNA density, the regime in which counting accuracy determines the biological conclusion. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_f1_by_density.csv`.

- `fig6_test_count_mae.svg` — **Count accuracy and directional bias on the held-out test split.** Left, mean absolute error between predicted and true ecDNA count per image. Right, mean signed bias, defined as predicted minus true count. Negative bias indicates systematic under-counting. ecCount (peaks) had both the lowest count MAE (13.3) and a mean signed bias closest to zero (+1.9 ecDNA per image); the remaining five methods all under-counted. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_count_mae.csv`.

- `fig6_test_count_agreement.svg` — **Per-image agreement between predicted and true ecDNA count.** Predicted against ground-truth ecDNA count for each of the 175 held-out test images, one panel per method. Axes are logarithmic and counts are displayed with a +1 offset so that images with zero counts remain visible; MAE and bias in each panel title are computed on the unshifted scale. The dashed line is the identity. Points for ecCount (peaks) track the identity line across the full count range, whereas the other methods fall progressively below it as ecDNA burden increases. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_count_agreement.csv`.

- `fig6_test_count_agreement_top3.svg` — **Per-image count agreement for the three highest-F1 methods.** As the six-panel version, restricted to ecCount (peaks), ecCount (threshold mask) and Label Engine for legibility at main-figure size. Only ecCount (peaks) remained centred on the identity line at high ecDNA burden. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_fig6_test_count_agreement_top3.csv`.
