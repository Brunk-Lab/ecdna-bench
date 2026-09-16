# Draft captions — test-split count heatmaps

Generated 2026-08-31 from Notebook 06, §12.

Matching policy: OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px.

Subset: held-out test split, 175 images (117 of them in a drug condition).

### `figC3_test_by_condition_relative_bias_pct`

**Per-condition counting accuracy on the held-out test split.** Mean ecDNA count per image for each experimental condition represented in the 175-image held-out test split, with the ground-truth mean in the top row and one row per method below. Cell shading encodes count MAE in ecDNA per image. Conditions contributing fewer than 5 test images are not shown; the number of contributing images is given beneath each column label. Method ranking was preserved across every condition with sufficient held-out coverage, but per-condition means on the test split rest on few images and the pooled panels should be read in preference. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_figC3_test_by_condition_relative_bias_pct.csv`.

### `figC3_test_by_cell_line_count_MAE`

**Counting accuracy by cell line on the held-out test split.** Mean ecDNA count per image for each of the four benchmark cell lines on the 175 held-out test images, with the ground-truth mean in the top row. Cell shading encodes count MAE in ecDNA per image. Every held-out image contributes to exactly one column. ecCount (peaks) tracked the true mean burden in all four cell lines, while ecSeg under-counted in every one. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_figC3_test_by_cell_line_count_MAE.csv`.

### `figC3_test_by_arm_count_MAE`

**Counting accuracy under treatment and control on the held-out test split.** Mean ecDNA count per image for treated and control arms, pooled over the 117 held-out images that belong to a drug condition. The large number in each cell is the mean count, the small number the mean signed error against ground truth. Cell shading encodes count MAE in ecDNA per image. A method whose error differs between arms carries that difference into any treatment effect it reports; ecCount (peaks) showed the smallest arm-to-arm change in error. OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_figC3_test_by_arm_count_MAE.csv`.

### `figC3_test_pooled_count_MAE`

**Counting accuracy on the held-out test split, pooled.** Mean ecDNA count per image over all 175 held-out test images. The large number in each cell is the mean count, the small number the mean signed error against ground truth. Cell shading encodes count MAE in ecDNA per image. The average held-out image contains 192 ecDNA; ecCount (peaks) reported 194 (count MAE 13.3, mean signed bias +1.9) and ecSeg 85 (count MAE 110.3, mean signed bias -107.4). OR matching, `d_max = 20 px`, `IoU_min = 0.1`, `alpha = 0.5`, 8-connectivity, minimum object area 3 px. Source data: `source_figC3_test_pooled_count_MAE.csv`.
