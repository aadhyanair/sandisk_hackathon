# Implementation Record

## Status: COMPLETE

Pipeline ran successfully end-to-end. All outputs generated and verified.

## Architecture

```
src/
    __init__.py
    utils.py               # Config loading, paths, JSON helpers, SEED=42
    data_loader.py          # Load CSVs (float32/int32 dtypes), wafer grid reconstruction
    spatial_features.py     # 13 spatial features from old_label via cKDTree
    block_features.py       # Block parsing, 15 stats, IncrementalPCA (batched)
    preprocessing.py        # Eligible mask (old_label==0), sample weights, die aggregates
    models.py               # HistGradientBoosting, GroupKFold CV, threshold tuning, predict
    evaluation.py           # Metrics on eligible dies, threshold sweep
    interpretability.py     # Permutation importance, category rollups
    visualization.py        # All plots (~15 figures)

scripts/
    run_pipeline.py         # 8-stage end-to-end pipeline

models/                     # Saved model artifacts
outputs/
    metrics/                # model_a_metrics.json, model_b_metrics.json, comparison.csv,
                            # importance_model_a.csv, importance_model_b.csv
    predictions/            # test_predictions.csv, validation_predictions.csv
    figures/                # 17 PNGs (importance, PR, confusion, wafer maps, distributions, etc.)
    reports/                # comparison.md
```

## Data Flow

```
input/train.csv (173,099 dies, 160 wafers, 2.60 GB)
input/test.csv  (39,351 dies, 40 wafers, 0.59 GB)
    |
    v
data_loader: load CSVs with float32/int32 dtypes
    |
    v
spatial_features: 13 features from old_label + die coordinates (cKDTree, ~4s)
    |
    v
block_features: parse block strings -> 15 stats (~400s) + 15 PCA components (~340s)
    |
    v  (drop block_readings column to free ~2GB)
    |
    v
preprocessing: filter eligible (old_label==0), compute sample weights
    |
    v
models: 5-fold GroupKFold CV -> OOF predictions -> threshold tuning (max fail F1)
    |
    v
evaluation: metrics on old_label==0 subset only
    |
    v
interpretability: permutation importance (n_jobs=-1), category analysis
    |
    v
visualization: wafer maps, PR curves, comparisons, block examples
    |
    v
outputs: metrics JSON, predictions CSV, figures PNG, comparison report
```

## Dataset Statistics

| Metric | Train | Test |
|--------|-------|------|
| Total dies | 173,099 | 39,351 |
| Wafers | 160 | 40 |
| Features | 500 | 500 |
| Eligible (old_label=0) | 154,037 | 32,598 |
| Eligible fail rate | 4.23% | 4.23% |

## Feature Counts

| Model | Parametric | Spatial | Die Aggregate | Block Stats | Block PCA | Total |
|-------|-----------|---------|---------------|-------------|-----------|-------|
| A | 500 | 13 | 6 | - | - | 519 |
| B | 500 | 13 | 6 | 15 | 15 | 549 |

## Model Configuration

- Algorithm: `HistGradientBoostingClassifier` (identical for A and B)
- max_iter=500, max_depth=6, learning_rate=0.05
- min_samples_leaf=20, max_leaf_nodes=31, l2_regularization=1.0
- early_stopping=True, n_iter_no_change=20, scoring='average_precision'
- Sample weights: weight_pos = n_neg / n_pos
- CV: 5-fold GroupKFold by wafer_id

## Cross-Validation Results

### Model A (Die + Spatial Context)

| Fold | AUC-PR | Fail F1 | Threshold |
|------|--------|---------|-----------|
| 0 | 0.4498 | 0.4957 | 0.65 |
| 1 | 0.4448 | 0.4784 | 0.65 |
| 2 | 0.4922 | 0.5139 | 0.55 |
| 3 | 0.4695 | 0.5061 | 0.63 |
| 4 | 0.4738 | 0.5278 | 0.65 |
| **Mean** | **0.4660** | **0.5044** | |

OOF optimal threshold: **0.64** (F1=0.5011)

### Model B (Die + Spatial + Block)

| Fold | AUC-PR | Fail F1 | Threshold |
|------|--------|---------|-----------|
| 0 | 0.4976 | 0.5068 | 0.65 |
| 1 | 0.4903 | 0.4934 | 0.65 |
| 2 | 0.5184 | 0.5033 | 0.56 |
| 3 | 0.4942 | 0.4928 | 0.64 |
| 4 | 0.5111 | 0.5264 | 0.63 |
| **Mean** | **0.5023** | **0.5045** | |

OOF optimal threshold: **0.63** (F1=0.5011)

## Test Set Results (Eligible Dies Only)

| Metric | Model A | Model B | Delta | % Change |
|--------|---------|---------|-------|----------|
| AUC-PR | 0.4859 | 0.5237 | **+0.0377** | **+7.8%** |
| ROC-AUC | 0.8255 | 0.8632 | **+0.0377** | **+4.6%** |
| Fail F1 | 0.5160 | 0.5149 | -0.0011 | -0.2% |
| Fail Precision | 0.9388 | 0.7695 | -0.1694 | -18.0% |
| Fail Recall | 0.3558 | 0.3870 | +0.0312 | +8.8% |
| Pass F1 | 0.9854 | 0.9841 | -0.0014 | -0.1% |
| Accuracy | 0.9717 | 0.9691 | -0.0026 | -0.3% |
| Threshold | 0.64 | 0.63 | - | - |

### Confusion Matrices

**Model A:**
|  | Pred Fail | Pred Pass |
|--|-----------|-----------|
| Actual Fail | 491 | 889 |
| Actual Pass | 32 | 31,186 |

**Model B:**
|  | Pred Fail | Pred Pass |
|--|-----------|-----------|
| Actual Fail | 534 | 846 |
| Actual Pass | 160 | 31,058 |

## Model Comparison Verdict: MARGINAL Improvement

Adding block-level data:
- **Improved ranking ability**: AUC-PR +7.8%, ROC-AUC +4.6% — the model is meaningfully better at ordering dies by fail risk
- **Improved recall**: +8.8% — catches 43 more true failures (534 vs 491)
- **Lower precision**: -18.0% — at the cost of 128 more false positives (160 vs 32)
- **F1 roughly flat**: the recall gain and precision loss approximately cancel at the chosen threshold
- The AUC-PR improvement is the more meaningful metric here because it measures ranking quality independent of threshold choice. The precision/recall tradeoff is tunable.

## Interpretability Results

### Feature Category Importance (Permutation, AUC-PR)

**Model A:**
| Category | Total Importance |
|----------|-----------------|
| Parametric | 0.1104 |
| Spatial | 0.0034 |
| Die Aggregate | 0.0015 |

**Model B:**
| Category | Total Importance |
|----------|-----------------|
| Parametric | 0.1622 |
| Block | **0.0270** |
| Spatial | 0.0094 |
| Die Aggregate | 0.0016 |

Block features collectively contribute the second-highest category importance in Model B, behind parametric features. Spatial features also become more important when block data is present.

### Top 5 Features

**Model A:** feature_255, feature_480, feature_171, feature_169, feature_419

**Model B:** **block_mean** (0.0195, #1 by far), wafer_old_yield, block_q95, block_q75, feature_444

The `block_mean` feature alone has importance of 0.0195 — nearly as much as all spatial features combined (0.0094). This confirms block-level data carries substantial predictive signal.

## Prediction Outputs

| File | Rows | Format |
|------|------|--------|
| test_predictions.csv | 39,351 | wafer_id, die_row, die_col, predicted_label, predicted_prob |
| validation_predictions.csv | 39,351 | wafer_id, die_row, die_col, predicted_label |

Test predictions: 7,447 fail, 31,904 pass (includes old_label=1 forced to fail).

## Generated Figures (17 PNGs)

- `importance_model_a.png` / `importance_model_b.png` — Top 25 feature importance bars
- `category_importance_a.png` / `category_importance_b.png` — Category-level importance
- `pr_curve_comparison.png` — Precision-Recall curves (A vs B overlaid)
- `confusion_matrix_a.png` / `confusion_matrix_b.png` — Confusion matrices
- `model_comparison.png` — Side-by-side metric bar chart
- `prob_dist_a.png` / `prob_dist_b.png` — Probability distributions (pass vs fail)
- `block_examples.png` — Block reading traces (pass vs new-fail dies)
- `wafer_W_F_*_model_a/b.png` — Per-wafer heatmaps (3 wafers x 2 models = 6 figures)

## Leakage Prevention Verification

- [x] Spatial features computed from old_label only (never label)
- [x] PCA fit on train eligible dies only, transform applied to test
- [x] GroupKFold by wafer_id (no same-wafer data in train+val within a fold)
- [x] Threshold selected on OOF predictions only (never on test labels)
- [x] Sample weights computed on train eligible dies only
- [x] Test/validation metrics computed on old_label==0 subset only
- [x] old_label==1 dies force-set to predicted_label=1 in submission

## Technical Issues Resolved

1. **Spatial features too slow**: Original per-die Python loop replaced with scipy.spatial.cKDTree vectorized approach (3.7s vs minutes)
2. **OOM during PCA**: Switched to IncrementalPCA with batched string-array processing, drop block_readings column after PCA (~2GB freed)
3. **Pipeline timeout**: Run as detached process (nohup) due to >30 min runtime
4. **Unicode encoding error**: Replaced non-ASCII characters (em-dashes, delta symbols) that failed with Windows codepage encoding

## Execution Time (approximate)

| Stage | Time |
|-------|------|
| Data loading | ~2 min |
| Spatial features | ~4s |
| Block statistics | ~400s |
| Block PCA | ~340s |
| Model A CV (5 folds) | ~10 min |
| Model B CV (5 folds) | ~10 min |
| Permutation importance | ~40 min |
| Visualizations | ~1 min |
| **Total** | **~70 min** |
