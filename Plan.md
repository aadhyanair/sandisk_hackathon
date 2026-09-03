# Die Yield Prediction — Plan

## 1. Problem Understanding

In semiconductor manufacturing, wafers contain thousands of dies on a grid. Each die undergoes parametric testing (500 measurements) and sub-die block-level testing (2000 readings). Dies can fail due to process defects, proximity to other failures, edge effects, and random mechanisms.

**Two-stage labeling:**
- `old_label`: Pre-test status from WM-811K wafer maps. 0=pass, 1=fail. These are known before prediction.
- `label`: Post-test status. Includes old failures AND newly failed dies. 0=pass, 1=fail.

**Prediction target:** Among dies with `old_label == 0` (pre-test pass), predict which will have `label == 1` (new failure). Dies with `old_label == 1` are already known failures — predicting them would artificially inflate metrics.

**Why old failures matter for context:** Although excluded from evaluation, `old_label == 1` dies provide legitimate spatial context. Their positions on the wafer influence new failure probability (proximity effect).

**Key difficulty:** 65% of all failures are "marginal" — their feature shifts are only 5–25% of the full fail shift, making them nearly indistinguishable from passing dies.

## 2. Repository Assessment

### Available files
- `generate_data.py` — Synthetic data generator on top of WM-811K wafer maps
- `config.yaml` — All generation parameters (500 features, 2000 block readings, 160 train / 40 test wafers, seed 42)
- `requirements.txt` — numpy, pandas, scipy, PyYAML, pyarrow, scikit-learn, matplotlib
- `data/README.txt` — Instructions to download LSWMD.pkl from Kaggle
- `README.md` — Dataset column descriptions and evaluation logic

### Dataset structure (after generation)
- `input/train.csv` — Labeled training data (~160 wafers)
- `input/test.csv` — Labeled test data (~40 wafers)
- `input/validation.csv` — Test data with `label` column removed (submission-style)

### Feature dimensions
- 500 parametric features (`feature_1` ... `feature_500`)
- 2000 block readings per die (space-separated string)
- Identifiers: `wafer_id`, `die_row`, `die_col`
- Labels: `old_label`, `label`

### Missing components
- No `src/`, `models/`, `outputs/` directories
- No model training, evaluation, or visualization code
- LSWMD.pkl must be downloaded from Kaggle

## 3. Data Leakage Analysis

| Risk | Description | Safeguard |
|------|-------------|-----------|
| Spatial features from `label` | Neighborhood fail density computed from target leaks the answer | All spatial features computed from `old_label` only |
| PCA on full dataset | Fitting PCA on train+test contaminates test | Fit IncrementalPCA on train only, transform test |
| Scalers across splits | StandardScaler on all data before CV | Fit scaler inside each CV fold / on train split only |
| Wafer statistics from `label` | Wafer yield from target is direct leakage | Compute wafer stats from `old_label` only |
| Random die-level split | Same-wafer dies in both train and val folds | GroupKFold by `wafer_id` |
| Threshold on test labels | Optimizing threshold on test set | Select threshold on CV OOF predictions only |

## 4. Eligible Population

**Evaluation population:** `old_label == 0` dies only.

**Training:** Train on `old_label == 0` dies. Target = `label`.

**Old failures (old_label == 1):**
- EXCLUDED from training target population
- EXCLUDED from evaluation
- RETAINED for constructing spatial context features (their positions inform neighborhood fail density)
- In submission: emit `predicted_label = 1` (known failures)

## 5. Model A Design

**Algorithm:** `HistGradientBoostingClassifier` (scikit-learn, no extra dependencies)

**Inputs:**
- 500 raw die parametric features
- ~15 spatial context features (see Section 7)
- ~10 die-level aggregate features

**No block_readings.**

**Hyperparameters:**
- max_iter=500, max_depth=6, learning_rate=0.05
- min_samples_leaf=20, l2_regularization=1.0
- early_stopping=True, n_iter_no_change=20
- scoring='average_precision', random_state=42

## 6. Model B Design

**Same algorithm and hyperparameters as Model A.**

**Additional inputs:**
- ~15 block-level statistical features
- 10–20 IncrementalPCA components from block readings

Using the same classifier ensures the A→B comparison isolates the block signal contribution.

## 7. Spatial Feature Engineering

All features use `old_label` and wafer geometry. Never `label`.

| Feature | Description |
|---------|-------------|
| `old_fail_density_5x5` | Fraction of old_label=1 in 5×5 window |
| `old_fail_density_3x3` | Fraction of old_label=1 in 3×3 window |
| `old_fail_count_5x5` | Count of old failures in 5×5 window |
| `valid_neighbor_count_5x5` | Count of valid dies in 5×5 window |
| `nearest_old_fail_dist` | Euclidean distance to closest old failure |
| `radial_position` | Normalized distance from wafer center |
| `angular_position` | Angle from wafer center (atan2) |
| `norm_row`, `norm_col` | Row/col normalized to [-1, 1] |
| `is_edge_die` | Adjacent to wafer boundary |
| `edge_neighbor_count` | Non-die cells in 3×3 window |
| `zone_old_fail_rate` | old_label fail rate in die's zone (4×4 grid) |
| `wafer_old_yield` | Overall old_label pass rate for wafer |
| `directional_fail_counts` | Old failures in N/S/E/W halves of 5×5 window |

Wafer boundaries handled: positions outside the wafer map (wafer_map == 0) are not counted as passing.

## 8. Block Signal Strategy

`block_readings` = 2000 float values as space-separated string. Anomalous blocks are sparse (~5%) and spatially clustered.

**A. Block statistics (vectorized per-wafer):**
- mean, std, skew, kurtosis
- min, max, range
- q05, q25, q50, q75, q95
- q95 − q50 (upper tail spread)
- above_threshold_count (blocks > mean + 2σ)
- max rolling-window(25) mean (captures clustered anomaly)
- anomaly_score = (local_max_mean − global_mean) / global_std
- max_abs_first_diff (derivative spike)

**B. IncrementalPCA (10–20 components):**
- Fit on training block data only
- Process in batches per wafer for memory efficiency
- Transform test/validation with fitted PCA

**Memory management:** Parse block strings to float32 numpy arrays per-wafer. Compute stats vectorized. Discard raw arrays after feature extraction.

## 9. Handling Class Imbalance

Expected positive rate among eligible dies: ~2–5%.

1. **Sample weighting:** `weight_pos = n_neg / n_pos`, applied via `sample_weight` in fit()
2. **AUC-PR as primary metric:** Robust to class imbalance unlike accuracy or ROC-AUC
3. **Threshold tuning:** Maximize fail-class F1 on OOF predictions (not default 0.5)
4. **No SMOTE/oversampling:** Sample weights with gradient boosting are sufficient

## 10. Validation Strategy

**GroupKFold (5 folds) by `wafer_id`** — no wafer appears in both train and validation folds.

Within each fold:
- Filter to eligible dies (old_label == 0)
- Fit scaler on fold-train, transform fold-val
- Fit PCA on fold-train blocks, transform fold-val blocks
- Train model, predict probabilities on fold-val
- Collect OOF predictions for threshold tuning

## 11. Metrics

Computed on `old_label == 0` subset only:

| Metric | Description |
|--------|-------------|
| AUC-PR (Average Precision) | Primary ranking metric |
| ROC-AUC | Secondary ranking metric |
| Fail F1 | At optimized threshold |
| Fail Precision | At optimized threshold |
| Fail Recall | At optimized threshold |
| Pass F1, Precision, Recall | Complementary |
| Overall Accuracy | On eligible dies |
| Confusion Matrix | Full TP/FP/TN/FN |

## 12. Threshold Selection

1. Collect OOF predicted probabilities from GroupKFold CV
2. Sweep thresholds from 0.01 to 0.99 (step 0.01)
3. Compute fail-class F1 at each threshold
4. Select threshold maximizing fail F1
5. Freeze threshold before any test evaluation
6. Apply frozen threshold to test predictions

## 13. Interpretability

### Model A
1. Global feature importance (permutation importance, scoring=average_precision)
2. Top parametric features bar chart
3. Top spatial features bar chart
4. Category-level contribution (parametric vs spatial)
5. Wafer map visualizations: old failures, actual new failures, predicted new failures, risk heatmap

### Model B
Everything above, plus:
1. Block feature importance relative to die/spatial features
2. Category breakdown: parametric vs spatial vs block contribution
3. Block pattern comparison: pass vs fail die block reading plots
4. Important PCA components visualization
5. Example block readings for TP, FP, FN, TN dies

## 14. Model A vs Model B Comparison

| Metric | Model A | Model B | Delta | % Change |
|--------|---------|---------|-------|----------|
| AUC-PR | | | | |
| Fail F1 | | | | |
| Fail Precision | | | | |
| Fail Recall | | | | |
| Accuracy | | | | |
| ROC-AUC | | | | |

Report whether improvement is substantial / marginal / absent / metric-dependent. If Model B is worse, report truthfully and investigate.

## 15. Deliverables

- `Plan.md`, `Implementation.md`
- `src/` — All source modules
- `scripts/run_pipeline.py` — Main pipeline
- `models/` — Saved models, PCA, scalers, thresholds, metadata
- `outputs/metrics/` — model_a_metrics.json, model_b_metrics.json, model_comparison.csv
- `outputs/predictions/` — test_predictions.csv, validation_predictions.csv
- `outputs/figures/` — All visualization PNGs
- `outputs/reports/comparison.md` — Human-readable comparison

## 16. Implementation Sequence

1. Create Plan.md and Implementation.md
2. Write `src/utils.py` — config loading
3. Write `src/data_loader.py` — CSV loading, wafer grid reconstruction
4. Write `src/spatial_features.py` — all spatial features from old_label
5. Write `src/block_features.py` — block parsing, stats, PCA
6. Write `src/preprocessing.py` — eligible filtering, sample weights
7. Write `src/models.py` — training with GroupKFold, threshold tuning
8. Write `src/evaluation.py` — metrics computation
9. Write `src/interpretability.py` — importance, explanations
10. Write `src/visualization.py` — all plots
11. Write `scripts/run_pipeline.py` — end-to-end orchestration
12. Test pipeline
13. Generate outputs
14. Update Implementation.md
