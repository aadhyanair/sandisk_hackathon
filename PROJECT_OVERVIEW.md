# Project Overview — Multi-Resolution Die Yield Prediction

This document explains, in plain terms, what this project does, why the data looks the way
it does, how the problem statement maps onto the code, and how the pipeline is built. Read
this before `Plan.md` / `Implementation.md` if you just want the big picture.

---

## 1. What problem are we solving?

In chip manufacturing, a wafer is a large silicon disc cut into a grid of individual chips
called **dies**. Before a wafer is diced up, every die goes through electrical testing.
Historically (in this dataset, via the public **WM-811K** wafer-map dataset) each die already
has a **pre-test verdict** — `old_label` — saying whether it looks good (`0`) or bad (`1`)
based on an earlier/cheaper test pass.

The catch: a die that looked fine on the cheap pre-test can still fail a more rigorous
post-test. Those are **new failures**. `label` is the *ground truth after* the full test:
it equals 1 for every die that was already bad (`old_label==1`) **plus** every die that newly
failed. Manufacturers want to predict, *before* running the expensive/slow final test, which
of the currently-passing dies (`old_label==0`) are actually going to fail it — so they can
prioritize retesting, binning, or scrapping decisions.

**That is the whole task: given a die's test measurements and its position on the wafer,
predict `label` for dies where `old_label==0`.** Dies already marked bad (`old_label==1`)
are not predicted — they are trivially known to stay bad — so they're excluded from both
training targets and evaluation, but their *locations* are still useful information (failures
cluster spatially on a wafer, so a passing die surrounded by pre-test failures is itself more
likely to be a hidden failure).

### Why this is hard
- **Severe class imbalance**: only ~4% of eligible dies newly fail.
- **Marginal failures**: the data generator makes 65% of new failures only "shift" their
  features by 5–25% of the full failure signature — i.e. most failing dies look almost
  identical to passing ones in the raw numbers.
- **Two resolutions of data per die**:
  1. 500 coarse **parametric features** (`feature_1`…`feature_500`) — one row of numbers per die.
  2. 2000 **block readings** — a much higher-resolution sub-die signal (a long space-separated
     string of floats), where failures show up as a small number of spatially-correlated
     anomalous blocks (~5% of blocks) rather than a global shift.
- **No free lunch across wafers**: a good model must generalize to *unseen wafers*, not just
  unseen dies on wafers it has already partly seen — that's why validation splits by
  `wafer_id`, not by row.

---

## 2. Where the data comes from (`generate_data.py`, `config.yaml`)

`generate_data.py` is a synthetic data generator built **on top of real WM-811K wafer maps**
(`data/LSWMD.pkl`, downloaded from Kaggle). It is provided as-is and was not modified.

Roughly, for every wafer map it uses from WM-811K:
1. Takes the real defect map as `old_label` (0/1 per die, on the real die grid layout).
2. Synthesizes 500 parametric features per die from configurable distributions
   (`config.yaml: feature_generation`), with a systematic mean shift for `old_label==1`
   dies (`fail_shift_fraction_range`), plus **radial/linear gradients** across the wafer and
   a **neighborhood-influence** term so features aren't independent noise — they carry real
   spatial structure, the same way real fab data does (edge effects, center-to-edge
   gradients, clustering).
3. Decides which passing dies **newly fail** post-test (`new_fail_rate`, `target_fail_rate`),
   with probability boosted by proximity to existing failures and by edge position — this is
   what produces `label`. 65% of these new failures are deliberately made "marginal"
   (`marginal_fail_fraction`) so their feature shift is tiny.
4. Synthesizes 2000 block readings per die (`num_block_readings`) as a base signal
   (`block_readings.base_mean/base_std`) where, for **some** failing dies, a **sparse cluster**
   of blocks (`anomalous_block_fraction≈5%`) around a random "defect seed" position gets a
   readable shift (`fail_shift_fraction`), smoothed spatially (`block_correlation_kernel`).
   This means the *global* stats of the block array often look almost normal — the signal is
   localized, which is exactly why summary stats *and* a shape-preserving PCA are both used
   downstream (see §5).
5. Splits wafers into `input/train.csv` (160 wafers), `input/test.csv` (40 wafers, includes
   `label`), and `input/validation.csv` (same as test but with `label` stripped — used to
   simulate a blind submission).

Column reference (from `README.md`):

| Column | Meaning |
|---|---|
| `wafer_id`, `die_row`, `die_col` | Wafer + on-wafer grid position |
| `feature_1..feature_500` | Parametric test measurements |
| `block_readings` | 2000 space-separated floats, sub-die signal |
| `old_label` | Pre-test status (0 pass / 1 fail) — **known, usable as a feature/context** |
| `label` | Post-test status (0 pass / 1 fail) — **the prediction target**, only meaningful/scored on `old_label==0` rows |

---

## 3. How the problem statement maps onto the code

The hackathon brief asked for a **leakage-free, wafer-grouped, interpretable** pipeline with
**two models** to isolate the value of block-level (high-resolution) data. Here's the direct
mapping:

| Requirement | Where it's implemented |
|---|---|
| Only train/evaluate on `old_label==0` | [`src/preprocessing.py`](src/preprocessing.py) `get_eligible_mask()`, applied before every fit/score in [`scripts/run_pipeline.py`](scripts/run_pipeline.py) |
| Spatial context from `old_label` only, never `label` | [`src/spatial_features.py`](src/spatial_features.py) — every feature is a function of `old_label` positions via a KD-tree, `label` is never read |
| No leakage across train/test | PCA (`IncrementalPCA`) and any fitted transform is `.fit()` on train only, `.transform()` on test — [`src/block_features.py`](src/block_features.py) |
| Wafer-grouped CV | `GroupKFold(n_splits=5)` keyed on `wafer_id` in [`src/models.py`](src/models.py) `train_with_cv()` |
| Class-imbalance handling | Sample weights (`weight_pos = n_neg/n_pos`) in `preprocessing.compute_sample_weights()`, `average_precision` as the early-stopping/scoring metric, and post-hoc threshold tuning |
| Threshold chosen without touching test labels | `evaluation.find_best_threshold()` runs only on **out-of-fold (OOF)** CV predictions inside `train_with_cv()`; the frozen threshold is then reused (never re-tuned) when scoring test/validation |
| Model A (die + spatial context) vs Model B (+ block data), same classifier | Both call the same `models.create_model()` (`HistGradientBoostingClassifier` with identical hyperparameters) — only the input feature matrix differs, isolating the effect of block data |
| Interpretability (30% of judging) | `src/interpretability.py` (permutation importance, feature-category rollups) + `src/visualization.py` (wafer heatmaps, PR curves, importance bar charts, block-signal example plots) |
| A→B comparison (20%) | Stage 5 of `run_pipeline.py` computes per-metric deltas and writes `outputs/reports/comparison.md` with a verdict |
| Submission format | `models.predict()` forces `predicted_label=1` for `old_label==1` dies and uses the model+threshold only for eligible dies, matching the `wafer_id,die_row,die_col,predicted_label` spec in `README.md` |

---

## 4. Repository layout

```
Hackathon Problem - Die Yield Prediction/
├── generate_data.py         # (given) synthetic data generator on WM-811K
├── config.yaml               # (given) generation parameters
├── README.md                 # (given) data contract & evaluation spec
├── requirements.txt          # (given) numpy/pandas/scipy/sklearn/matplotlib/pyyaml/pyarrow
├── Plan.md                   # Design plan written before implementation
├── Implementation.md         # Log of implementation decisions + results
├── PROJECT_OVERVIEW.md        # This file
├── input/                    # Generated: train.csv, test.csv, validation.csv
├── data/LSWMD.pkl             # Downloaded WM-811K source (Kaggle)
├── src/
│   ├── utils.py               # config loading, paths, JSON helpers, SEED
│   ├── data_loader.py         # CSV loading (float32/int32 dtypes), wafer grid reconstruction
│   ├── spatial_features.py    # old_label-only spatial context features (KD-tree based)
│   ├── block_features.py      # block string parsing, per-die stats, IncrementalPCA
│   ├── preprocessing.py       # eligible mask, sample weights, die-aggregate features
│   ├── models.py               # model factory, CV training, final fit, prediction logic
│   ├── evaluation.py           # metrics (AUC-PR, F1, etc.), threshold search
│   ├── interpretability.py    # permutation importance, feature categorization
│   └── visualization.py        # all figure-generating functions
├── scripts/
│   └── run_pipeline.py        # the single end-to-end entry point (8 stages, see §6)
└── outputs/
    ├── metrics/                # model_a_metrics.json, model_b_metrics.json, comparison.csv
    ├── figures/                 # ~15 PNGs: importance, PR curves, wafer maps, distributions
    ├── predictions/            # test_predictions.csv, validation_predictions.csv
    └── reports/                 # comparison.md (verdict + tables)
```

---

## 5. Feature engineering in detail

### 5.1 Spatial features (`src/spatial_features.py`) — used by **both** models
Computed **only** from `old_label` + die grid coordinates, per wafer, using `scipy.spatial.cKDTree`
for fast neighbor lookups (a naive per-die Python loop was the original implementation and was
too slow — see `Implementation.md` for that fix). 13 features:

- `old_fail_density_5x5`, `old_fail_density_3x3`, `old_fail_count_5x5`, `valid_neighbor_count_5x5`
  — local neighborhood fail concentration at two window sizes
- `nearest_old_fail_dist` — distance to closest pre-test failure (proximity effect)
- `radial_position`, `angular_position`, `norm_row`, `norm_col` — where the die sits on the wafer
  (captures center-vs-edge gradients baked into generation)
- `is_edge_die`, `edge_neighbor_count` — edge dies fail more often by design
- `zone_old_fail_rate` — fail rate within a coarse 4×4 wafer zone
- `wafer_old_yield` — whole-wafer pre-test yield, a global prior per wafer

### 5.2 Die aggregate features (`src/preprocessing.py`)
6 simple summary stats **across the 500 parametric features of the same die**
(`die_feat_mean/std/q25/q50/q75/range`) — cheap signal about whether a die's overall
measurement profile is unusually shifted, independent of which specific feature moved.

### 5.3 Block features (`src/block_features.py`) — Model B only
Because the defect signal in `block_readings` is *sparse and localized* (only ~5% of the
2000 blocks are anomalous, per the generator), two complementary representations are built:

1. **15 hand-crafted statistics** per die: mean, std, skew, kurtosis, min/max/range,
   quantiles (q05/q25/q50/q75/q95), count above 2σ, a rolling-window "local anomaly score"
   (mean of the highest-density 25-block window vs global mean), and max adjacent-block jump.
   These are designed to catch a *localized spike* that a plain mean/std would smear out.
2. **15-component IncrementalPCA** over the raw 2000-length arrays, fit in batches
   (`fit_block_pca_from_strings`) on **train-eligible dies only**, then applied to train and
   test (`transform_block_pca_from_strings`). Incremental/batched fitting was necessary
   because materializing all 2000-float arrays for ~150k train dies at once caused an OOM
   (see `Implementation.md`); batches of 2000 rows keep peak memory bounded.

After PCA is fit and both splits are transformed, the raw `block_readings` string column
(≈2 GB in memory) is dropped and garbage-collected — it's no longer needed.

### Feature matrix sizes
- **Model A**: 500 parametric + 13 spatial + 6 die-aggregate = **519 features**
- **Model B**: Model A's 519 + 15 block stats + 15 PCA components = **549 features**

---

## 6. Modeling & pipeline flow (`scripts/run_pipeline.py`)

Both models use `sklearn.ensemble.HistGradientBoostingClassifier` with identical
hyperparameters (`max_iter=500, max_depth=6, learning_rate=0.05, min_samples_leaf=20,
max_leaf_nodes=31, l2_regularization=1.0, early_stopping=True, scoring='average_precision'`) —
keeping the algorithm and hyperparameters fixed is what makes the Model A → Model B
comparison an isolated test of "does block-level data help," not "did we also change the model."

**8 stages, run end-to-end by `python scripts/run_pipeline.py`:**

1. **Load data** — `data_loader.load_dataset()`, float32/int32 dtypes to control memory.
2. **Feature engineering** — spatial features, die aggregates, block stats, block PCA (as above).
3. **Model A cross-validation** — `models.train_with_cv()`: 5-fold `GroupKFold` by `wafer_id`,
   restricted to eligible (`old_label==0`) rows, sample-weighted for imbalance, OOF predictions
   collected, then `evaluation.find_best_threshold()` sweeps 0.01–0.99 maximizing fail-class F1
   on those OOF predictions only. A final model is then refit on **all** eligible train data at
   the frozen threshold and scored once against the held-out test set.
4. **Model B cross-validation** — identical procedure, on the larger (549-feature) matrix.
5. **Model A vs B comparison** — per-metric deltas (AUC-PR, F1, precision/recall, accuracy),
   verdict logic (substantial/marginal/no/negative improvement), written to
   `outputs/reports/comparison.md`.
6. **Interpretability** — `sklearn.inspection.permutation_importance` (scored on
   `average_precision`) for both models, plus roll-up of importance by feature category
   (parametric / spatial / block / die_aggregate).
7. **Visualizations** — ~15 figures: feature/category importance bar charts, PR curves
   (A vs B overlaid), confusion matrices, per-wafer heatmaps (pre-test map vs post-test map vs
   predicted-probability map), probability distributions (pass vs fail), and example block-signal
   traces for a few pass/fail dies.
8. **Predictions** — `models.predict()` applied to `input/test.csv` and, if present,
   `input/validation.csv`; old failures are force-set to `predicted_label=1`, eligible dies use
   model probability + frozen threshold. Written as `wafer_id,die_row,die_col,predicted_label`
   CSVs per the submission spec.

### Leakage guardrails baked into the flow
- Spatial/PCA/scalers are always fit before touching test data, and never refit on it.
- CV folds never split a wafer across train/validation within a fold (`GroupKFold`).
- The classification threshold is chosen once, on CV OOF predictions, and is never re-tuned
  against test or validation labels.
- Test/validation metrics are computed only on `old_label==0` rows, matching the grading spec.

---

## 7. Results (from the completed pipeline run)

Evaluated on the held-out test wafers, eligible (`old_label==0`) dies only, at each model's
own CV-selected threshold:

| Metric | Model A (die+spatial) | Model B (+block data) | Δ |
|---|---|---|---|
| AUC-PR | 0.4859 | 0.5237 | **+0.0377 (+7.8%)** |
| ROC-AUC | 0.8255 | 0.8632 | **+0.0377 (+4.6%)** |
| Fail F1 | 0.5160 | 0.5149 | −0.0011 (−0.2%) |
| Fail Precision | 0.9388 | 0.7695 | −0.1694 (−18.0%) |
| Fail Recall | 0.3558 | 0.3870 | +0.0312 (+8.8%) |
| Threshold (CV-selected) | 0.64 | 0.63 | — |

**Reading this:** adding block-level data measurably improves the model's ability to *rank*
dies by fail risk (AUC-PR/ROC-AUC both up) and catches more true failures (higher recall), but
at the selected threshold it also produces more false positives, so F1 is roughly flat and
precision drops. Given the marginal-failure design of the dataset (65% of failures are
deliberately near-invisible in raw features), the AUC-PR gain is the more meaningful signal —
it says block data genuinely carries extra separating information, even if a single fixed
threshold doesn't fully cash that in. See `outputs/reports/comparison.md` and
`Implementation.md` for the full write-up and per-fold numbers.

---

## 8. How to reproduce

```bash
pip install -r requirements.txt
python generate_data.py          # needs data/LSWMD.pkl from Kaggle first
python scripts/run_pipeline.py   # full 8-stage pipeline, writes everything under outputs/
```

Expect the full run to take **~30–40 minutes** (block-statistics computation over 2000-length
arrays for ~210k dies and permutation importance are the dominant costs); it was run as a
detached background process during development for exactly this reason.

---

## 9. Add-on features (explainability & engineer tooling)

On top of the two core models, five add-on capabilities were built — an interactive XAI
wafer dashboard, a multi-resolution information-gain engine, a focal-loss ablation, spatial
risk-zone detection, and an engineer investigation/root-cause engine. They reuse the saved
models and the same feature pipeline (no retraining, except the isolated focal ablation).
See **[EXTRA_FEATURES.md](EXTRA_FEATURES.md)** for design and usage, and
`outputs/reports/extra_features.md` for results. Run them with:

```bash
python scripts/run_extra_features.py        # features 1, 2, 4, 5 (fast)
python scripts/build_dashboard.py           # render the dashboard HTML
python scripts/run_extra_features.py --focal  # feature 3 ablation (~30-50 min)
```

Headline result: the information-gain engine shows block-level data (Model B) caught **54
genuine failures the die+spatial model alone would have missed** (the "hidden-risk" dies),
while CONFIRMED_RISK dies truly fail 97.6% of the time.
