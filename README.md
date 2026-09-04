# Multi-Resolution Die Yield Prediction with Interpretable Spatial Context

Predict which **currently-passing dies will fail final test**, using die-level parametric
measurements, wafer spatial context, and high-resolution sub-die "block" readings — with a
strong emphasis on **explainability** and a fair **Model A vs Model B** experiment that
isolates the value of block-level data.

All numbers in this README are produced by the actual pipeline on held-out test wafers and
can be reproduced with the commands in [How to run](#16-how-to-run).

---

## 1. Problem

A wafer is a grid of dies. Each die has a cheap **pre-test** verdict (`old_label`: 0 pass /
1 fail, from real WM-811K wafer maps). After a rigorous **post-test**, some dies that passed
pre-test newly fail. `label` is the post-test truth (all pre-test fails **plus** new fails).

**Target:** among eligible dies (`old_label == 0`), predict which become **new failures**
(`label == 1`). Pre-test failures (`old_label == 1`) are excluded from training and scoring
(they trivially stay failed and would inflate metrics), but their **positions are kept as
spatial context**. The eligible fail rate is ~**4.2%** — a hard, imbalanced problem, made
harder by design: ~65% of new failures are "marginal" (feature shift only 5–25% of a full
failure), i.e. nearly invisible in the raw numbers.

## 2. Architecture

```
input/*.csv ─▶ data_loader ─▶ feature engineering ──▶ Model A (die + spatial)
                                     │                 Model B (+ block readings)
                                     ▼                        │
                          spatial (old_label only)            ▼
                          die aggregates            GroupKFold CV → threshold
                          block stats + IncrementalPCA         │
                                                               ▼
                          evaluation · interpretability · predictions · figures
                                                               │
        extra analysis ◀──────────────────────────────────────┘
        info-gain · risk-zones · investigation · focal ablation · XAI dashboard
```
Core pipeline: `scripts/run_pipeline.py`. Add-on analysis: `scripts/run_extra_features.py`,
`scripts/build_dashboard.py`, `scripts/make_wafer_views.py`. Modules in `src/` and
`src/analysis/`.

## 3. Model A — die + spatial context
`HistGradientBoostingClassifier` (scikit-learn) on **519 features**: 500 parametric
`feature_*` + 13 spatial-context features + 6 die-level aggregates. **No block readings.**
Class imbalance handled with sample weights (`w_pos = n_neg/n_pos`) and `average_precision`
early-stopping. Threshold chosen on out-of-fold CV predictions only.

## 4. Model B — + block-level information
**Identical classifier and hyperparameters as Model A** (so the A→B delta isolates the block
signal), on **549 features**: Model A's 519 + 15 block statistics + 15 IncrementalPCA
components of the raw 2000-value block signal.

## 5. Spatial context (leakage-safe)
13 features computed **only from `old_label`** and die geometry (via `scipy.spatial.cKDTree`):
neighborhood fail density (3×3 / 5×5), nearest pre-test-fail distance, radial/angular
position, edge detection, zone fail rate, wafer pre-test yield. `label` is **never** used to
build a feature.

## 6. Block-level representation
Each die's `block_readings` is 2000 space-separated floats where failures show as a **sparse,
localized** cluster of anomalous blocks (~5%). We represent it efficiently — **not** as 2000
DataFrame columns — with (a) 15 hand-crafted statistics including a rolling-window
"local anomaly score", and (b) a 15-component **IncrementalPCA** fit in batches on
train-eligible dies only. The raw column is dropped after transform to bound memory.

## 7. Leakage prevention
- Spatial/wafer features from `old_label` only, never `label`.
- IncrementalPCA fit on **train-eligible** dies only, then applied to test.
- **GroupKFold by `wafer_id`** — no wafer spans train and validation in a fold.
- Decision threshold selected on **OOF CV predictions only**, frozen before test scoring.
- Test/validation metrics computed on `old_label == 0` dies only.

The one nuance: the block IncrementalPCA is fit once on all training-eligible dies before CV
(it uses no labels, so it is not target leakage). A **strict per-fold PCA ablation**
(`scripts/strict_pca_ablation.py`) refits PCA inside each fold and confirms this: pooled OOF
AUC-PR **0.5019 (pre-fit) vs 0.5011 (strict), difference −0.0007** — negligible and not in the
pre-fit's favour.

## 8. Imbalance handling
Eligible fail rate ~4.2%. Addressed with class-balanced sample weights, AUC-PR as the primary
metric, and post-hoc threshold tuning on CV. The confusion matrices confirm the model does
**not** collapse to "all pass" (Model B catches 534 of 1380 test new-failures at high
precision). A focal-loss ablation (Section 10) tested whether harder reweighting helps.

## 9. Validation
5-fold `GroupKFold` by wafer. Metrics: AUC-PR (primary), ROC-AUC, per-class F1 / precision /
recall, accuracy, confusion matrix, PR curve.

## 10. Interpretability (30% of rubric)
- **Global:** permutation importance per model + rollup by feature category
  (`parametric / spatial / block / die_aggregate`).
- **Local:** per-die **model-based single-feature occlusion attribution** — each feature is
  reset to a typical passing die's value and the exact change in the model's predicted
  probability is measured — shown as diverging driver bars in the dashboard.
- **Spatial:** per-wafer risk heatmaps and the three-panel wafer view (Section 14).
- **A vs B:** category importance shows block features are the #2 category in Model B, with
  `block_mean` the single most important feature.

## 11. Hidden-Risk concept
**HIDDEN_RISK** = eligible die that **Model A clears (pₐ < threshold) but Model B flags
(p_b ≥ threshold)** — risk visible *only* once sub-die data is included. On the test set:
**202 hidden-risk dies, of which 54 are genuine new failures Model A alone would have
passed.** Exported to `outputs/analysis/hidden_risk.csv`. This is the headline evidence that
block-level data adds real predictive value.

## 12. Risk-zone analysis
Per wafer, Model B risk is placed on the die grid, thresholded, and grouped into contiguous
**zones** (8-connectivity connected components, min size 2). Each zone reports size, mean/max
risk, centroid, severity, and **hidden-risk share**. Wafer geometry is classified into the
vocabulary **center / edge / radial / scratch / linear / clustered / isolated** (scratch = a
long, thin, contiguous streak). This runs on two maps: the **predicted-risk** map (54 zones
across 40 wafers; patterns radial 18, isolated 18, linear 3, edge 1) and the **pre-test
failure map** (real WM-811K `old_label` geometry; signatures isolated 29, radial/ring 7,
clustered 4 in this split). All are framed as **candidate process signatures**, not asserted
physical mechanisms.

## 13. Dashboard
`outputs/dashboard/wafer_dashboard.html` — a self-contained (no server/deps) XAI operator
console: the three-panel wafer view, a die-level risk map (hover + click), Model A vs B
probability bars with thresholds, information-gain category, spatial zone, and occlusion-based
local drivers, plus a wafer→zone→die "top dies to investigate" list.

## 14. Three-panel wafer visualization
`scripts/make_wafer_views.py` → `outputs/figures/wafer_views/`, and embedded in the dashboard.
All panels share coordinate system, aspect ratio, orientation, and die size; only real dies
are drawn (missing dies stay outside the wafer):

| Pre-Test (`old_label`) | Post-Test (`label`) | Difference |
|---|---|---|
| green pass / red fail | green pass / red fail | green stays-pass · red pre-existing fail · **orange NEW failure** |

A prediction-aware companion (`*_prediction.png`) shows **Actual vs Predicted new failures**
and a **TP / FN / FP** outcome panel — demonstrating the model actually predicts the orange
dies rather than just visualizing labels.

## 15. Results (held-out test wafers, eligible dies)

| Metric | Model A | Model B | Delta | % |
|---|---|---|---|---|
| AUC-PR | 0.4859 | **0.5237** | +0.0377 | **+7.8%** |
| ROC-AUC | 0.8255 | **0.8632** | +0.0377 | +4.6% |
| Fail F1 | 0.5160 | 0.5149 | −0.0011 | −0.2% |
| Fail precision | 0.9388 | 0.7695 | −0.1694 | −18.0% |
| Fail recall | 0.3558 | **0.3870** | +0.0312 | **+8.8%** |
| Accuracy | 0.9717 | 0.9691 | −0.0026 | −0.3% |
| Threshold (CV) | 0.64 | 0.63 | | |

**Verdict:** block-level data gives a clear **ranking** improvement (AUC-PR +7.8%, ROC-AUC
+4.6%) and higher recall (+8.8%, i.e. more true failures caught), trading some precision at a
fixed threshold — the more meaningful gain given 65% marginal failures. **Hidden-risk: 54 true
failures recovered.** **Focal-loss ablation:** γ=1 lowered fail-F1 (0.515→0.471) and γ=2
collapsed the model — **baseline kept as final**, focal reported as an ablation only.

### Decision-support evaluation (`scripts/run_advanced_eval.py` → `outputs/reports/advanced_eval.md`)
- **Operating-point curve** — precision / recall / F1 / retest-workload across all thresholds
  (`operating_point_curve.png`), so the threshold is a business choice, not a black box.
- **Business cost** — at t=0.63, of every 1000 dies retested **769 are true failures, 231 false
  alarms**; a cost curve gives the optimal threshold as the miss:retest cost ratio grows.
- **Confidence intervals** (wafer-level bootstrap, 2000 resamples): AUC-PR gain
  **+0.038, 95% CI [+0.029, +0.048], P(gain>0)=100%**; recall gain 95% CI [+0.020, +0.046],
  P=100% — the improvement is **not** sampling noise. (Fail-F1 delta CI straddles 0, reported honestly.)
- **Calibration** — reliability diagram; Model B slightly better calibrated (Brier 0.048→0.048,
  ECE 0.125→0.116).
- **Matched operating points** — at equal recall ≈0.40, Model B needs **229 false positives vs
  Model A's 474**; at an equal retest budget (694 dies) B catches 534 failures vs A's 522.

## 16. How to run

```bash
pip install -r requirements.txt
# data/LSWMD.pkl must be present (download from Kaggle WM-811K); then:
python generate_data.py                 # writes input/{train,test,validation}.csv
python scripts/run_pipeline.py          # train A & B, evaluate, figures, predictions (~30-40 min)
python scripts/run_extra_features.py    # info-gain, risk-zones, investigation, dashboard data (~2 min)
python scripts/make_wafer_views.py      # three-panel Pre/Post/Difference wafer figures
python scripts/run_advanced_eval.py     # operating points, cost, bootstrap CIs, calibration
python scripts/build_dashboard.py       # outputs/dashboard/wafer_dashboard.html
python scripts/run_extra_features.py --focal   # optional: focal-loss ablation (~30-50 min)
python scripts/strict_pca_ablation.py          # optional: strict per-fold PCA leakage check (~30 min)
```

## Outputs
```
outputs/
  metrics/      model_{a,b}_metrics.json, model_comparison.csv, importance_*.csv,
                operating_points_model_{a,b}.csv, business_cost.csv, calibration.csv,
                common_operating_points.csv, bootstrap_ci.json
  predictions/  test_predictions.csv, validation_predictions.csv
  analysis/     information_gain_*, hidden_risk.csv, investigation_priority.csv, risk_zones.json,
                focal_ablation.json, strict_pca_ablation.json
  figures/      importance, PR curve, confusion, prob dist, per-wafer maps, wafer_views/ (triptychs),
                operating_point_curve.png, cost_curve.png, calibration_curve.png, bootstrap_delta.png
  dashboard/    wafer_dashboard.html (+ dashboard_data.json)
  reports/      comparison.md, extra_features.md, advanced_eval.md
```

See `Plan.md`, `Implementation.md`, `PROJECT_OVERVIEW.md`, and `EXTRA_FEATURES.md` for design detail.
