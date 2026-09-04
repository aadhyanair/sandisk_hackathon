# Extra Features — Design & Usage

Five add-on capabilities layered on top of the trained Model A / Model B **without
retraining them** (the focal ablation is the one deliberate exception, and it is kept
as an ablation, not the final model). Everything reuses the existing feature pipeline
(`src/spatial_features.py`, `src/block_features.py`, `src/preprocessing.py`) and the
persisted artifacts (`models/model_a.pkl`, `models/model_b.pkl`, `models/block_pca.pkl`).

## How to run

```bash
# Features 1, 2, 4, 5 (fast: builds a test-set feature cache ~100s, then analysis ~1 min)
python scripts/run_extra_features.py

# Render the interactive dashboard (Feature 1) to a self-contained HTML file
python scripts/build_dashboard.py

# Feature 3 focal-loss ablation (also builds the train feature cache; ~30-50 min total)
python scripts/run_extra_features.py --focal
```

Outputs land in `outputs/analysis/`, `outputs/dashboard/`, and `outputs/reports/extra_features.md`.

## Architecture

```
src/analysis/
    feature_cache.py     # rebuild + cache Model A/B matrices from saved block PCA (no retrain)
    local_explain.py     # per-die model-based single-feature occlusion attribution
    info_gain.py         # Feature 2: per-die A-vs-B categorisation + summary
    risk_zones.py        # Feature 4: connected-component zones + pattern classification
    investigation.py     # Feature 5: priority ranking + evidence generation
    focal_ablation.py    # Feature 3: focal-weighted retraining ablation (wafer-grouped CV)
scripts/
    run_extra_features.py  # orchestrates features 1,2,4,5 (+ 3 with --focal)
    build_dashboard.py     # renders Feature 1 dashboard HTML from exported data
```

The cache (`outputs/cache/`) stores the reconstructed feature matrices so the analysis
never has to re-run the 70-minute training pipeline; it is byte-for-byte the same feature
engineering the models were trained on.

---

## Feature 1 — Interactive XAI Wafer Intelligence Dashboard
`outputs/dashboard/wafer_dashboard.html` (self-contained, no server/deps).

A fab-operator console: pick a wafer, see every die coloured by Model B risk band
(critical / high / medium / low / known-fail), hover for quick stats, and click a die to
open its full explanation — Model A vs Model B probability bars (with each model's decision
threshold marked), its information-gain category, the spatial zone it belongs to, and its
**occlusion-based local drivers** as diverging bars (which features pushed failure probability up
vs a typical passing die). A global "top dies to investigate" list threads
wafer → risk zone → die. Data for the four most informative wafers is embedded inline
(the artifact CSP blocks fetching local files).

## Feature 2 — Multi-Resolution Information Gain Engine
`src/analysis/info_gain.py` → `outputs/analysis/information_gain_*.{csv,json}`.

For every eligible die it compares Model A and Model B probabilities and assigns one
category: **CONFIRMED_RISK** (both flag), **HIDDEN_RISK** (only B flags — block data
reveals risk A missed), **MODEL_DISAGREEMENT** (only A flags, or the probabilities differ
strongly), **REDUNDANT_INFORMATION** (agree, block barely moved it), **LOW_RISK** (both
confidently clear). It then answers the two required questions directly: *how much did
Model B improve?* (AUC-PR / recall gains) and *which dies benefit most?* (actual failure
rate per category — e.g. HIDDEN_RISK dies caught **54** true failures the die+spatial model
alone would have passed).

## Feature 3 — Marginal Failure Detection with Focal Loss (ablation)
`src/analysis/focal_ablation.py` → `outputs/analysis/focal_ablation.json`.

`HistGradientBoostingClassifier` takes no custom loss, so focal emphasis is applied the
standard way for a fixed-loss booster: **focal sample weighting**. A warm-start model gives
each training die's probability `p`; the final fit multiplies the class-balancing weight by
the focal modulation `(1 - p_t)^gamma`, up-weighting hard, marginal failures. `gamma = 0`
recovers the exact baseline, so any change is attributable to focal emphasis. It is
evaluated with wafer-grouped CV on minority-class F1, AUC-PR, recall and precision, and is
**adopted only if it demonstrably beats the baseline**; otherwise the baseline stays the
final model and focal is reported as an ablation.

## Feature 4 — Spatial Failure Pattern & Risk-Zone Detection
`src/analysis/risk_zones.py` → `outputs/analysis/risk_zones.json`.

Per wafer, Model B risk is placed on the die grid, thresholded, and grouped into contiguous
**zones** (8-connectivity connected components, minimum size 2). Each zone gets size, mean /
max risk, centroid and a severity score (size × mean risk). The wafer's dominant pattern is
classified from the geometry of its high-risk dies — **center / edge / radial / linear /
clustered / isolated** — using radial position, ring concentration, and PCA elongation. Each
zone also reports its **hidden-risk share**, linking back to Feature 2: a high share means
that region's danger is visible mainly because of block-level data.

## Feature 5 — Engineer Investigation & Triage Engine
`src/analysis/investigation.py` → `outputs/analysis/top_dies_to_investigate.json`.

Ranks eligible dies by an investigation-priority score fusing five real signals —
predicted risk (0.40), information gain (0.20), sub-die block anomaly (0.20) and spatial-zone
severity (0.20), with the strongest local attribution surfaced as supporting evidence. The
"TOP DIES TO INVESTIGATE" records carry evidence lines generated from the actual computed
values (probabilities, category, zone severity, block anomaly score, top local drivers), so
an engineer can move systematically from wafer → risk zone → high-risk die → candidate process signature.

---

## Honesty / anti-fabrication notes
- No die-level number is invented: probabilities come from the saved models, categories and
  zones from those probabilities, local drivers from exact single-feature occlusion of the
  real model, and every "actual" column from the true test labels.
- Local attribution uses the model's own `predict_proba` (no surrogate), against a background
  equal to the median eligible (passing) die.
- The focal study only changes the final model if it actually wins on wafer-grouped CV.
