"""
End-to-end pipeline for Die Yield Prediction.
Usage: python scripts/run_pipeline.py [--config config.yaml] [--skip-cv] [--no-plots]
"""
import sys
import os
import argparse
import json
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import load_config, ensure_dirs, save_json, SEED
from src.data_loader import load_dataset, get_feature_cols, get_wafer_ids
from src.spatial_features import compute_spatial_features
from src.block_features import (
    compute_block_features_df, fit_block_pca_from_strings,
    transform_block_pca_from_strings, BLOCK_STAT_NAMES
)
from src.preprocessing import get_eligible_mask, compute_sample_weights, get_die_aggregate_features
from src.models import train_with_cv, train_final_model, predict, save_model
from src.evaluation import compute_metrics, find_best_threshold, print_metrics
from src.interpretability import compute_permutation_importance, category_importance, categorize_features
from src.visualization import (
    plot_feature_importance, plot_category_importance, plot_pr_curve,
    plot_confusion_matrix, plot_wafer_map, plot_model_comparison,
    plot_block_examples_from_arrays, plot_probability_distribution, save_fig
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--input", default=None, help="Validation CSV for prediction only")
    args = parser.parse_args()

    config = load_config(args.config)
    np.random.seed(SEED)

    root = Path(os.path.dirname(__file__)).parent
    input_dir = root / "input"
    model_dir = root / "models"
    out_dir = root / "outputs"
    ensure_dirs(
        model_dir,
        out_dir / "metrics",
        out_dir / "predictions",
        out_dir / "figures",
        out_dir / "reports",
    )

    # ================================================================
    # STAGE 1: Load Data
    # ================================================================
    print("=" * 60)
    print("STAGE 1: Loading data")
    print("=" * 60)

    train_path = input_dir / "train.csv"
    test_path = input_dir / "test.csv"

    if not train_path.exists():
        print(f"ERROR: {train_path} not found. Run generate_data.py first.")
        sys.exit(1)

    train_df, feature_cols, meta_cols = load_dataset(train_path)
    test_df, _, _ = load_dataset(test_path)

    n_features = len(feature_cols)
    print(f"  Train: {len(train_df)} dies, {train_df['wafer_id'].nunique()} wafers")
    print(f"  Test:  {len(test_df)} dies, {test_df['wafer_id'].nunique()} wafers")
    print(f"  Features: {n_features}")

    # Verify no wafer overlap
    train_wafers = set(get_wafer_ids(train_df))
    test_wafers = set(get_wafer_ids(test_df))
    assert len(train_wafers & test_wafers) == 0, "Train/test wafer overlap detected!"

    # Report class balance
    train_eligible = get_eligible_mask(train_df)
    test_eligible = get_eligible_mask(test_df)
    print(f"\n  Train eligible (old_label=0): {train_eligible.sum()}")
    print(f"  Train eligible fail rate: {train_df.loc[train_eligible, 'label'].mean():.4f}")
    print(f"  Test eligible (old_label=0):  {test_eligible.sum()}")
    print(f"  Test eligible fail rate:  {test_df.loc[test_eligible, 'label'].mean():.4f}")

    # ================================================================
    # STAGE 2: Feature Engineering
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 2: Feature Engineering")
    print("=" * 60)

    # 2a. Spatial features (from old_label only)
    print("  Computing spatial features (train)...")
    t0 = time.time()
    train_spatial = compute_spatial_features(train_df, config)
    print(f"    Done in {time.time()-t0:.1f}s")

    print("  Computing spatial features (test)...")
    t0 = time.time()
    test_spatial = compute_spatial_features(test_df, config)
    print(f"    Done in {time.time()-t0:.1f}s")

    spatial_cols = list(train_spatial.columns)

    # 2b. Die aggregate features
    print("  Computing die aggregate features...")
    train_die_agg = get_die_aggregate_features(train_df, feature_cols)
    test_die_agg = get_die_aggregate_features(test_df, feature_cols)
    die_agg_cols = list(train_die_agg.columns)

    # 2c. Block features (Model B only)
    import gc

    print("  Computing block statistics (train)...")
    t0 = time.time()
    train_block_stats = compute_block_features_df(train_df)
    print(f"    Done in {time.time()-t0:.1f}s")

    print("  Computing block statistics (test)...")
    t0 = time.time()
    test_block_stats = compute_block_features_df(test_df)
    print(f"    Done in {time.time()-t0:.1f}s")

    # 2d. Block PCA (fit on train eligible only)
    # Extract block_readings as separate series to allow freeing df memory
    print("  Fitting block PCA on train eligible dies...")
    t0 = time.time()
    train_block_strings = train_df.loc[train_eligible, "block_readings"].values
    n_pca = min(15, len(train_block_strings) // 2)
    block_pca = fit_block_pca_from_strings(train_block_strings, n_components=n_pca, batch_size=2000)
    del train_block_strings
    print(f"    PCA fitted with {n_pca} components in {time.time()-t0:.1f}s")

    print("  Transforming block PCA (train)...")
    train_block_pca = transform_block_pca_from_strings(
        train_df["block_readings"].values, block_pca, index=train_df.index)
    print("  Transforming block PCA (test)...")
    test_block_pca = transform_block_pca_from_strings(
        test_df["block_readings"].values, block_pca, index=test_df.index)
    block_pca_cols = list(train_block_pca.columns)

    # Save block examples before dropping block_readings
    test_elig_for_examples = test_df[test_df["old_label"] == 0]
    block_example_passes = []
    block_example_fails = []
    if "label" in test_df.columns:
        passes_ex = test_elig_for_examples[test_elig_for_examples["label"] == 0]
        fails_ex = test_elig_for_examples[test_elig_for_examples["label"] == 1]
        for i in range(min(3, len(passes_ex))):
            block_example_passes.append(np.fromstring(passes_ex.iloc[i]["block_readings"], dtype=np.float32, sep=' '))
        for i in range(min(3, len(fails_ex))):
            block_example_fails.append(np.fromstring(fails_ex.iloc[i]["block_readings"], dtype=np.float32, sep=' '))
    del test_elig_for_examples

    # Drop block_readings to free ~2GB
    train_df.drop(columns=["block_readings"], inplace=True)
    test_df.drop(columns=["block_readings"], inplace=True)
    gc.collect()
    print("  Freed block_readings memory")

    # Save PCA
    import joblib
    joblib.dump(block_pca, model_dir / "block_pca.pkl")

    # ================================================================
    # Assemble feature matrices
    # ================================================================
    model_a_cols = feature_cols + spatial_cols + die_agg_cols
    model_b_cols = model_a_cols + BLOCK_STAT_NAMES + block_pca_cols

    X_train_a = pd.concat([train_df[feature_cols], train_spatial, train_die_agg], axis=1).values.astype(np.float32)
    X_test_a = pd.concat([test_df[feature_cols], test_spatial, test_die_agg], axis=1).values.astype(np.float32)

    X_train_b = pd.concat([train_df[feature_cols], train_spatial, train_die_agg,
                           train_block_stats, train_block_pca], axis=1).values.astype(np.float32)
    X_test_b = pd.concat([test_df[feature_cols], test_spatial, test_die_agg,
                          test_block_stats, test_block_pca], axis=1).values.astype(np.float32)

    y_train = train_df["label"].values
    y_test = test_df["label"].values
    old_labels_train = train_df["old_label"].values
    old_labels_test = test_df["old_label"].values
    groups_train = train_df["wafer_id"].values

    # Sanity: no NaN/Inf
    for name, arr in [("X_train_a", X_train_a), ("X_test_a", X_test_a),
                      ("X_train_b", X_train_b), ("X_test_b", X_test_b)]:
        assert np.isfinite(arr).all(), f"{name} contains NaN/Inf!"
    print(f"\n  Model A features: {X_train_a.shape[1]}")
    print(f"  Model B features: {X_train_b.shape[1]}")

    # ================================================================
    # STAGE 3: Model A - Cross-Validation
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 3: Model A - Cross-Validation")
    print("=" * 60)

    if not args.skip_cv:
        oof_a, threshold_a, fold_metrics_a = train_with_cv(
            X_train_a, y_train, old_labels_train, groups_train
        )
    else:
        threshold_a = 0.3
        print(f"  Skipping CV, using default threshold={threshold_a}")

    # Train final Model A on all train data
    print("  Training final Model A...")
    model_a = train_final_model(X_train_a, y_train, old_labels_train)
    save_model(model_a, model_dir / "model_a.pkl")

    # Evaluate on test
    probs_a, preds_a = predict(model_a, X_test_a, old_labels_test, threshold_a)
    test_elig = old_labels_test == 0
    metrics_a = compute_metrics(y_test[test_elig], probs_a[test_elig], threshold_a)
    print_metrics(metrics_a, "Model A (Test)")
    save_json(metrics_a, out_dir / "metrics" / "model_a_metrics.json")

    # ================================================================
    # STAGE 4: Model B - Cross-Validation
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 4: Model B - Cross-Validation")
    print("=" * 60)

    if not args.skip_cv:
        oof_b, threshold_b, fold_metrics_b = train_with_cv(
            X_train_b, y_train, old_labels_train, groups_train
        )
    else:
        threshold_b = 0.3
        print(f"  Skipping CV, using default threshold={threshold_b}")

    # Train final Model B
    print("  Training final Model B...")
    model_b = train_final_model(X_train_b, y_train, old_labels_train)
    save_model(model_b, model_dir / "model_b.pkl")

    # Evaluate on test
    probs_b, preds_b = predict(model_b, X_test_b, old_labels_test, threshold_b)
    metrics_b = compute_metrics(y_test[test_elig], probs_b[test_elig], threshold_b)
    print_metrics(metrics_b, "Model B (Test)")
    save_json(metrics_b, out_dir / "metrics" / "model_b_metrics.json")

    # ================================================================
    # STAGE 5: Model Comparison
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 5: Model A vs Model B Comparison")
    print("=" * 60)

    comparison = []
    for metric_name in ["auc_pr", "roc_auc", "fail_f1", "fail_precision",
                        "fail_recall", "pass_f1", "accuracy"]:
        va = metrics_a[metric_name]
        vb = metrics_b[metric_name]
        delta = vb - va
        pct = (delta / va * 100) if va != 0 else 0.0
        comparison.append({
            "metric": metric_name, "model_a": va, "model_b": vb,
            "delta": delta, "pct_change": pct
        })
        print(f"  {metric_name:20s}: A={va:.4f}  B={vb:.4f}  D={delta:+.4f} ({pct:+.1f}%)")

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(out_dir / "metrics" / "model_comparison.csv", index=False)

    # Comparison report
    auc_delta = metrics_b["auc_pr"] - metrics_a["auc_pr"]
    f1_delta = metrics_b["fail_f1"] - metrics_a["fail_f1"]
    if auc_delta > 0.02 and f1_delta > 0.02:
        verdict = "SUBSTANTIAL improvement from block-level data"
    elif auc_delta > 0 or f1_delta > 0:
        verdict = "MARGINAL improvement from block-level data"
    elif auc_delta == 0 and f1_delta == 0:
        verdict = "NO improvement from block-level data"
    else:
        verdict = "Block-level data did NOT improve performance (possible degradation)"

    report = f"""# Model Comparison Report

## Verdict: {verdict}

Adding block-level information changed:
- AUC-PR from {metrics_a['auc_pr']:.4f} to {metrics_b['auc_pr']:.4f} (Delta={auc_delta:+.4f})
- Fail F1 from {metrics_a['fail_f1']:.4f} to {metrics_b['fail_f1']:.4f} (Delta={f1_delta:+.4f})

## Detailed Metrics

| Metric | Model A | Model B | Delta |
|--------|---------|---------|-------|
"""
    for row in comparison:
        report += f"| {row['metric']} | {row['model_a']:.4f} | {row['model_b']:.4f} | {row['delta']:+.4f} |\n"

    report += f"""
## Thresholds
- Model A: {threshold_a:.2f}
- Model B: {threshold_b:.2f}

## Model A Confusion Matrix
|  | Pred Fail | Pred Pass |
|--|-----------|-----------|
| Actual Fail | {metrics_a['tp']} | {metrics_a['fn']} |
| Actual Pass | {metrics_a['fp']} | {metrics_a['tn']} |

## Model B Confusion Matrix
|  | Pred Fail | Pred Pass |
|--|-----------|-----------|
| Actual Fail | {metrics_b['tp']} | {metrics_b['fn']} |
| Actual Pass | {metrics_b['fp']} | {metrics_b['tn']} |
"""
    with open(out_dir / "reports" / "comparison.md", "w") as f:
        f.write(report)

    # ================================================================
    # STAGE 6: Interpretability
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 6: Interpretability")
    print("=" * 60)

    X_test_elig_a = X_test_a[test_elig]
    X_test_elig_b = X_test_b[test_elig]
    y_test_elig = y_test[test_elig]

    print("  Computing Model A permutation importance...")
    imp_a = compute_permutation_importance(model_a, X_test_elig_a, y_test_elig, model_a_cols, n_repeats=5)
    imp_a.to_csv(out_dir / "metrics" / "importance_model_a.csv", index=False)

    print("  Computing Model B permutation importance...")
    imp_b = compute_permutation_importance(model_b, X_test_elig_b, y_test_elig, model_b_cols, n_repeats=5)
    imp_b.to_csv(out_dir / "metrics" / "importance_model_b.csv", index=False)

    cat_a = category_importance(imp_a, model_a_cols)
    cat_b = category_importance(imp_b, model_b_cols)
    print("\n  Model A category importance:")
    for cat, val in cat_a.items():
        print(f"    {cat}: {val:.4f}")
    print("\n  Model B category importance:")
    for cat, val in cat_b.items():
        print(f"    {cat}: {val:.4f}")

    # ================================================================
    # STAGE 7: Visualizations
    # ================================================================
    if not args.no_plots:
        print("\n" + "=" * 60)
        print("STAGE 7: Visualizations")
        print("=" * 60)
        fig_dir = out_dir / "figures"

        plot_feature_importance(imp_a, "Model A - Top 25 Features", fig_dir / "importance_model_a.png")
        plot_feature_importance(imp_b, "Model B - Top 25 Features", fig_dir / "importance_model_b.png")
        plot_category_importance(cat_a, "Model A - Category Importance", fig_dir / "category_importance_a.png")
        plot_category_importance(cat_b, "Model B - Category Importance", fig_dir / "category_importance_b.png")

        plot_pr_curve(y_test_elig, probs_a[test_elig], probs_b[test_elig],
                      fig_dir / "pr_curve_comparison.png")

        plot_confusion_matrix(metrics_a, "Model A Confusion Matrix", fig_dir / "confusion_matrix_a.png")
        plot_confusion_matrix(metrics_b, "Model B Confusion Matrix", fig_dir / "confusion_matrix_b.png")

        plot_model_comparison(metrics_a, metrics_b, fig_dir / "model_comparison.png")

        plot_probability_distribution(y_test_elig, probs_a[test_elig],
                                      "Model A - Prediction Distribution", fig_dir / "prob_dist_a.png")
        plot_probability_distribution(y_test_elig, probs_b[test_elig],
                                      "Model B - Prediction Distribution", fig_dir / "prob_dist_b.png")

        # Wafer maps for first 3 test wafers
        test_wafer_list = sorted(test_df["wafer_id"].unique())[:3]
        test_df_viz = test_df.copy()
        test_df_viz["prob_a"] = probs_a
        test_df_viz["prob_b"] = probs_b
        test_df_viz["pred_a"] = preds_a
        test_df_viz["pred_b"] = preds_b

        for wid in test_wafer_list:
            wdf = test_df_viz[test_df_viz["wafer_id"] == wid]
            plot_wafer_map(wdf, "prob_a", f"Wafer {wid} - Model A",
                           fig_dir / f"wafer_{wid}_model_a.png")
            plot_wafer_map(wdf, "prob_b", f"Wafer {wid} - Model B",
                           fig_dir / f"wafer_{wid}_model_b.png")

        # Block examples (using pre-extracted arrays)
        plot_block_examples_from_arrays(block_example_passes, block_example_fails,
                                         path=fig_dir / "block_examples.png")

        print("  Figures saved.")

    # ================================================================
    # STAGE 8: Generate Predictions
    # ================================================================
    print("\n" + "=" * 60)
    print("STAGE 8: Generating Predictions")
    print("=" * 60)

    # Test predictions
    test_pred_df = test_df[["wafer_id", "die_row", "die_col"]].copy()
    test_pred_df["predicted_label"] = preds_b
    test_pred_df["predicted_prob"] = probs_b
    test_pred_df.to_csv(out_dir / "predictions" / "test_predictions.csv", index=False)

    # Validation predictions (if validation.csv exists)
    val_path = input_dir / "validation.csv"
    if args.input:
        val_path = Path(args.input)

    if val_path.exists():
        print(f"  Generating predictions for {val_path}...")
        val_df, _, _ = load_dataset(val_path, has_label=False)
        val_spatial = compute_spatial_features(val_df, config)
        val_die_agg = get_die_aggregate_features(val_df, feature_cols)
        val_block_stats = compute_block_features_df(val_df)
        val_block_pca_feat = transform_block_pca_from_strings(
            val_df["block_readings"].values, block_pca, index=val_df.index)

        X_val_b = pd.concat([val_df[feature_cols], val_spatial, val_die_agg,
                             val_block_stats, val_block_pca_feat], axis=1).values.astype(np.float32)
        old_labels_val = val_df["old_label"].values

        probs_val, preds_val = predict(model_b, X_val_b, old_labels_val, threshold_b)

        val_pred_df = val_df[["wafer_id", "die_row", "die_col"]].copy()
        val_pred_df["predicted_label"] = preds_val
        val_pred_df.to_csv(out_dir / "predictions" / "validation_predictions.csv", index=False)
        print(f"  Saved validation predictions ({len(val_pred_df)} dies)")

    # Count predictions
    n_test_pred_fail = (preds_b == 1).sum()
    n_test_pred_pass = (preds_b == 0).sum()
    print(f"\n  Test predictions: {n_test_pred_fail} fail, {n_test_pred_pass} pass")
    assert len(preds_b) == len(test_df), "Prediction count mismatch!"

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Models saved to: {model_dir}")
    print(f"  Metrics saved to: {out_dir / 'metrics'}")
    print(f"  Figures saved to: {out_dir / 'figures'}")
    print(f"  Predictions saved to: {out_dir / 'predictions'}")
    print(f"  Report saved to: {out_dir / 'reports'}")


if __name__ == "__main__":
    main()
