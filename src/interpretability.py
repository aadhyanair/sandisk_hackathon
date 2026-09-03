import numpy as np
from sklearn.inspection import permutation_importance
from .utils import SEED


def compute_permutation_importance(model, X, y, feature_names, n_repeats=10):
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        random_state=SEED,
        scoring="average_precision",
        n_jobs=-1,
    )
    importance_df = {
        "feature": feature_names,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }
    import pandas as pd
    df = pd.DataFrame(importance_df)
    df = df.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return df


def categorize_features(feature_names):
    categories = {}
    for f in feature_names:
        if f.startswith("feature_"):
            categories[f] = "parametric"
        elif f.startswith("block_"):
            categories[f] = "block"
        elif f.startswith("die_feat_"):
            categories[f] = "die_aggregate"
        else:
            categories[f] = "spatial"
    return categories


def category_importance(importance_df, feature_names):
    cats = categorize_features(feature_names)
    importance_df = importance_df.copy()
    importance_df["category"] = importance_df["feature"].map(cats)
    return importance_df.groupby("category")["importance_mean"].sum().sort_values(ascending=False)
