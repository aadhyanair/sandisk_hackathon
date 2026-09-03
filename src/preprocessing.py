import numpy as np


def get_eligible_mask(df):
    return df["old_label"].values == 0


def compute_sample_weights(y_eligible):
    n_neg = (y_eligible == 0).sum()
    n_pos = (y_eligible == 1).sum()
    if n_pos == 0:
        return np.ones(len(y_eligible), dtype=np.float32)
    weight_pos = n_neg / n_pos
    weights = np.where(y_eligible == 1, weight_pos, 1.0).astype(np.float32)
    return weights


def get_die_aggregate_features(df, feature_cols):
    import pandas as pd
    feat_data = df[feature_cols].values.astype(np.float32)

    agg = np.zeros((len(df), 6), dtype=np.float32)
    agg[:, 0] = feat_data.mean(axis=1)
    agg[:, 1] = feat_data.std(axis=1)
    agg[:, 2] = np.percentile(feat_data, 25, axis=1)
    agg[:, 3] = np.percentile(feat_data, 50, axis=1)
    agg[:, 4] = np.percentile(feat_data, 75, axis=1)
    agg[:, 5] = feat_data.max(axis=1) - feat_data.min(axis=1)

    cols = ["die_feat_mean", "die_feat_std", "die_feat_q25",
            "die_feat_q50", "die_feat_q75", "die_feat_range"]
    return pd.DataFrame(agg, columns=cols, index=df.index)
