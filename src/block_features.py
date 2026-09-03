import numpy as np
import pandas as pd
from scipy import stats


def parse_block_readings(block_str):
    return np.fromstring(block_str, dtype=np.float32, sep=' ')


def compute_block_stats(block_array):
    n = len(block_array)
    if n == 0:
        return np.zeros(15, dtype=np.float32)

    mean = block_array.mean()
    std = block_array.std()
    std_safe = std if std > 1e-8 else 1.0

    q05, q25, q50, q75, q95 = np.percentile(block_array, [5, 25, 50, 75, 95])

    above_2std = np.sum(block_array > mean + 2 * std_safe)

    window = 25
    if n >= window:
        cumsum = np.cumsum(block_array)
        rolling_mean = (cumsum[window-1:] - np.concatenate([[0], cumsum[:-window]])) / window
        local_max_mean = rolling_mean.max()
        local_max_idx = rolling_mean.argmax()
        local_window = block_array[local_max_idx:local_max_idx+window]
        local_max_std = local_window.std()
    else:
        local_max_mean = mean
        local_max_std = std

    anomaly_score = (local_max_mean - mean) / std_safe

    diffs = np.abs(np.diff(block_array))
    max_abs_diff = diffs.max() if len(diffs) > 0 else 0.0

    skewness = float(stats.skew(block_array))
    kurtosis_val = float(stats.kurtosis(block_array))

    return np.array([
        mean, std, skewness, kurtosis_val,
        block_array.min(), block_array.max(), block_array.max() - block_array.min(),
        q05, q25, q50, q75, q95,
        above_2std, anomaly_score, max_abs_diff,
    ], dtype=np.float32)


BLOCK_STAT_NAMES = [
    "block_mean", "block_std", "block_skew", "block_kurtosis",
    "block_min", "block_max", "block_range",
    "block_q05", "block_q25", "block_q50", "block_q75", "block_q95",
    "block_above_2std", "block_anomaly_score", "block_max_abs_diff",
]


def compute_block_features_df(df):
    n = len(df)
    stats_array = np.zeros((n, len(BLOCK_STAT_NAMES)), dtype=np.float32)

    for i in range(n):
        block_str = df.iloc[i]["block_readings"]
        arr = parse_block_readings(block_str)
        stats_array[i] = compute_block_stats(arr)

    return pd.DataFrame(stats_array, columns=BLOCK_STAT_NAMES, index=df.index)


def fit_block_pca_from_strings(block_strings, n_components=15, batch_size=2000):
    from sklearn.decomposition import IncrementalPCA
    ipca = IncrementalPCA(n_components=n_components)

    n = len(block_strings)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.stack([
            parse_block_readings(block_strings[i])
            for i in range(start, end)
        ])
        if batch.shape[0] >= n_components:
            ipca.partial_fit(batch)

    return ipca


def transform_block_pca_from_strings(block_strings, ipca, batch_size=2000, index=None):
    n = len(block_strings)
    n_components = ipca.n_components_
    result = np.zeros((n, n_components), dtype=np.float32)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = np.stack([
            parse_block_readings(block_strings[i])
            for i in range(start, end)
        ])
        result[start:end] = ipca.transform(batch).astype(np.float32)

    cols = [f"block_pca_{i}" for i in range(n_components)]
    return pd.DataFrame(result, columns=cols, index=index)
