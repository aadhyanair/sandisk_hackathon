import pandas as pd
import numpy as np
from pathlib import Path


def load_dataset(path, has_label=True):
    feature_dtype = {}
    # Read header to detect feature columns
    header = pd.read_csv(path, nrows=0).columns.tolist()
    for c in header:
        if c.startswith("feature_"):
            feature_dtype[c] = np.float32
        elif c in ("old_label", "label", "die_row", "die_col"):
            feature_dtype[c] = np.int32

    print(f"  Loading {path}...")
    df = pd.read_csv(path, dtype=feature_dtype)
    feature_cols = sorted([c for c in df.columns if c.startswith("feature_")],
                          key=lambda x: int(x.split("_")[1]))
    meta_cols = ["wafer_id", "die_row", "die_col"]
    print(f"  Loaded {len(df)} rows, {len(feature_cols)} features, "
          f"memory: {df.memory_usage(deep=True).sum() / 1e9:.2f} GB")
    return df, feature_cols, meta_cols


def get_feature_cols(df):
    return sorted([c for c in df.columns if c.startswith("feature_")],
                  key=lambda x: int(x.split("_")[1]))


def reconstruct_wafer_grid(wafer_df, value_col="old_label", fill_value=-1):
    rows = wafer_df["die_row"].values
    cols = wafer_df["die_col"].values
    max_r, max_c = rows.max() + 1, cols.max() + 1
    grid = np.full((max_r, max_c), fill_value, dtype=np.float32)
    grid[rows, cols] = wafer_df[value_col].values
    valid_mask = np.full((max_r, max_c), False)
    valid_mask[rows, cols] = True
    return grid, valid_mask


def get_wafer_ids(df):
    return df["wafer_id"].unique()
