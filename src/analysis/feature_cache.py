"""
Reconstruct and cache the assembled Model A / Model B feature matrices so the
downstream analysis modules (info-gain, risk-zones, investigation, focal ablation,
dashboard) can run WITHOUT re-running the full 70-minute training pipeline.

Everything here reuses the exact same feature functions the training pipeline used
(src.spatial_features, src.block_features, src.preprocessing) and the SAME fitted
block PCA that was persisted to models/block_pca.pkl. No model is retrained and no
feature definition is changed, so the cached matrices are identical to what the
pipeline fed the models.
"""
import os
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ..utils import load_config
from ..data_loader import load_dataset
from ..spatial_features import compute_spatial_features
from ..block_features import (
    compute_block_features_df, transform_block_pca_from_strings, BLOCK_STAT_NAMES,
)
from ..preprocessing import get_die_aggregate_features


META_COLS = ["wafer_id", "die_row", "die_col", "old_label", "label"]


def _assemble(df, feature_cols, config, block_pca):
    """Build Model A and Model B matrices for one split, matching the pipeline order."""
    spatial = compute_spatial_features(df, config)
    die_agg = get_die_aggregate_features(df, feature_cols)
    block_stats = compute_block_features_df(df)
    block_pca_df = transform_block_pca_from_strings(
        df["block_readings"].values, block_pca, index=df.index)

    model_a_cols = list(feature_cols) + list(spatial.columns) + list(die_agg.columns)
    model_b_cols = model_a_cols + list(BLOCK_STAT_NAMES) + list(block_pca_df.columns)

    X_a = pd.concat([df[feature_cols], spatial, die_agg], axis=1).values.astype(np.float32)
    X_b = pd.concat([df[feature_cols], spatial, die_agg, block_stats, block_pca_df],
                    axis=1).values.astype(np.float32)
    return X_a, X_b, model_a_cols, model_b_cols


def build_cache(root, splits=("test",), force=False, verbose=True):
    """
    Build feature cache for the requested splits ("train" and/or "test").

    Writes into outputs/cache/:
        X_<split>_a.npy, X_<split>_b.npy      float32 feature matrices
        <split>_meta.parquet                   wafer_id/die_row/die_col/old_label/label
        model_a_cols.json, model_b_cols.json   feature name lists

    Returns a dict with loaded arrays/metadata for immediate use.
    """
    root = Path(root)
    cache_dir = root / "outputs" / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(root / "config.yaml")
    block_pca = joblib.load(root / "models" / "block_pca.pkl")

    out = {"cache_dir": cache_dir}
    cols_written = False

    for split in splits:
        xa_path = cache_dir / f"X_{split}_a.npy"
        xb_path = cache_dir / f"X_{split}_b.npy"
        meta_path = cache_dir / f"{split}_meta.parquet"

        if not force and xa_path.exists() and xb_path.exists() and meta_path.exists():
            if verbose:
                print(f"  [cache] {split}: already built, loading.")
            out[f"X_{split}_a"] = np.load(xa_path)
            out[f"X_{split}_b"] = np.load(xb_path)
            out[f"{split}_meta"] = pd.read_parquet(meta_path)
            continue

        t0 = time.time()
        if verbose:
            print(f"  [cache] {split}: building features...")
        df, feature_cols, _ = load_dataset(root / "input" / f"{split}.csv")

        # validation.csv has no label column; add a placeholder so meta is uniform
        if "label" not in df.columns:
            df["label"] = -1

        X_a, X_b, a_cols, b_cols = _assemble(df, feature_cols, config, block_pca)
        meta = df[META_COLS].reset_index(drop=True)

        np.save(xa_path, X_a)
        np.save(xb_path, X_b)
        meta.to_parquet(meta_path, index=False)

        if not cols_written:
            with open(cache_dir / "model_a_cols.json", "w") as f:
                json.dump(a_cols, f)
            with open(cache_dir / "model_b_cols.json", "w") as f:
                json.dump(b_cols, f)
            cols_written = True

        out[f"X_{split}_a"] = X_a
        out[f"X_{split}_b"] = X_b
        out[f"{split}_meta"] = meta

        del df
        gc.collect()
        if verbose:
            print(f"  [cache] {split}: done in {time.time()-t0:.1f}s "
                  f"(A={X_a.shape}, B={X_b.shape})")

    with open(cache_dir / "model_a_cols.json") as f:
        out["model_a_cols"] = json.load(f)
    with open(cache_dir / "model_b_cols.json") as f:
        out["model_b_cols"] = json.load(f)
    return out


def load_block_arrays(root, split, row_indices):
    """
    Load raw 2000-length block reading arrays for specific rows of a split, on demand.
    Used by the investigation engine to attach real block-level evidence to flagged
    dies without caching all block arrays. row_indices are positional (0-based) into
    the split CSV in file order.
    """
    root = Path(root)
    path = root / "input" / f"{split}.csv"
    row_indices = np.asarray(sorted(set(int(i) for i in row_indices)))
    if len(row_indices) == 0:
        return {}
    # Read only the block_readings column, then select rows.
    col = pd.read_csv(path, usecols=["block_readings"])["block_readings"].values
    result = {}
    for i in row_indices:
        result[int(i)] = np.fromstring(col[i], dtype=np.float32, sep=" ")
    return result
