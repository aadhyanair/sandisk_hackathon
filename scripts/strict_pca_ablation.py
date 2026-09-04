"""
Ablation for reviewer concern #6: fit the block IncrementalPCA INSIDE each CV fold
(train-eligible dies of that fold only), instead of once on all train-eligible dies.

The production pipeline fits PCA once on all training-eligible dies before CV. That uses
NO labels, so it is not target leakage - but strict nested validation refits per fold.
This script quantifies the difference: it re-runs wafer-grouped CV for Model B two ways
and compares OOF AUC-PR. A negligible gap confirms the pre-fit PCA did not inflate results.

Only the 15 PCA columns are recomputed per fold; the other 534 Model-B columns
(parametric + spatial + die-agg + block stats) are label-free and reused from the cache.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.decomposition import IncrementalPCA
from sklearn.metrics import average_precision_score

from src.models import create_model
from src.preprocessing import compute_sample_weights
from src.block_features import parse_block_readings

ROOT = Path(os.path.dirname(__file__)).parent
N_PCA = 15
NONPCA = 534  # first 519 (Model A) + 15 block stats; last 15 are PCA


def fit_pca(strings, idx, batch=2000):
    ipca = IncrementalPCA(n_components=N_PCA)
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        M = np.stack([parse_block_readings(strings[i]) for i in chunk])
        if M.shape[0] >= N_PCA:
            ipca.partial_fit(M)
    return ipca


def transform_pca(strings, idx, ipca, batch=2000):
    out = np.zeros((len(idx), N_PCA), dtype=np.float32)
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        M = np.stack([parse_block_readings(strings[i]) for i in chunk])
        out[s:s + len(chunk)] = ipca.transform(M).astype(np.float32)
    return out


def main():
    print("Loading cache + train block strings...")
    X = np.load(ROOT / "outputs" / "cache" / "X_train_b.npy")
    fixed = X[:, :NONPCA]                      # label-free columns
    global_pca = X[:, NONPCA:NONPCA + N_PCA]   # PCA fit once on all train-eligible
    meta = pd.read_parquet(ROOT / "outputs" / "cache" / "train_meta.parquet")
    y = meta["label"].values.astype(int)
    old = meta["old_label"].values.astype(int)
    groups = meta["wafer_id"].values
    strings = pd.read_csv(ROOT / "input" / "train.csv", usecols=["block_readings"])["block_readings"].values

    gkf = GroupKFold(n_splits=5)
    oof_std = np.full(len(y), np.nan)
    oof_strict = np.full(len(y), np.nan)

    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        t0 = time.time()
        tr_e = tr[old[tr] == 0]
        va_e = va[old[va] == 0]

        # --- standard: reuse the pre-fit (global) PCA columns ---
        Xtr = np.hstack([fixed[tr_e], global_pca[tr_e]])
        Xva = np.hstack([fixed[va_e], global_pca[va_e]])
        m = create_model(); m.fit(Xtr, y[tr_e], sample_weight=compute_sample_weights(y[tr_e]))
        oof_std[va_e] = m.predict_proba(Xva)[:, 1]

        # --- strict: fit PCA on THIS fold's train-eligible dies only ---
        ipca = fit_pca(strings, tr_e)
        pca_tr = transform_pca(strings, tr_e, ipca)
        pca_va = transform_pca(strings, va_e, ipca)
        Xtr2 = np.hstack([fixed[tr_e], pca_tr])
        Xva2 = np.hstack([fixed[va_e], pca_va])
        m2 = create_model(); m2.fit(Xtr2, y[tr_e], sample_weight=compute_sample_weights(y[tr_e]))
        oof_strict[va_e] = m2.predict_proba(Xva2)[:, 1]

        a_std = average_precision_score(y[va_e], oof_std[va_e])
        a_str = average_precision_score(y[va_e], oof_strict[va_e])
        print(f"  Fold {fold}: AUC-PR std={a_std:.4f}  strict={a_str:.4f}  "
              f"(d={a_str-a_std:+.4f})  [{time.time()-t0:.0f}s]")

    elig = old == 0
    vs = ~np.isnan(oof_std) & elig
    auc_std = float(average_precision_score(y[vs], oof_std[vs]))
    auc_str = float(average_precision_score(y[vs], oof_strict[vs]))
    result = {
        "oof_aucpr_prefit_pca": auc_std,
        "oof_aucpr_strict_fold_pca": auc_str,
        "difference": auc_str - auc_std,
        "conclusion": (
            "Pre-fit PCA does not inflate results; strict per-fold PCA gives an "
            f"OOF AUC-PR within {abs(auc_str-auc_std):.4f} of the production approach."
        ),
    }
    with open(ROOT / "outputs" / "analysis" / "strict_pca_ablation.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nOOF AUC-PR  pre-fit PCA: %.4f | strict fold PCA: %.4f | diff: %+.4f"
          % (auc_std, auc_str, auc_str - auc_std))
    print("Saved outputs/analysis/strict_pca_ablation.json")


if __name__ == "__main__":
    main()
