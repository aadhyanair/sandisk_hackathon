import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

from .preprocessing import get_eligible_mask, compute_sample_weights
from .evaluation import find_best_threshold, compute_metrics
from .utils import SEED


def create_model():
    return HistGradientBoostingClassifier(
        max_iter=500,
        max_depth=6,
        learning_rate=0.05,
        min_samples_leaf=20,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        scoring="average_precision",
        random_state=SEED,
    )


def train_with_cv(X, y, old_labels, groups, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof_probs = np.full(len(y), np.nan)
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups)):
        train_eligible = train_idx[old_labels[train_idx] == 0]
        val_eligible = val_idx[old_labels[val_idx] == 0]

        if len(train_eligible) == 0 or len(val_eligible) == 0:
            continue

        X_tr, y_tr = X[train_eligible], y[train_eligible]
        X_val, y_val = X[val_eligible], y[val_eligible]

        weights = compute_sample_weights(y_tr)
        model = create_model()
        model.fit(X_tr, y_tr, sample_weight=weights)

        probs = model.predict_proba(X_val)[:, 1]
        oof_probs[val_eligible] = probs

        t, _ = find_best_threshold(y_val, probs)
        m = compute_metrics(y_val, probs, threshold=t)
        m["fold"] = fold
        fold_metrics.append(m)
        print(f"  Fold {fold}: AUC-PR={m['auc_pr']:.4f}, Fail-F1={m['fail_f1']:.4f} (t={t:.2f})")

    eligible_mask = old_labels == 0
    valid_oof = ~np.isnan(oof_probs) & eligible_mask
    best_threshold, best_f1 = find_best_threshold(
        y[valid_oof], oof_probs[valid_oof]
    )
    print(f"  OOF optimal threshold: {best_threshold:.2f} (F1={best_f1:.4f})")

    return oof_probs, best_threshold, fold_metrics


def train_final_model(X, y, old_labels, sample_weight_eligible=True):
    eligible = old_labels == 0
    X_elig, y_elig = X[eligible], y[eligible]
    weights = compute_sample_weights(y_elig) if sample_weight_eligible else None
    model = create_model()
    model.fit(X_elig, y_elig, sample_weight=weights)
    return model


def predict(model, X, old_labels, threshold):
    probs = np.zeros(len(X), dtype=np.float32)
    preds = np.zeros(len(X), dtype=np.int32)

    preds[old_labels == 1] = 1
    probs[old_labels == 1] = 1.0

    eligible = old_labels == 0
    if eligible.sum() > 0:
        p = model.predict_proba(X[eligible])[:, 1]
        probs[eligible] = p.astype(np.float32)
        preds[eligible] = (p >= threshold).astype(np.int32)

    return probs, preds


def save_model(model, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
