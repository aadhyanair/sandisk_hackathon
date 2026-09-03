"""
Feature 3: Marginal Failure Detection with Focal Loss (ablation study).

HistGradientBoostingClassifier does not accept a custom loss, so we implement the focal
mechanism the way it is standardly applied on top of a fixed-loss booster: focal SAMPLE
WEIGHTING. A warm-start model provides each training die's predicted probability p; we
then down-weight easy, confidently-correct dies and up-weight hard ones with the focal
modulation (1 - p_t)^gamma, multiplied by the existing class-balancing weight. gamma=0
recovers the exact baseline, so any difference is attributable to focal emphasis on hard
(marginal) failures.

This is run as an ABLATION. The baseline remains the final model unless focal
demonstrably improves minority-class detection on wafer-grouped CV.
"""
import numpy as np
from sklearn.model_selection import GroupKFold

from ..models import create_model
from ..preprocessing import compute_sample_weights
from ..evaluation import find_best_threshold, compute_metrics
from ..utils import SEED


def focal_weights(y, p, gamma=2.0):
    """class-balanced weight * (1 - p_t)^gamma, with p_t = p if y==1 else 1-p."""
    base = compute_sample_weights(y)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    p_t = np.where(y == 1, p, 1.0 - p)
    modulation = (1.0 - p_t) ** gamma
    return (base * modulation).astype(np.float32)


def _fit_focal(X_tr, y_tr, gamma):
    """Warm-start to get probabilities, then refit with focal weights."""
    warm = create_model()
    warm.fit(X_tr, y_tr, sample_weight=compute_sample_weights(y_tr))
    p_warm = warm.predict_proba(X_tr)[:, 1]
    w = focal_weights(y_tr, p_warm, gamma=gamma)
    model = create_model()
    model.fit(X_tr, y_tr, sample_weight=w)
    return model


def focal_cv(X, y, old_labels, groups, gamma=2.0, n_splits=5, verbose=True):
    """Wafer-grouped CV for the focal-weighted model. Returns (oof, threshold, folds)."""
    gkf = GroupKFold(n_splits=n_splits)
    oof = np.full(len(y), np.nan)
    folds = []
    for fold, (tr, va) in enumerate(gkf.split(X, y, groups)):
        tr_e = tr[old_labels[tr] == 0]
        va_e = va[old_labels[va] == 0]
        if len(tr_e) == 0 or len(va_e) == 0:
            continue
        model = _fit_focal(X[tr_e], y[tr_e], gamma)
        probs = model.predict_proba(X[va_e])[:, 1]
        oof[va_e] = probs
        t, _ = find_best_threshold(y[va_e], probs)
        m = compute_metrics(y[va_e], probs, threshold=t)
        m["fold"] = fold
        folds.append(m)
        if verbose:
            print(f"    [focal g={gamma}] Fold {fold}: "
                  f"AUC-PR={m['auc_pr']:.4f}, Fail-F1={m['fail_f1']:.4f} (t={t:.2f})")
    elig = old_labels == 0
    valid = ~np.isnan(oof) & elig
    t_best, f1_best = find_best_threshold(y[valid], oof[valid])
    if verbose:
        print(f"    [focal g={gamma}] OOF threshold {t_best:.2f} (F1={f1_best:.4f})")
    return oof, t_best, folds


def run_ablation(cache, metrics_b_baseline, gammas=(1.0, 2.0), verbose=True):
    """
    Full focal ablation using the TRAIN and TEST feature caches for Model B.
    Compares each gamma against the baseline Model B on the held-out test set,
    and returns a verdict dict. Requires cache to contain train + test Model B matrices.
    """
    X_train = cache["X_train_b"]
    X_test = cache["X_test_b"]
    tr_meta = cache["train_meta"]
    te_meta = cache["test_meta"]

    y_train = tr_meta["label"].values.astype(int)
    old_train = tr_meta["old_label"].values.astype(int)
    groups = tr_meta["wafer_id"].values
    y_test = te_meta["label"].values.astype(int)
    old_test = te_meta["old_label"].values.astype(int)
    test_elig = old_test == 0

    results = []
    for g in gammas:
        if verbose:
            print(f"  Focal ablation gamma={g} ...")
        _, t_focal, _ = focal_cv(X_train, y_train, old_train, groups, gamma=g, verbose=verbose)
        model = _fit_focal(X_train[old_train == 0], y_train[old_train == 0], g)
        p_test = model.predict_proba(X_test[test_elig])[:, 1]
        m = compute_metrics(y_test[test_elig], p_test, threshold=t_focal)
        m["gamma"] = g
        results.append(m)

    # Decide honestly whether focal beat the baseline on minority-class detection.
    baseline_f1 = metrics_b_baseline["fail_f1"]
    baseline_aucpr = metrics_b_baseline["auc_pr"]
    baseline_recall = metrics_b_baseline["fail_recall"]

    best = max(results, key=lambda m: (m["fail_f1"], m["auc_pr"]))
    improved = (
        best["fail_f1"] > baseline_f1 + 1e-4
        and best["auc_pr"] >= baseline_aucpr - 1e-3
    )
    verdict = {
        "baseline": {
            "fail_f1": baseline_f1, "auc_pr": baseline_aucpr,
            "fail_recall": baseline_recall,
            "fail_precision": metrics_b_baseline["fail_precision"],
        },
        "focal_results": results,
        "best_gamma": best["gamma"],
        "focal_improves": bool(improved),
        "decision": (
            "ADOPT focal loss (improves minority-class F1 without hurting AUC-PR)"
            if improved else
            "KEEP baseline as final model; focal loss reported as ablation only"
        ),
    }
    return verdict
