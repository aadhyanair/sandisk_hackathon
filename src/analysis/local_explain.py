"""
Per-die local feature attribution (SHAP-style marginal contributions).

Why not the `shap` library: the final models are HistGradientBoostingClassifier,
which shap's fast TreeExplainer does not support, and shap's model-agnostic explainer
is too slow to run over thousands of dies. Instead we use a vectorized single-feature
occlusion attribution, which is the same additive/marginal idea SHAP formalizes:

    contribution_j(x) = p(x) - p(x with feature j reset to a neutral background value)

A positive contribution means "this feature's actual value pushed the failure
probability UP relative to a typical passing die." The background is the per-feature
median of the training-eligible (old_label==0) population, i.e. a representative
"normal passing die." This is exact model output (no surrogate), fully reproducible,
and cheap because each feature is occluded for all dies in a single batched
predict_proba call.
"""
import numpy as np


def compute_background(X_background):
    """Neutral reference row = per-feature median of a background population."""
    return np.median(X_background, axis=0).astype(np.float32)


def local_attributions(model, X_rows, background, batch_features=None):
    """
    Returns (base_prob, attributions) where:
        base_prob        scalar p(background row)
        attributions     (n_rows, n_features) marginal contribution of each feature

    contribution_j = p(x) - p(x_j := background_j)
    """
    X_rows = np.asarray(X_rows, dtype=np.float32)
    n_rows, n_features = X_rows.shape

    base_prob = float(model.predict_proba(background.reshape(1, -1))[0, 1])
    p_full = model.predict_proba(X_rows)[:, 1]  # (n_rows,)

    attributions = np.zeros((n_rows, n_features), dtype=np.float32)
    feats = range(n_features) if batch_features is None else batch_features
    for j in feats:
        Xj = X_rows.copy()
        Xj[:, j] = background[j]
        p_occ = model.predict_proba(Xj)[:, 1]
        attributions[:, j] = (p_full - p_occ).astype(np.float32)
    return base_prob, attributions


def top_contributors(attributions, feature_names, k=5):
    """
    For each row, return a list of (feature_name, contribution) sorted by absolute
    contribution, keeping the sign so direction (risk-increasing / -decreasing) is clear.
    """
    names = np.asarray(feature_names)
    out = []
    order = np.argsort(-np.abs(attributions), axis=1)[:, :k]
    for i in range(attributions.shape[0]):
        idx = order[i]
        out.append([(str(names[j]), float(attributions[i, j])) for j in idx])
    return out
