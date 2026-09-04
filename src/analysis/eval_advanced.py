"""
Decision-support evaluation on top of the two trained models:

  1. operating_point_table   precision/recall/F1/FP/retest-workload vs threshold
  2. business_cost           failures caught & false alarms per 1000 retests; cost model
  3. bootstrap_ci            wafer-level bootstrap CIs on A/B metrics and the A->B delta
  4. calibration             reliability (predicted prob vs observed failure rate) + Brier/ECE
  5. common_operating_points compare A vs B at matched recall / FPR / retest budget

All functions take arrays of the ELIGIBLE test dies (old_label==0):
    y   true post-test label (0/1)
    p   model failure probability
    w   wafer_id (for wafer-grouped bootstrap)
Nothing here retrains a model; it reads the saved predictions.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def _counts(y, p, t):
    pred = p >= t
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return tp, fp, fn, tn


def _prf(tp, fp, fn):
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


# ---------- 1. operating points ----------
def operating_point_table(y, p, thresholds=None):
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.99, 0.02), 2)
    n = len(y)
    rows = []
    for t in thresholds:
        tp, fp, fn, tn = _counts(y, p, t)
        prec, rec, f1 = _prf(tp, fp, fn)
        flagged = tp + fp
        rows.append({
            "threshold": float(t), "precision": prec, "recall": rec, "f1": f1,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "flagged_for_retest": flagged,
            "retests_per_1000_dies": 1000.0 * flagged / n,
            "workload_fraction": flagged / n,
        })
    return pd.DataFrame(rows)


# ---------- 2. business cost ----------
def business_cost_at(y, p, t):
    tp, fp, fn, tn = _counts(y, p, t)
    retests = tp + fp
    return {
        "threshold": float(t),
        "flagged_for_retest": retests,
        "failures_caught": tp,
        "failures_missed": fn,
        "false_alarms": fp,
        "failures_caught_per_1000_retests": (1000.0 * tp / retests) if retests else 0.0,
        "false_alarms_per_1000_retests": (1000.0 * fp / retests) if retests else 0.0,
    }


def cost_curve(y, p, miss_to_retest_ratios=(5, 10, 25, 50), thresholds=None):
    """
    Total operational cost vs threshold for several cost ratios.
    Cost unit = 1 retest. Missing a true failure costs `ratio` retests.
        cost(t) = ratio * FN(t) + 1 * (TP(t) + FP(t))
    Returns (per-threshold DataFrame, optimal-threshold dict per ratio).
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.02, 0.99, 0.02), 2)
    n = len(y)
    npos = int(np.sum(y == 1))
    recs = []
    for t in thresholds:
        tp, fp, fn, tn = _counts(y, p, t)
        row = {"threshold": float(t)}
        for r in miss_to_retest_ratios:
            row[f"cost_r{r}"] = r * fn + (tp + fp)
        recs.append(row)
    df = pd.DataFrame(recs)
    optima = {}
    for r in miss_to_retest_ratios:
        col = f"cost_r{r}"
        i = int(df[col].idxmin())
        # reference policies
        retest_all = tp_all = None
        cost_all = r * 0 + n            # flag everything: FN=0, retests=n
        cost_none = r * npos + 0        # flag nothing: FN=npos, retests=0
        optima[r] = {
            "best_threshold": float(df.loc[i, "threshold"]),
            "best_cost": float(df.loc[i, col]),
            "cost_retest_all": float(cost_all),
            "cost_retest_none": float(cost_none),
        }
    return df, optima


# ---------- 3. wafer-level bootstrap CIs ----------
def bootstrap_ci(y, pa, pb, w, ta, tb, n_boot=2000, seed=42):
    """
    Resample WAFERS with replacement; recompute metrics on the pooled dies each time.
    Paired (same resampled wafers for A and B) so the delta CI is meaningful.
    """
    rng = np.random.default_rng(seed)
    wafers = np.unique(w)
    idx_by_wafer = {u: np.where(w == u)[0] for u in wafers}

    def metrics(yy, pa_, pb_):
        out = {}
        if len(np.unique(yy)) > 1:
            out["aucpr_a"] = average_precision_score(yy, pa_)
            out["aucpr_b"] = average_precision_score(yy, pb_)
        else:
            out["aucpr_a"] = out["aucpr_b"] = np.nan
        _, ra, fa = _prf(*_counts(yy, pa_, ta)[:3])
        _, rb, fb = _prf(*_counts(yy, pb_, tb)[:3])
        out["recall_a"], out["f1_a"] = ra, fa
        out["recall_b"], out["f1_b"] = rb, fb
        return out

    keys = ["aucpr_a", "aucpr_b", "recall_a", "recall_b", "f1_a", "f1_b"]
    samples = {k: [] for k in keys}
    d_auc, d_rec, d_f1 = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(wafers, size=len(wafers), replace=True)
        idx = np.concatenate([idx_by_wafer[u] for u in pick])
        m = metrics(y[idx], pa[idx], pb[idx])
        for k in keys:
            samples[k].append(m[k])
        d_auc.append(m["aucpr_b"] - m["aucpr_a"])
        d_rec.append(m["recall_b"] - m["recall_a"])
        d_f1.append(m["f1_b"] - m["f1_a"])

    def ci(a):
        a = np.asarray(a, dtype=float)
        a = a[~np.isnan(a)]
        return {"mean": float(np.mean(a)),
                "lo95": float(np.percentile(a, 2.5)),
                "hi95": float(np.percentile(a, 97.5))}

    return {
        "n_boot": n_boot,
        "point": {k: ci(samples[k]) for k in keys},
        "delta_aucpr": {**ci(d_auc), "p_gt_0": float(np.mean(np.array(d_auc) > 0))},
        "delta_recall": {**ci(d_rec), "p_gt_0": float(np.mean(np.array(d_rec) > 0))},
        "delta_f1": {**ci(d_f1), "p_gt_0": float(np.mean(np.array(d_f1) > 0))},
        "_raw": {"delta_aucpr": d_auc, "delta_recall": d_rec, "delta_f1": d_f1},
    }


# ---------- 4. calibration ----------
def calibration_table(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin_lo": float(lo), "bin_hi": float(hi),
            "mean_predicted": float(p[m].mean()),
            "observed_fail_rate": float(y[m].mean()),
            "count": int(m.sum()),
        })
    df = pd.DataFrame(rows)
    brier = float(brier_score_loss(y, p))
    ece = float(np.sum(df["count"] * np.abs(df["mean_predicted"] - df["observed_fail_rate"])) / len(y))
    return df, brier, ece


# ---------- 5. common operating points ----------
def _threshold_for_recall(y, p, target_recall):
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    best_t, best = 1.0, None
    for t in ts:
        tp, fp, fn, tn = _counts(y, p, t)
        _, rec, _ = _prf(tp, fp, fn)
        if rec >= target_recall:
            best_t = t  # highest threshold still meeting recall (most precise)
    return best_t


def _threshold_for_fpr(y, p, target_fpr):
    ts = np.round(np.arange(0.01, 1.0, 0.01), 2)
    best_t = 0.99
    for t in ts:
        tp, fp, fn, tn = _counts(y, p, t)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        if fpr <= target_fpr:
            best_t = t
            break
    return best_t


def common_operating_points(y, pa, pb, recalls=(0.30, 0.40, 0.50), fprs=(0.01, 0.02, 0.05)):
    rows = []
    for tr in recalls:
        for name, p in [("A", pa), ("B", pb)]:
            t = _threshold_for_recall(y, p, tr)
            tp, fp, fn, tn = _counts(y, p, t)
            prec, rec, f1 = _prf(tp, fp, fn)
            rows.append({"mode": f"recall>={tr:.2f}", "model": name, "threshold": t,
                         "precision": prec, "recall": rec, "f1": f1,
                         "false_positives": fp, "flagged": tp + fp})
    for tf in fprs:
        for name, p in [("A", pa), ("B", pb)]:
            t = _threshold_for_fpr(y, p, tf)
            tp, fp, fn, tn = _counts(y, p, t)
            prec, rec, f1 = _prf(tp, fp, fn)
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            rows.append({"mode": f"FPR<={tf:.2f}", "model": name, "threshold": t,
                         "precision": prec, "recall": rec, "f1": f1,
                         "false_positives": fp, "flagged": tp + fp, "fpr": fpr})
    # same retest budget: flag the top-K dies by probability, K set by B at its own threshold
    return pd.DataFrame(rows)


def same_budget(y, pa, pb, budget):
    """Flag exactly `budget` highest-probability dies for each model; compare catches."""
    out = {}
    for name, p in [("A", pa), ("B", pb)]:
        order = np.argsort(-p)[:budget]
        flagged = np.zeros(len(y), dtype=bool)
        flagged[order] = True
        tp = int(np.sum(flagged & (y == 1)))
        out[name] = {"budget": int(budget), "failures_caught": tp,
                     "recall": float(tp / max(int(np.sum(y == 1)), 1)),
                     "precision": float(tp / max(budget, 1))}
    return out
