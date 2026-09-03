"""
Feature 2: Multi-Resolution Information Gain Engine.

Quantifies what block-level data (Model B) adds over die+spatial data (Model A),
per die, by comparing the two models' failure probabilities. Every die is placed in
exactly one category. All numbers come from the two models' actual predictions on the
same dies; nothing is fabricated.

Categories (evaluated only on eligible dies, old_label == 0):
    CONFIRMED_RISK        both models flag it (A and B >= their thresholds)
    HIDDEN_RISK           only Model B flags it -> block data REVEALED risk A missed
    MODEL_DISAGREEMENT    only Model A flags it, or the two probs differ strongly
    REDUNDANT_INFORMATION both agree, block moved the probability only slightly
    LOW_RISK              both models confidently clear it
"""
import numpy as np
import pandas as pd

BIG_GAP = 0.20     # |pb - pa| considered a strong disagreement
SMALL_GAP = 0.05   # |pb - pa| considered "block added little"
LOW_ABS = 0.10     # both probs below this => confidently low risk

CATEGORIES = [
    "CONFIRMED_RISK", "HIDDEN_RISK", "MODEL_DISAGREEMENT",
    "REDUNDANT_INFORMATION", "LOW_RISK",
]


def categorize(pa, pb, ta, tb):
    """Vectorized category assignment. pa, pb are arrays; ta, tb scalar thresholds."""
    pa = np.asarray(pa, dtype=np.float64)
    pb = np.asarray(pb, dtype=np.float64)
    gap = pb - pa
    a_flag = pa >= ta
    b_flag = pb >= tb

    cat = np.empty(len(pa), dtype=object)
    cat[:] = None

    confirmed = a_flag & b_flag
    hidden = b_flag & ~a_flag
    disagree_a = a_flag & ~b_flag
    both_low = ~a_flag & ~b_flag

    cat[confirmed] = "CONFIRMED_RISK"
    cat[hidden] = "HIDDEN_RISK"
    cat[disagree_a] = "MODEL_DISAGREEMENT"

    # Among dies neither model flags: split by how much block data moved things.
    strong = both_low & (np.abs(gap) >= BIG_GAP)
    low = both_low & (np.abs(gap) < BIG_GAP) & (np.maximum(pa, pb) < LOW_ABS)
    redundant = both_low & (np.abs(gap) < BIG_GAP) & (np.maximum(pa, pb) >= LOW_ABS)

    cat[strong] = "MODEL_DISAGREEMENT"
    cat[low] = "LOW_RISK"
    cat[redundant] = "REDUNDANT_INFORMATION"
    return cat


def build_info_gain_table(meta_elig, pa, pb, ta, tb):
    """
    meta_elig: DataFrame with wafer_id/die_row/die_col (+ label if known) for eligible dies.
    Returns a per-die DataFrame with prob_a, prob_b, information_gain and category.
    """
    df = meta_elig.reset_index(drop=True).copy()
    df["prob_a"] = np.asarray(pa, dtype=np.float32)
    df["prob_b"] = np.asarray(pb, dtype=np.float32)
    df["prob_delta"] = (df["prob_b"] - df["prob_a"]).astype(np.float32)
    df["information_gain"] = np.abs(df["prob_delta"]).astype(np.float32)
    df["category"] = categorize(pa, pb, ta, tb)
    return df


def summarize(df, metrics_a, metrics_b):
    """Build the counts + a plain-language analysis dict answering the doc's questions."""
    counts = {c: int((df["category"] == c).sum()) for c in CATEGORIES}
    total = len(df)
    shares = {c: (counts[c] / total if total else 0.0) for c in CATEGORIES}

    # "How much did Model B improve?" -> from the already-computed test metrics.
    auc_gain = metrics_b["auc_pr"] - metrics_a["auc_pr"]
    recall_gain = metrics_b["fail_recall"] - metrics_a["fail_recall"]

    # "Which dies benefit most?" -> where actual outcome is known, check whether
    # HIDDEN_RISK dies were truly failures (block data caught real, missed failures).
    benefit = {}
    if "label" in df.columns and (df["label"] >= 0).all() and df["label"].nunique() > 1:
        for c in CATEGORIES:
            sub = df[df["category"] == c]
            if len(sub):
                benefit[c] = {
                    "count": int(len(sub)),
                    "actual_fail_rate": float((sub["label"] == 1).mean()),
                    "mean_information_gain": float(sub["information_gain"].mean()),
                }
        hidden = df[df["category"] == "HIDDEN_RISK"]
        hidden_true_fail = int((hidden["label"] == 1).sum()) if len(hidden) else 0
    else:
        hidden_true_fail = None

    return {
        "counts": counts,
        "shares": shares,
        "auc_pr_gain": float(auc_gain),
        "fail_recall_gain": float(recall_gain),
        "hidden_risk_true_failures": hidden_true_fail,
        "per_category_outcome": benefit,
        "mean_information_gain": float(df["information_gain"].mean()),
    }
