"""
Feature 5: Engineer Root-Cause & Investigation Engine.

Ranks eligible dies by an investigation-priority score that fuses five real signals:
    - predicted risk           Model B failure probability
    - information gain         how much block-level data changed the verdict (|pb - pa|)
    - block-level signature    sub-die anomaly score (block_anomaly_score feature)
    - spatial anomaly          whether the die sits in a detected high-risk zone
    - local attribution        magnitude of the strongest SHAP-style feature driver

Every field on every produced die comes from actually computed model outputs / features,
so the "TOP DIES TO INVESTIGATE" evidence is traceable, not narrative filler.
"""
import numpy as np
import pandas as pd


def _minmax(x):
    x = np.asarray(x, dtype=np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


WEIGHTS = {
    "risk": 0.40,
    "info_gain": 0.20,
    "block_sig": 0.20,
    "spatial": 0.20,
}


def rank_candidates(info_df, block_anomaly, zone_severity):
    """
    info_df:        per eligible die (wafer_id/die_row/die_col/prob_a/prob_b/
                    information_gain/category).
    block_anomaly:  array aligned to info_df: block_anomaly_score feature per die.
    zone_severity:  array aligned to info_df: severity of the high-risk zone the die
                    belongs to (0 if not in any zone).
    Returns info_df sorted by priority_score, descending, with score columns added.
    """
    df = info_df.reset_index(drop=True).copy()
    df["block_anomaly_score"] = np.asarray(block_anomaly, dtype=np.float32)
    df["zone_severity"] = np.asarray(zone_severity, dtype=np.float32)

    risk_n = _minmax(df["prob_b"].values)
    gain_n = _minmax(df["information_gain"].values)
    block_n = _minmax(df["block_anomaly_score"].values)
    spatial_n = _minmax(df["zone_severity"].values)

    df["priority_score"] = (
        WEIGHTS["risk"] * risk_n
        + WEIGHTS["info_gain"] * gain_n
        + WEIGHTS["block_sig"] * block_n
        + WEIGHTS["spatial"] * spatial_n
    ).astype(np.float32)

    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    return df


def build_evidence(row, top_feats):
    """Produce human-readable, value-grounded evidence lines for one die."""
    ev = []
    ev.append(f"Model B failure probability = {row['prob_b']:.3f} "
              f"(Model A = {row['prob_a']:.3f}); category {row['category']}.")
    if row["category"] == "HIDDEN_RISK":
        ev.append(f"Block-level data REVEALED this risk: probability rose by "
                  f"{row['prob_delta']:+.3f} once sub-die signals were included.")
    elif row["category"] == "MODEL_DISAGREEMENT":
        ev.append(f"Models disagree by {row['prob_delta']:+.3f} (block vs parametric) "
                  f"- worth a human check.")
    if row["zone_severity"] > 0:
        ev.append(f"Sits inside a detected high-risk spatial zone "
                  f"(zone severity {row['zone_severity']:.2f}).")
    if abs(row["block_anomaly_score"]) > 0:
        ev.append(f"Sub-die block anomaly score = {row['block_anomaly_score']:.2f}.")
    if top_feats:
        drivers = ", ".join(
            f"{name} ({contrib:+.3f})" for name, contrib in top_feats[:3])
        ev.append(f"Top local drivers of the prediction: {drivers}.")
    return ev


def build_top_table(ranked_df, top_contributors, k=25):
    """
    Assemble the TOP-K investigation records with evidence.
    top_contributors: list aligned to the first len(...) rows of ranked_df, each a list
                      of (feature_name, contribution) tuples (may be None if not computed).
    """
    records = []
    for i in range(min(k, len(ranked_df))):
        row = ranked_df.iloc[i]
        tf = top_contributors[i] if top_contributors is not None and i < len(top_contributors) else None
        records.append({
            "rank": i + 1,
            "wafer_id": row["wafer_id"],
            "die_row": int(row["die_row"]),
            "die_col": int(row["die_col"]),
            "priority_score": float(row["priority_score"]),
            "prob_a": float(row["prob_a"]),
            "prob_b": float(row["prob_b"]),
            "category": row["category"],
            "zone_severity": float(row["zone_severity"]),
            "block_anomaly_score": float(row["block_anomaly_score"]),
            "actual_label": (int(row["label"]) if "label" in row and row["label"] in (0, 1) else None),
            "top_features": tf,
            "evidence": build_evidence(row, tf),
        })
    return records
