"""
Feature 4: Spatial Failure Pattern & Risk-Zone Detection.

For each wafer we build a grid of Model B failure probabilities over its eligible dies,
threshold it to a high-risk mask, and find contiguous high-risk regions (connected
components, 8-connectivity). Each zone gets severity metrics. We then classify the
wafer's dominant spatial pattern from the geometry of its high-risk dies, and link each
zone back to the information-gain analysis (what fraction of a zone is HIDDEN_RISK, i.e.
only visible thanks to block-level data).

All quantities are computed from real predicted probabilities and real die coordinates.
"""
import numpy as np
from scipy import ndimage

PATTERNS = ["center", "edge", "radial", "linear", "clustered", "isolated"]


def _classify_pattern(rows, cols, max_r, max_c):
    """Heuristic dominant-pattern classifier for a set of high-risk die coordinates."""
    n = len(rows)
    if n == 0:
        return "none", {}
    center_r, center_c = (max_r - 1) / 2.0, (max_c - 1) / 2.0
    max_rd = np.sqrt(center_r ** 2 + center_c ** 2) or 1.0
    radial = np.sqrt((rows - center_r) ** 2 + (cols - center_c) ** 2) / max_rd
    mean_radial = float(radial.mean())
    std_radial = float(radial.std())

    # Elongation via PCA of coordinates (linear streak detection).
    pts = np.column_stack([rows, cols]).astype(np.float64)
    elong = 0.0
    if n >= 3:
        c = pts - pts.mean(axis=0)
        cov = np.cov(c.T)
        ev = np.sort(np.linalg.eigvalsh(cov))[::-1]
        if ev[0] > 1e-9:
            elong = float(1.0 - ev[1] / ev[0])  # ~1 => line, ~0 => blob

    # Number of separate compact clusters among the high-risk dies.
    n_clusters = _count_clusters(rows, cols, max_r, max_c)

    # Average cluster size = how contiguous the high-risk dies are.
    avg_cluster_size = n / n_clusters if n_clusters else n
    stats = {
        "mean_radial": mean_radial, "std_radial": std_radial,
        "elongation": elong, "n_clusters": int(n_clusters), "n_high_risk": int(n),
        "avg_cluster_size": float(avg_cluster_size),
    }

    # Judge global geometry first; fall back to clustered/isolated by contiguity.
    if n <= 2:
        return "isolated", stats
    if elong > 0.82 and n >= 4:
        return "linear", stats
    if mean_radial < 0.33:
        return "center", stats
    if mean_radial > 0.66:
        return "edge", stats
    if std_radial < 0.16 and n >= 5:
        return "radial", stats  # concentrated at a consistent ring radius
    # Not a clean global shape -> decide by how contiguous the dies are.
    if avg_cluster_size >= 2.5:
        return "clustered", stats
    return "isolated", stats


def _count_clusters(rows, cols, max_r, max_c):
    mask = np.zeros((max_r, max_c), dtype=bool)
    mask[rows, cols] = True
    _, n = ndimage.label(mask, structure=np.ones((3, 3)))
    return n


def detect_wafer_zones(wafer_meta, prob_b, threshold, info_gain_cat=None, min_zone_size=2):
    """
    wafer_meta: DataFrame of ELIGIBLE dies for one wafer with die_row/die_col.
    prob_b:     Model B probabilities aligned to wafer_meta rows.
    info_gain_cat: optional array of info-gain category strings aligned to wafer_meta.

    Returns (zones_list, wafer_pattern, pattern_stats).
    """
    rows = wafer_meta["die_row"].values
    cols = wafer_meta["die_col"].values
    prob_b = np.asarray(prob_b)
    max_r, max_c = int(rows.max()) + 1, int(cols.max()) + 1

    risk_grid = np.full((max_r, max_c), np.nan, dtype=np.float64)
    risk_grid[rows, cols] = prob_b
    high_mask = np.zeros((max_r, max_c), dtype=bool)
    high_mask[rows, cols] = prob_b >= threshold

    labeled, n_zones = ndimage.label(high_mask, structure=np.ones((3, 3)))

    # Map each die back to (row,col) -> index for hidden-risk linkage.
    hidden_grid = np.zeros((max_r, max_c), dtype=bool)
    if info_gain_cat is not None:
        hidden_grid[rows, cols] = np.asarray(info_gain_cat) == "HIDDEN_RISK"

    zones = []
    n_singletons = 0
    for z in range(1, n_zones + 1):
        zr, zc = np.where(labeled == z)
        risks = risk_grid[zr, zc]
        size = int(len(zr))
        if size < min_zone_size:
            n_singletons += 1
            continue
        mean_risk = float(np.nanmean(risks))
        max_risk = float(np.nanmax(risks))
        hidden_frac = float(hidden_grid[zr, zc].mean()) if info_gain_cat is not None else None
        zones.append({
            "zone_id": z,
            "size": size,
            "mean_risk": mean_risk,
            "max_risk": max_risk,
            "severity": float(size * mean_risk),
            "centroid_row": float(zr.mean()),
            "centroid_col": float(zc.mean()),
            "hidden_risk_fraction": hidden_frac,
            "cells": list(zip([int(x) for x in zr], [int(x) for x in zc])),
        })

    zones.sort(key=lambda d: d["severity"], reverse=True)

    hr_rows, hr_cols = np.where(high_mask)
    pattern, stats = _classify_pattern(hr_rows, hr_cols, max_r, max_c)
    stats["n_singleton_dies"] = int(n_singletons)
    return zones, pattern, stats


def detect_all(info_df, threshold):
    """
    info_df: per-die info-gain table (eligible dies) with wafer_id/die_row/die_col/
             prob_b/category.
    Returns dict wafer_id -> {zones, pattern, stats}.
    """
    out = {}
    for wid, g in info_df.groupby("wafer_id"):
        zones, pattern, stats = detect_wafer_zones(
            g, g["prob_b"].values, threshold,
            info_gain_cat=g["category"].values if "category" in g else None)
        out[wid] = {"zones": zones, "pattern": pattern, "stats": stats}
    return out
