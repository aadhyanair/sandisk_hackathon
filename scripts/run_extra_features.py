"""
Orchestrates the five add-on features on top of the already-trained models:

  1. XAI Wafer Intelligence Dashboard   (data exported here; HTML built separately)
  2. Multi-Resolution Information Gain Engine
  3. Marginal Failure Detection w/ Focal Loss (ablation; run with --focal)
  4. Spatial Failure Pattern & Risk-Zone Detection
  5. Engineer Investigation & Triage Engine

Reuses the saved models (models/model_a.pkl, model_b.pkl, block_pca.pkl) and the exact
feature-engineering code from the training pipeline. No model is retrained (except the
optional focal ablation, which is clearly isolated). Outputs go to outputs/analysis/,
outputs/reports/extra_features.md, and outputs/dashboard/dashboard_data.json.

Usage:
  python scripts/run_extra_features.py            # features 1,2,4,5 (fast, test-only)
  python scripts/run_extra_features.py --focal    # also run focal ablation (needs train cache, ~15 min)
"""
import sys, os, json, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from pathlib import Path

from src.utils import save_json
from src.models import load_model
from src.analysis.feature_cache import build_cache, load_block_arrays
from src.analysis import info_gain as IG
from src.analysis import risk_zones as RZ
from src.analysis import investigation as INV
from src.analysis import local_explain as LX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", action="store_true", help="run focal-loss ablation (builds train cache)")
    ap.add_argument("--top-k", type=int, default=25, help="TOP dies to investigate")
    ap.add_argument("--dashboard-wafers", type=int, default=4)
    ap.add_argument("--force-cache", action="store_true")
    args = ap.parse_args()

    root = Path(os.path.dirname(__file__)).parent
    out = root / "outputs"
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "dashboard").mkdir(parents=True, exist_ok=True)

    # ---- Load thresholds + baseline metrics from the completed pipeline ----
    with open(out / "metrics" / "model_a_metrics.json") as f:
        metrics_a = json.load(f)
    with open(out / "metrics" / "model_b_metrics.json") as f:
        metrics_b = json.load(f)
    ta, tb = metrics_a["threshold"], metrics_b["threshold"]

    # ---- Build / load feature cache ----
    print("=" * 60); print("Feature cache"); print("=" * 60)
    splits = ("train", "test") if args.focal else ("test",)
    cache = build_cache(root, splits=splits, force=args.force_cache)

    model_a = load_model(root / "models" / "model_a.pkl")
    model_b = load_model(root / "models" / "model_b.pkl")
    b_cols = cache["model_b_cols"]

    X_test_a = cache["X_test_a"]
    X_test_b = cache["X_test_b"]
    te_meta = cache["test_meta"].reset_index(drop=True)
    old_test = te_meta["old_label"].values.astype(int)
    elig = old_test == 0
    elig_idx = np.where(elig)[0]

    # ---- Predict on eligible dies ----
    pa = model_a.predict_proba(X_test_a[elig])[:, 1]
    pb = model_b.predict_proba(X_test_b[elig])[:, 1]
    meta_elig = te_meta.loc[elig, ["wafer_id", "die_row", "die_col", "label"]].reset_index(drop=True)

    # ================================================================
    # FEATURE 2: Information Gain Engine
    # ================================================================
    print("\n" + "=" * 60); print("FEATURE 2: Information Gain Engine"); print("=" * 60)
    info_df = IG.build_info_gain_table(meta_elig, pa, pb, ta, tb)
    ig_summary = IG.summarize(info_df, metrics_a, metrics_b)
    info_df.to_csv(out / "analysis" / "information_gain_per_die.csv", index=False)
    save_json(ig_summary, out / "analysis" / "information_gain_summary.json")

    # hidden_risk.csv: dies Model A cleared but Model B flagged (block data reveals risk).
    hidden = info_df[info_df["category"] == "HIDDEN_RISK"].copy()
    hidden = hidden.sort_values("prob_delta", ascending=False)
    if "label" in hidden.columns:
        hidden["true_new_failure"] = (hidden["label"] == 1).astype(int)
    hidden.to_csv(out / "analysis" / "hidden_risk.csv", index=False)
    for c in IG.CATEGORIES:
        print(f"  {c:22s}: {ig_summary['counts'][c]:6d}  ({ig_summary['shares'][c]*100:4.1f}%)")
    if ig_summary["hidden_risk_true_failures"] is not None:
        print(f"  HIDDEN_RISK dies that were TRUE failures: {ig_summary['hidden_risk_true_failures']}")

    # ================================================================
    # FEATURE 4: Spatial Risk-Zone Detection
    # ================================================================
    print("\n" + "=" * 60); print("FEATURE 4: Spatial Risk-Zone Detection"); print("=" * 60)
    zones_by_wafer = RZ.detect_all(info_df, tb)
    # die -> zone severity map (aligned to info_df rows)
    sev_map = {}
    pattern_counts = {}
    for wid, z in zones_by_wafer.items():
        pattern_counts[z["pattern"]] = pattern_counts.get(z["pattern"], 0) + 1
        for zone in z["zones"]:
            for (r, c) in zone["cells"]:
                sev_map[(wid, r, c)] = max(sev_map.get((wid, r, c), 0.0), zone["severity"])
    info_df["zone_severity"] = [
        sev_map.get((w, int(r), int(c)), 0.0)
        for w, r, c in zip(info_df["wafer_id"], info_df["die_row"], info_df["die_col"])
    ]
    n_zones_total = sum(len(z["zones"]) for z in zones_by_wafer.values())
    print(f"  Detected {n_zones_total} high-risk zones across {len(zones_by_wafer)} wafers.")
    print(f"  Dominant wafer patterns: " +
          ", ".join(f"{k}={v}" for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])))
    # Persist a compact zone summary
    zone_summary = {}
    for wid, z in zones_by_wafer.items():
        zone_summary[str(wid)] = {
            "pattern": z["pattern"], "stats": z["stats"],
            "zones": [{k: v for k, v in zone.items() if k != "cells"} for zone in z["zones"]],
        }

    # Pre-test spatial signatures over the real WM-811K old_label maps.
    pretest = RZ.classify_pretest_patterns(te_meta)
    pretest_counts = {}
    for wid, p in pretest.items():
        pretest_counts[p["pattern"]] = pretest_counts.get(p["pattern"], 0) + 1
    print("  Pre-test failure-map signatures: " +
          ", ".join(f"{k}={v}" for k, v in sorted(pretest_counts.items(), key=lambda x: -x[1])))
    save_json({"predicted_risk_zones": zone_summary,
               "pretest_signatures": {str(k): v for k, v in pretest.items()},
               "pretest_signature_counts": pretest_counts},
              out / "analysis" / "risk_zones.json")

    # ================================================================
    # FEATURE 5: Investigation Engine (+ local attributions)
    # ================================================================
    print("\n" + "=" * 60); print("FEATURE 5: Investigation Engine"); print("=" * 60)
    ba_idx = b_cols.index("block_anomaly_score")
    block_anom_elig = X_test_b[elig][:, ba_idx]
    ranked = INV.rank_candidates(info_df, block_anom_elig, info_df["zone_severity"].values)

    # Choose dashboard wafers = union of top triage wafers and highest-scoring wafers
    wafer_score = (info_df.assign(hr=(info_df["category"] == "HIDDEN_RISK").astype(int))
                   .groupby("wafer_id")
                   .agg(hidden=("hr", "sum"), sev=("zone_severity", "max"),
                        maxrisk=("prob_b", "max")))
    wafer_score["score"] = wafer_score["hidden"] * 2 + wafer_score["sev"] + wafer_score["maxrisk"]
    
    triage_wafers = list(ranked.head(args.top_k)["wafer_id"].unique())
    score_wafers = list(wafer_score.sort_values("score", ascending=False).index)
    dash_wafers = []
    for wid in triage_wafers + score_wafers:
        if wid not in dash_wafers:
            dash_wafers.append(wid)
        if len(dash_wafers) >= max(args.dashboard_wafers, 8):
            break
    print(f"  Dashboard wafers ({len(dash_wafers)}): {dash_wafers}")

    # Background for local attribution = typical eligible die (median of eligible test set)
    background = LX.compute_background(X_test_b[elig])

    # Rows needing local attribution: top-K investigation + all eligible dies of dash wafers.
    top_keys = set(zip(ranked["wafer_id"].head(max(args.top_k, 50)),
                       ranked["die_row"].head(max(args.top_k, 50)).astype(int),
                       ranked["die_col"].head(max(args.top_k, 50)).astype(int)))
    dash_mask = info_df["wafer_id"].isin(dash_wafers)
    dash_keys = set(zip(info_df.loc[dash_mask, "wafer_id"],
                        info_df.loc[dash_mask, "die_row"].astype(int),
                        info_df.loc[dash_mask, "die_col"].astype(int)))
    need_keys = top_keys | dash_keys

    key_to_eligpos = {(w, int(r), int(c)): i
                      for i, (w, r, c) in enumerate(zip(meta_elig["wafer_id"],
                                                        meta_elig["die_row"], meta_elig["die_col"]))}
    need_pos = sorted({key_to_eligpos[k] for k in need_keys if k in key_to_eligpos})
    print(f"  Computing local attributions for {len(need_pos)} dies...")
    t0 = time.time()
    X_need = X_test_b[elig][need_pos]
    _, attr = LX.local_attributions(model_b, X_need, background)
    top_feats_list = LX.top_contributors(attr, b_cols, k=6)
    pos_to_topfeats = {need_pos[i]: top_feats_list[i] for i in range(len(need_pos))}
    print(f"    Done in {time.time()-t0:.1f}s")

    # Build TOP dies table (map ranked rows -> eligible positions for their top features)
    ranked_pos = [key_to_eligpos.get((w, int(r), int(c)))
                  for w, r, c in zip(ranked["wafer_id"], ranked["die_row"], ranked["die_col"])]
    ranked_topfeats = [pos_to_topfeats.get(p) for p in ranked_pos]
    top_records = INV.build_top_table(ranked, ranked_topfeats, k=args.top_k)
    save_json(top_records, out / "analysis" / "top_dies_to_investigate.json")

    # investigation_priority.csv (full ranked list, human-readable)
    inv_cols = ["wafer_id", "die_row", "die_col", "priority_score", "prob_a", "prob_b",
                "prob_delta", "category", "zone_severity", "block_anomaly_score"]
    inv_df = ranked[inv_cols].copy()
    if "label" in ranked.columns:
        inv_df["actual_label"] = ranked["label"].values
    inv_df.insert(0, "rank", range(1, len(inv_df) + 1))
    inv_df.to_csv(out / "analysis" / "investigation_priority.csv", index=False)
    print(f"  TOP {len(top_records)} dies to investigate saved (+ investigation_priority.csv).")
    for rec in top_records[:5]:
        lab = "" if rec["actual_label"] is None else f" [actual={rec['actual_label']}]"
        print(f"    #{rec['rank']} {rec['wafer_id']} ({rec['die_row']},{rec['die_col']}) "
              f"pB={rec['prob_b']:.3f} {rec['category']}{lab}")

    # ================================================================
    # FEATURE 1: Dashboard data export
    # ================================================================
    print("\n" + "=" * 60); print("FEATURE 1: Dashboard data export"); print("=" * 60)
    dashboard = build_dashboard_data(
        dash_wafers, te_meta, info_df, zones_by_wafer, pos_to_topfeats,
        key_to_eligpos, ta, tb, metrics_a, metrics_b, ig_summary)
    with open(out / "dashboard" / "dashboard_data.json", "w") as f:
        json.dump(dashboard, f)
    print(f"  Exported dashboard data for {len(dash_wafers)} wafers.")

    # ================================================================
    # FEATURE 3: Focal-loss ablation (optional)
    # ================================================================
    focal_verdict = None
    focal_path = out / "analysis" / "focal_ablation.json"
    if args.focal:
        print("\n" + "=" * 60); print("FEATURE 3: Focal-Loss Ablation"); print("=" * 60)
        from src.analysis import focal_ablation as FA
        focal_verdict = FA.run_ablation(cache, metrics_b)
        save_json(focal_verdict, focal_path)
        print(f"  Decision: {focal_verdict['decision']}")
    elif focal_path.exists():
        # Reuse a previously-computed ablation so the report keeps its Feature 3 results.
        with open(focal_path) as f:
            focal_verdict = json.load(f)
        print("  (Focal ablation loaded from previous run.)")

    # ================================================================
    # Consolidated report
    # ================================================================
    write_report(out, ig_summary, zones_by_wafer, pattern_counts, n_zones_total,
                 top_records, dash_wafers, focal_verdict, ta, tb, pretest_counts)
    print("\nAll extra features complete. Report: outputs/reports/extra_features.md")


def build_dashboard_data(dash_wafers, te_meta, info_df, zones_by_wafer, pos_to_topfeats,
                         key_to_eligpos, ta, tb, metrics_a, metrics_b, ig_summary):
    """Compact per-wafer JSON: every die (incl. old fails) with risk band + details."""
    def band(pb, old):
        if old == 1:
            return "known_fail"
        if pb >= 0.5:
            return "critical"
        if pb >= tb:
            return "high"
        if pb >= 0.15:
            return "medium"
        return "low"

    ig_lookup = {(r["wafer_id"], int(r["die_row"]), int(r["die_col"])): r
                 for _, r in info_df.iterrows()}

    wafers = {}
    for wid in dash_wafers:
        wdf = te_meta[te_meta["wafer_id"] == wid]
        dies = []
        for _, r in wdf.iterrows():
            key = (wid, int(r["die_row"]), int(r["die_col"]))
            old = int(r["old_label"])
            rec = ig_lookup.get(key)
            pa_ = float(rec["prob_a"]) if rec is not None else (1.0 if old == 1 else 0.0)
            pb_ = float(rec["prob_b"]) if rec is not None else (1.0 if old == 1 else 0.0)
            cat = rec["category"] if rec is not None else ("known_fail" if old == 1 else None)
            entry = {
                "r": int(r["die_row"]), "c": int(r["die_col"]),
                "old": old, "pa": round(pa_, 4), "pb": round(pb_, 4),
                "band": band(pb_, old), "cat": cat,
            }
            if "label" in r and r["label"] in (0, 1):
                entry["y"] = int(r["label"])
            pos = key_to_eligpos.get(key)
            if pos is not None and pos in pos_to_topfeats:
                entry["tf"] = [[n, round(v, 4)] for n, v in pos_to_topfeats[pos][:6]]
            dies.append(entry)
        z = zones_by_wafer.get(wid, {"pattern": "none", "zones": []})
        wafers[str(wid)] = {
            "pattern": z["pattern"],
            "zones": [{k: v for k, v in zone.items() if k != "cells"} for zone in z["zones"]],
            "dies": dies,
        }
    return {
        "thresholds": {"model_a": ta, "model_b": tb},
        "metrics": {"model_a": metrics_a, "model_b": metrics_b},
        "info_gain_summary": ig_summary,
        "wafers": wafers,
    }


def write_report(out, ig, zones_by_wafer, pattern_counts, n_zones_total,
                 top_records, dash_wafers, focal_verdict, ta, tb, pretest_counts=None):
    L = []
    L.append("# Extra Features Report\n")
    L.append("Add-on analytics built on top of the trained Model A / Model B, reusing the "
             "existing feature pipeline and saved models. All numbers are computed from real "
             "model outputs on the held-out test wafers (eligible dies, old_label==0).\n")

    L.append("\n## Feature 2: Multi-Resolution Information Gain\n")
    L.append("How much did block-level data (Model B) add over die+spatial data (Model A)?\n")
    L.append(f"- Test AUC-PR gain: **{ig['auc_pr_gain']:+.4f}**")
    L.append(f"- Test fail-recall gain: **{ig['fail_recall_gain']:+.4f}**")
    L.append(f"- Mean per-die information gain (|pB - pA|): {ig['mean_information_gain']:.4f}\n")
    L.append("| Category | Count | Share |")
    L.append("|---|---|---|")
    for c in IG_ORDER:
        L.append(f"| {c} | {ig['counts'][c]} | {ig['shares'][c]*100:.1f}% |")
    if ig["hidden_risk_true_failures"] is not None:
        L.append(f"\n**Which dies benefit most from block-level information?** "
                 f"Of the HIDDEN_RISK dies (flagged only by Model B), "
                 f"{ig['hidden_risk_true_failures']} were genuine post-test failures that the "
                 f"die+spatial model alone would have missed.\n")
        if ig["per_category_outcome"]:
            L.append("\nActual failure rate by category:\n")
            L.append("| Category | Count | Actual fail rate | Mean info gain |")
            L.append("|---|---|---|---|")
            for c in IG_ORDER:
                o = ig["per_category_outcome"].get(c)
                if o:
                    L.append(f"| {c} | {o['count']} | {o['actual_fail_rate']*100:.1f}% | {o['mean_information_gain']:.4f} |")

    L.append("\n## Feature 4: Spatial Failure Patterns & Risk Zones\n")
    L.append(f"- Total high-risk zones detected: **{n_zones_total}** across {len(zones_by_wafer)} wafers "
             f"(zone = contiguous cluster of dies with Model B risk >= {tb:.2f}).")
    L.append(f"- Dominant predicted-risk patterns: " +
             ", ".join(f"{k} ({v})" for k, v in sorted(pattern_counts.items(), key=lambda x: -x[1])) + "\n")
    if pretest_counts:
        L.append(f"- Pre-test failure-map signatures (real WM-811K `old_label` geometry, "
                 f"candidate process signatures): " +
                 ", ".join(f"{k} ({v})" for k, v in sorted(pretest_counts.items(), key=lambda x: -x[1])) + "\n")
    # top few zones by severity
    all_zones = []
    for wid, z in zones_by_wafer.items():
        for zone in z["zones"]:
            all_zones.append((wid, z["pattern"], zone))
    all_zones.sort(key=lambda t: t[2]["severity"], reverse=True)
    if all_zones:
        L.append("Top risk zones by severity (size x mean risk):\n")
        L.append("| Wafer | Pattern | Size | Mean risk | Severity | Hidden-risk share |")
        L.append("|---|---|---|---|---|---|")
        for wid, pat, zone in all_zones[:10]:
            hr = zone.get("hidden_risk_fraction")
            hr_s = f"{hr*100:.0f}%" if hr is not None else "-"
            L.append(f"| {wid} | {pat} | {zone['size']} | {zone['mean_risk']:.3f} | "
                     f"{zone['severity']:.2f} | {hr_s} |")
        L.append("\nThe *hidden-risk share* links zones to Feature 2: a high share means that "
                 "region's danger is visible mainly because of block-level data.\n")

    L.append("\n## Feature 5: Top Dies to Investigate\n")
    L.append("Ranked by a priority score fusing predicted risk (0.40), information gain (0.20), "
             "sub-die block anomaly (0.20) and spatial-zone severity (0.20). Evidence is generated "
             "from actual computed values.\n")
    L.append("| Rank | Wafer | (row,col) | Priority | pA | pB | Category | Actual |")
    L.append("|---|---|---|---|---|---|---|---|")
    for rec in top_records:
        lab = "-" if rec["actual_label"] is None else rec["actual_label"]
        L.append(f"| {rec['rank']} | {rec['wafer_id']} | ({rec['die_row']},{rec['die_col']}) | "
                 f"{rec['priority_score']:.3f} | {rec['prob_a']:.3f} | {rec['prob_b']:.3f} | "
                 f"{rec['category']} | {lab} |")
    if top_records:
        L.append("\n**Example investigation evidence (rank 1):**\n")
        for line in top_records[0]["evidence"]:
            L.append(f"- {line}")

    L.append("\n## Feature 1: XAI Wafer Intelligence Dashboard\n")
    L.append(f"Interactive dashboard data exported for wafers {dash_wafers} to "
             f"`outputs/dashboard/dashboard_data.json`, and rendered by "
             f"`scripts/build_dashboard.py` into the self-contained "
             f"`outputs/dashboard/wafer_dashboard.html`. The dashboard shows die-level risk "
             f"maps with hover stats and click-through explanations (Model A vs B probability, "
             f"information-gain category, spatial zone, and occlusion-based local feature drivers).\n")

    L.append("\n## Feature 3: Focal-Loss Ablation\n")
    if focal_verdict is None:
        L.append("_Not run in this pass. Run `python scripts/run_extra_features.py --focal` "
                 "to execute the wafer-grouped focal ablation._\n")
    else:
        b = focal_verdict["baseline"]
        L.append(f"Baseline Model B: fail-F1 {b['fail_f1']:.4f}, AUC-PR {b['auc_pr']:.4f}, "
                 f"recall {b['fail_recall']:.4f}, precision {b['fail_precision']:.4f}\n")
        L.append("| gamma | fail-F1 | AUC-PR | recall | precision |")
        L.append("|---|---|---|---|---|")
        for m in focal_verdict["focal_results"]:
            L.append(f"| {m['gamma']} | {m['fail_f1']:.4f} | {m['auc_pr']:.4f} | "
                     f"{m['fail_recall']:.4f} | {m['fail_precision']:.4f} |")
        L.append(f"\n**Decision: {focal_verdict['decision']}**\n")

    with open(out / "reports" / "extra_features.md", "w") as f:
        f.write("\n".join(L))


IG_ORDER = ["CONFIRMED_RISK", "HIDDEN_RISK", "MODEL_DISAGREEMENT",
            "REDUNDANT_INFORMATION", "LOW_RISK"]


if __name__ == "__main__":
    main()
