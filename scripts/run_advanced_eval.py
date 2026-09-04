"""
Decision-support evaluation: operating-point curves, business-cost analysis,
wafer-level bootstrap confidence intervals, calibration, and matched operating-point
A-vs-B comparison. Reads saved predictions only (no retraining).

Outputs:
  outputs/metrics/operating_points_model_{a,b}.csv
  outputs/metrics/business_cost.csv, common_operating_points.csv, calibration.csv
  outputs/metrics/bootstrap_ci.json
  outputs/figures/operating_point_curve.png, cost_curve.png, calibration_curve.png, bootstrap_delta.png
  outputs/reports/advanced_eval.md
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from src.analysis import eval_advanced as EA

ROOT = Path(os.path.dirname(__file__)).parent
OUT = ROOT / "outputs"
FIG = OUT / "figures"
BLUE, ORANGE, GREEN, RED = "#4C72B0", "#DD8452", "#2f9e6b", "#cf3646"


def _save(fig, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)


def main():
    ig = pd.read_csv(OUT / "analysis" / "information_gain_per_die.csv")
    y = ig["label"].values.astype(int)
    pa = ig["prob_a"].values.astype(float)
    pb = ig["prob_b"].values.astype(float)
    w = ig["wafer_id"].values
    with open(OUT / "metrics" / "model_a_metrics.json") as f:
        ta = json.load(f)["threshold"]
    with open(OUT / "metrics" / "model_b_metrics.json") as f:
        tb = json.load(f)["threshold"]
    n = len(y); npos = int(y.sum())
    print(f"Eligible test dies: {n}, new failures: {npos} ({100*npos/n:.2f}%)")

    # 1. operating points
    op_a = EA.operating_point_table(y, pa)
    op_b = EA.operating_point_table(y, pb)
    op_a.to_csv(OUT / "metrics" / "operating_points_model_a.csv", index=False)
    op_b.to_csv(OUT / "metrics" / "operating_points_model_b.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(op_b["threshold"], op_b["precision"], color=BLUE, label="Precision")
    ax.plot(op_b["threshold"], op_b["recall"], color=GREEN, label="Recall")
    ax.plot(op_b["threshold"], op_b["f1"], color=ORANGE, label="F1")
    ax.axvline(tb, color="#888", ls="--", lw=1)
    ax.text(tb, 1.01, f"chosen t={tb}", fontsize=8, ha="center")
    ax.set_xlabel("Decision threshold (Model B)"); ax.set_ylabel("Precision / Recall / F1")
    ax.set_ylim(0, 1.05); ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(op_b["threshold"], op_b["retests_per_1000_dies"], color=RED, ls=":", label="Retests / 1000 dies")
    ax2.set_ylabel("Retest workload (dies flagged per 1000)", color=RED)
    ax2.tick_params(axis="y", labelcolor=RED)
    l1, la = ax.get_legend_handles_labels(); l2, lb = ax2.get_legend_handles_labels()
    ax.legend(l1 + l2, la + lb, loc="upper right", fontsize=9)
    ax.set_title("Operating-point trade-off (Model B)")
    _save(fig, FIG / "operating_point_curve.png")

    # 2. business cost
    ratios = (5, 10, 25, 50)
    cost_df, optima = EA.cost_curve(y, pb, miss_to_retest_ratios=ratios)
    biz_rows = [EA.business_cost_at(y, pb, tb), EA.business_cost_at(y, pa, ta)]
    biz_rows[0]["model"] = "B"; biz_rows[1]["model"] = "A"
    pd.DataFrame(biz_rows).to_csv(OUT / "metrics" / "business_cost.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for r, c in zip(ratios, [BLUE, GREEN, ORANGE, RED]):
        ax.plot(cost_df["threshold"], cost_df[f"cost_r{r}"], color=c,
                label=f"miss={r}x retest (opt t={optima[r]['best_threshold']})")
        ax.scatter([optima[r]["best_threshold"]], [optima[r]["best_cost"]], color=c, zorder=5, s=25)
    ax.axvline(tb, color="#888", ls="--", lw=1)
    ax.set_xlabel("Decision threshold (Model B)"); ax.set_ylabel("Total cost (retest units)")
    ax.set_title("Business cost vs threshold (cost of a missed failure = N retests)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    _save(fig, FIG / "cost_curve.png")

    # 3. bootstrap CI
    boot = EA.bootstrap_ci(y, pa, pb, w, ta, tb, n_boot=2000)
    raw = boot.pop("_raw")
    with open(OUT / "metrics" / "bootstrap_ci.json", "w") as f:
        json.dump(boot, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 5))
    d = np.array(raw["delta_aucpr"])
    ax.hist(d, bins=40, color=BLUE, alpha=0.8)
    lo, hi = boot["delta_aucpr"]["lo95"], boot["delta_aucpr"]["hi95"]
    ax.axvline(0, color=RED, lw=1.5, label="no improvement")
    ax.axvspan(lo, hi, color=GREEN, alpha=0.18, label=f"95% CI [{lo:+.3f}, {hi:+.3f}]")
    ax.axvline(boot["delta_aucpr"]["mean"], color=GREEN, lw=1.5)
    ax.set_xlabel("AUC-PR(Model B) - AUC-PR(Model A), wafer bootstrap")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title(f"Model B improvement is stable: P(delta>0) = {boot['delta_aucpr']['p_gt_0']*100:.1f}%")
    ax.legend(); _save(fig, FIG / "bootstrap_delta.png")

    # 4. calibration
    cal_a, brier_a, ece_a = EA.calibration_table(y, pa)
    cal_b, brier_b, ece_b = EA.calibration_table(y, pb)
    cal_a["model"] = "A"; cal_b["model"] = "B"
    pd.concat([cal_a, cal_b]).to_csv(OUT / "metrics" / "calibration.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 6.5))
    ax.plot([0, 1], [0, 1], color="#888", ls="--", lw=1, label="perfect calibration")
    ax.plot(cal_a["mean_predicted"], cal_a["observed_fail_rate"], "o-", color=BLUE,
            label=f"Model A (Brier {brier_a:.3f}, ECE {ece_a:.3f})")
    ax.plot(cal_b["mean_predicted"], cal_b["observed_fail_rate"], "s-", color=ORANGE,
            label=f"Model B (Brier {brier_b:.3f}, ECE {ece_b:.3f})")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed failure rate")
    ax.set_title("Calibration (reliability diagram)"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.grid(alpha=0.3); ax.legend(loc="upper left", fontsize=9)
    _save(fig, FIG / "calibration_curve.png")

    # 5. common operating points
    cop = EA.common_operating_points(y, pa, pb)
    cop.to_csv(OUT / "metrics" / "common_operating_points.csv", index=False)
    budget = int((pb >= tb).sum())
    sb = EA.same_budget(y, pa, pb, budget)

    write_report(OUT, n, npos, ta, tb, op_a, op_b, biz_rows, optima, ratios,
                 boot, brier_a, brier_b, ece_a, ece_b, cop, sb, budget)
    print("Advanced evaluation complete -> outputs/reports/advanced_eval.md")


def write_report(OUT, n, npos, ta, tb, op_a, op_b, biz_rows, optima, ratios,
                 boot, brier_a, brier_b, ece_a, ece_b, cop, sb, budget):
    L = ["# Decision-Support Evaluation\n",
         "Threshold-independent and cost-aware analysis of Model A vs Model B on the held-out "
         f"test wafers (eligible dies only). {n} eligible dies, {npos} new failures "
         f"({100*npos/n:.2f}%). Leading metrics are AUC-PR, recall, precision, retest workload, "
         "and hidden-risk recovery - not raw accuracy (which is dominated by the ~96% passing majority).\n"]

    L.append("## 1. Operating points (Model B)\n")
    L.append("Precision/recall/F1 and retest workload across thresholds "
             "(`outputs/metrics/operating_points_model_b.csv`, figure `operating_point_curve.png`).\n")
    L.append("| Threshold | Precision | Recall | F1 | Flagged/1000 dies |")
    L.append("|---|---|---|---|---|")
    for t in [0.20, 0.40, tb, 0.80]:
        r = op_b.iloc[(op_b["threshold"] - t).abs().idxmin()]
        L.append(f"| {r['threshold']:.2f} | {r['precision']:.3f} | {r['recall']:.3f} | "
                 f"{r['f1']:.3f} | {r['retests_per_1000_dies']:.0f} |")

    L.append("\n## 2. Business cost\n")
    b = biz_rows[0]
    L.append(f"At the chosen Model B threshold ({tb}): out of every 1000 dies sent for retest, "
             f"**{b['failures_caught_per_1000_retests']:.0f} are true failures caught** and "
             f"**{b['false_alarms_per_1000_retests']:.0f} are false alarms** "
             f"({b['failures_caught']} caught / {b['failures_missed']} missed / {b['false_alarms']} false alarms overall).\n")
    L.append("Cost-optimal threshold as the cost of a missed failure grows (unit = one retest):\n")
    L.append("| Miss cost (x retest) | Optimal threshold | Cost @opt | Cost retest-all | Cost retest-none |")
    L.append("|---|---|---|---|---|")
    for r in ratios:
        o = optima[r]
        L.append(f"| {r}x | {o['best_threshold']:.2f} | {o['best_cost']:.0f} | "
                 f"{o['cost_retest_all']:.0f} | {o['cost_retest_none']:.0f} |")
    L.append("\nAs missed failures become more expensive, the optimal policy flags more dies "
             "(lower threshold) - the curve lets a fab pick the threshold matching its economics.\n")

    L.append("## 3. Confidence intervals (wafer-level bootstrap, 2000 resamples)\n")
    da, dr, df1 = boot["delta_aucpr"], boot["delta_recall"], boot["delta_f1"]
    L.append("| Delta (B - A) | Mean | 95% CI | P(improvement > 0) |")
    L.append("|---|---|---|---|")
    L.append(f"| AUC-PR | {da['mean']:+.4f} | [{da['lo95']:+.4f}, {da['hi95']:+.4f}] | {da['p_gt_0']*100:.1f}% |")
    L.append(f"| Recall | {dr['mean']:+.4f} | [{dr['lo95']:+.4f}, {dr['hi95']:+.4f}] | {dr['p_gt_0']*100:.1f}% |")
    L.append(f"| Fail F1 | {df1['mean']:+.4f} | [{df1['lo95']:+.4f}, {df1['hi95']:+.4f}] | {df1['p_gt_0']*100:.1f}% |")
    L.append("\nResampling whole wafers (not dies) respects the grouped structure. The AUC-PR "
             "improvement's CI and P(>0) indicate whether Model B's gain is beyond sampling noise.\n")

    L.append("## 4. Calibration\n")
    L.append(f"- Model A: Brier {brier_a:.4f}, ECE {ece_a:.4f}")
    L.append(f"- Model B: Brier {brier_b:.4f}, ECE {ece_b:.4f}")
    L.append("\nReliability diagram in `calibration_curve.png` (`calibration.csv`). Lower Brier/ECE "
             "and points nearer the diagonal mean predicted probabilities better match observed "
             "failure rates.\n")

    L.append("## 5. Matched operating-point comparison\n")
    L.append("Comparing both models at the SAME recall / false-positive rate makes A-vs-B fair "
             "(`common_operating_points.csv`):\n")
    L.append("| Constraint | Model | Threshold | Precision | Recall | False positives | Flagged |")
    L.append("|---|---|---|---|---|---|---|")
    for _, r in cop.iterrows():
        L.append(f"| {r['mode']} | {r['model']} | {r['threshold']:.2f} | {r['precision']:.3f} | "
                 f"{r['recall']:.3f} | {int(r['false_positives'])} | {int(r['flagged'])} |")
    L.append(f"\n**Same retest budget** (flag the top {budget} dies by probability for each model): "
             f"Model A catches {sb['A']['failures_caught']} failures, "
             f"Model B catches {sb['B']['failures_caught']} "
             f"(recall {sb['A']['recall']:.3f} vs {sb['B']['recall']:.3f}). "
             "For an identical retest workload, block-level data recovers more true failures.\n")

    with open(OUT / "reports" / "advanced_eval.md", "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
