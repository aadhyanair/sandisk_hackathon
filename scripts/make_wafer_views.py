"""
Generate the required three-panel wafer visualisation (Pre-Test / Post-Test /
Difference-new-fails) and the prediction-aware companion (Actual / Predicted /
Outcome) for the most informative test wafers.

Reads only the light metadata columns from input/test.csv plus the saved
predictions -- no model is run. Figures go to outputs/figures/wafer_views/.
"""
import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from pathlib import Path

from src.visualization import plot_wafer_triptych, plot_wafer_prediction_diff

ROOT = Path(os.path.dirname(__file__)).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wafers", type=int, default=4)
    ap.add_argument("--wafers", nargs="*", default=None, help="explicit wafer ids")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "input" / "test.csv",
                       usecols=["wafer_id", "die_row", "die_col", "old_label", "label"])
    preds = pd.read_csv(ROOT / "outputs" / "predictions" / "test_predictions.csv")
    df = meta.merge(preds, on=["wafer_id", "die_row", "die_col"], how="left")
    df["predicted_label"] = df["predicted_label"].fillna(0).astype(int)

    df["new_fail"] = ((df["old_label"] == 0) & (df["label"] == 1)).astype(int)
    if args.wafers:
        wafers = args.wafers
    else:
        ranked = df.groupby("wafer_id")["new_fail"].sum().sort_values(ascending=False)
        wafers = list(ranked.head(args.n_wafers).index)

    out = ROOT / "outputs" / "figures" / "wafer_views"
    out.mkdir(parents=True, exist_ok=True)
    print(f"Generating wafer views for: {wafers}")
    for wid in wafers:
        wdf = df[df["wafer_id"] == wid]
        nnew = int(wdf["new_fail"].sum())
        plot_wafer_triptych(wdf, f"Wafer {wid}  -  {nnew} new failures",
                            out / f"wafer_{wid}_triptych.png")
        plot_wafer_prediction_diff(wdf, "predicted_label",
                                   f"Wafer {wid}  -  Model B prediction vs actual",
                                   out / f"wafer_{wid}_prediction.png")
        print(f"  {wid}: {nnew} new fails -> triptych + prediction views")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
