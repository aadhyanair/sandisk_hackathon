import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


def save_fig(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_feature_importance(importance_df, title, path, top_n=25):
    df = importance_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(df)), df["importance_mean"].values, xerr=df["importance_std"].values,
            color="#4C72B0", alpha=0.8)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Permutation Importance (AUC-PR)")
    ax.set_title(title)
    save_fig(fig, path)


def plot_category_importance(cat_imp, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"parametric": "#4C72B0", "spatial": "#DD8452",
              "block": "#55A868", "die_aggregate": "#C44E52"}
    c = [colors.get(cat, "#999999") for cat in cat_imp.index]
    ax.bar(cat_imp.index, cat_imp.values, color=c, alpha=0.8)
    ax.set_ylabel("Total Permutation Importance")
    ax.set_title(title)
    save_fig(fig, path)


def plot_pr_curve(y_true, y_prob_a, y_prob_b, path):
    from sklearn.metrics import precision_recall_curve, average_precision_score
    fig, ax = plt.subplots(figsize=(8, 6))

    prec_a, rec_a, _ = precision_recall_curve(y_true, y_prob_a)
    ap_a = average_precision_score(y_true, y_prob_a)
    ax.plot(rec_a, prec_a, label=f"Model A (AP={ap_a:.4f})", color="#4C72B0", lw=2)

    if y_prob_b is not None:
        prec_b, rec_b, _ = precision_recall_curve(y_true, y_prob_b)
        ap_b = average_precision_score(y_true, y_prob_b)
        ax.plot(rec_b, prec_b, label=f"Model B (AP={ap_b:.4f})", color="#DD8452", lw=2)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve (Eligible Dies)")
    ax.legend()
    ax.grid(alpha=0.3)
    save_fig(fig, path)


def plot_confusion_matrix(metrics, title, path):
    cm = np.array(metrics["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred Fail", "Pred Pass"])
    ax.set_yticklabels(["Actual Fail", "Actual Pass"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    ax.set_title(title)
    fig.colorbar(im)
    save_fig(fig, path)


# ------------------------------------------------------------------
# Pass / fail / new-fail semantic colors (shared across wafer views)
# ------------------------------------------------------------------
C_PASS = "#2f9e6b"     # green  - passing die
C_FAIL = "#cf3646"     # red    - failed die (pre-existing)
C_NEW = "#e8792b"      # orange - NEW failure (old_label 0 -> label 1)
C_MISS = "#cf3646"     # red    - missed new failure (FN)
C_FP = "#7b61c9"       # purple - false positive
C_TN = "#cfe8dc"       # faint green - true negative
C_KNOWN = "#8a94a3"    # grey   - pre-test failure / excluded


def _wafer_panel(ax, cols, rows, colors, title, xlim, ylim, s):
    ax.scatter(cols, rows, c=colors, s=s, marker="s", linewidths=0)
    ax.set_title(title, fontsize=12, fontweight="600")
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.invert_yaxis()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#c2ccd6")


def _panel_geometry(rows, cols):
    """Shared limits + marker size so all panels are visually comparable."""
    pad = 1.0
    span = max(cols.max() - cols.min(), rows.max() - rows.min()) + 1
    s = max(6, min(90, int(9000 / max(span, 1))))
    return (cols.min() - pad, cols.max() + pad), (rows.min() - pad, rows.max() + pad), s


def plot_wafer_triptych(wafer_df, title, path):
    """
    Required three-panel view for one wafer, sharing coordinate system / size:
        Pre-Test (old_label)   green=pass, red=fail
        Post-Test (label)      green=pass, red=fail
        Difference             green=stays pass, red=pre-existing fail, orange=NEW fail
    Only real dies are drawn; missing dies stay outside the wafer.
    """
    rows = wafer_df["die_row"].values
    cols = wafer_df["die_col"].values
    old = wafer_df["old_label"].values
    lab = wafer_df["label"].values
    xlim, ylim, s = _panel_geometry(rows, cols)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6))

    pre = np.where(old == 1, C_FAIL, C_PASS)
    _wafer_panel(axes[0], cols, rows, pre, "Pre-Test (old_label)", xlim, ylim, s)

    post = np.where(lab == 1, C_FAIL, C_PASS)
    _wafer_panel(axes[1], cols, rows, post, "Post-Test (label)", xlim, ylim, s)

    diff = np.full(len(rows), C_PASS, dtype=object)
    diff[old == 1] = C_FAIL
    new_fail = (old == 0) & (lab == 1)
    diff[new_fail] = C_NEW
    _wafer_panel(axes[2], cols, rows, diff, "Difference (New Fails Highlighted)", xlim, ylim, s)

    from matplotlib.patches import Patch
    axes[2].legend(handles=[
        Patch(color=C_PASS, label="Passing"),
        Patch(color=C_FAIL, label="Pre-existing fail"),
        Patch(color=C_NEW, label=f"NEW failure ({int(new_fail.sum())})"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3, frameon=False, fontsize=9)

    fig.suptitle(title, fontsize=14, fontweight="700")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, path)


def plot_wafer_prediction_diff(wafer_df, pred_col, title, path):
    """
    Prediction-aware companion view (proves the model predicts the orange dies):
        Actual New Failures | Predicted New Failures | Prediction Outcome (TP/FN/FP)
    Evaluated on eligible dies (old_label==0); pre-test fails shown grey.
    """
    rows = wafer_df["die_row"].values
    cols = wafer_df["die_col"].values
    old = wafer_df["old_label"].values
    lab = wafer_df["label"].values
    pred = wafer_df[pred_col].values
    xlim, ylim, s = _panel_geometry(rows, cols)
    elig = old == 0

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 6))

    a_col = np.where(old == 1, C_KNOWN, np.where((old == 0) & (lab == 1), C_NEW, C_TN))
    _wafer_panel(axes[0], cols, rows, a_col, "Actual New Failures", xlim, ylim, s)

    p_col = np.where(old == 1, C_KNOWN, np.where((old == 0) & (pred == 1), C_NEW, C_TN))
    _wafer_panel(axes[1], cols, rows, p_col, "Predicted New Failures", xlim, ylim, s)

    outcome = np.full(len(rows), C_TN, dtype=object)
    outcome[old == 1] = C_KNOWN
    tp = elig & (lab == 1) & (pred == 1)
    fn = elig & (lab == 1) & (pred == 0)
    fp = elig & (lab == 0) & (pred == 1)
    outcome[tp] = C_NEW
    outcome[fn] = C_MISS
    outcome[fp] = C_FP
    _wafer_panel(axes[2], cols, rows, outcome, "Prediction Outcome", xlim, ylim, s)

    from matplotlib.patches import Patch
    axes[2].legend(handles=[
        Patch(color=C_NEW, label=f"TP correct ({int(tp.sum())})"),
        Patch(color=C_MISS, label=f"FN missed ({int(fn.sum())})"),
        Patch(color=C_FP, label=f"FP false alarm ({int(fp.sum())})"),
        Patch(color=C_TN, label="TN"),
    ], loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=4, frameon=False, fontsize=8.5)

    fig.suptitle(title, fontsize=14, fontweight="700")
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    save_fig(fig, path)


def plot_wafer_map(wafer_df, prob_col, title, path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    rows, cols = wafer_df["die_row"].values, wafer_df["die_col"].values
    old_labels = wafer_df["old_label"].values

    colors = np.where(old_labels == 1, "red", "green")
    axes[0].scatter(cols, rows, c=colors, s=8, marker="s")
    axes[0].set_title("Pre-test (old_label)")
    axes[0].invert_yaxis()
    axes[0].set_aspect("equal")

    if "label" in wafer_df.columns:
        labels = wafer_df["label"].values
        c2 = np.where(old_labels == 1, "red",
                      np.where(labels == 1, "orange", "green"))
        axes[1].scatter(cols, rows, c=c2, s=8, marker="s")
        axes[1].set_title("Post-test (label)")
        axes[1].invert_yaxis()
        axes[1].set_aspect("equal")

    if prob_col in wafer_df.columns:
        eligible = old_labels == 0
        sc = axes[2].scatter(cols[eligible], rows[eligible],
                             c=wafer_df[prob_col].values[eligible],
                             cmap="YlOrRd", s=8, marker="s", vmin=0, vmax=1)
        old_fail = old_labels == 1
        axes[2].scatter(cols[old_fail], rows[old_fail], c="gray", s=8, marker="x", alpha=0.5)
        axes[2].set_title("Predicted Probability")
        axes[2].invert_yaxis()
        axes[2].set_aspect("equal")
        fig.colorbar(sc, ax=axes[2])

    fig.suptitle(title)
    save_fig(fig, path)


def plot_model_comparison(metrics_a, metrics_b, path):
    metric_names = ["auc_pr", "fail_f1", "fail_precision", "fail_recall", "accuracy", "roc_auc"]
    labels = ["AUC-PR", "Fail F1", "Fail Prec", "Fail Rec", "Accuracy", "ROC-AUC"]

    vals_a = [metrics_a[m] for m in metric_names]
    vals_b = [metrics_b[m] for m in metric_names]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, vals_a, width, label="Model A", color="#4C72B0", alpha=0.8)
    ax.bar(x + width/2, vals_b, width, label="Model B", color="#DD8452", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_title("Model A vs Model B Comparison")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, 1)
    save_fig(fig, path)


def plot_block_examples(df, n_examples=3, path="outputs/figures/block_examples.png"):
    eligible = df[df["old_label"] == 0]
    if "label" not in eligible.columns:
        return

    passes = eligible[eligible["label"] == 0]
    fails = eligible[eligible["label"] == 1]

    fig, axes = plt.subplots(2, n_examples, figsize=(5*n_examples, 8))
    if n_examples == 1:
        axes = axes.reshape(2, 1)

    for i in range(min(n_examples, len(passes))):
        arr = np.fromstring(passes.iloc[i]["block_readings"], dtype=np.float32, sep=' ')
        axes[0, i].plot(arr, lw=0.3, color="#4C72B0")
        axes[0, i].set_title(f"Pass Die #{i+1}")
        axes[0, i].set_ylabel("Reading")

    for i in range(min(n_examples, len(fails))):
        arr = np.fromstring(fails.iloc[i]["block_readings"], dtype=np.float32, sep=' ')
        axes[1, i].plot(arr, lw=0.3, color="#DD8452")
        axes[1, i].set_title(f"New Fail Die #{i+1}")
        axes[1, i].set_ylabel("Reading")

    fig.suptitle("Block Reading Examples: Pass vs New Fail")
    fig.tight_layout()
    save_fig(fig, path)


def plot_block_examples_from_arrays(pass_arrays, fail_arrays, path="outputs/figures/block_examples.png"):
    n = max(len(pass_arrays), len(fail_arrays), 1)
    fig, axes = plt.subplots(2, n, figsize=(5*n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)
    for i, arr in enumerate(pass_arrays):
        axes[0, i].plot(arr, lw=0.3, color="#4C72B0")
        axes[0, i].set_title(f"Pass Die #{i+1}")
        axes[0, i].set_ylabel("Reading")
    for i, arr in enumerate(fail_arrays):
        axes[1, i].plot(arr, lw=0.3, color="#DD8452")
        axes[1, i].set_title(f"New Fail Die #{i+1}")
        axes[1, i].set_ylabel("Reading")
    fig.suptitle("Block Reading Examples: Pass vs New Fail")
    fig.tight_layout()
    save_fig(fig, path)


def plot_probability_distribution(y_true, y_prob, title, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_true == 0], bins=50, alpha=0.6, label="Pass", color="#4C72B0", density=True)
    ax.hist(y_prob[y_true == 1], bins=50, alpha=0.6, label="Fail", color="#DD8452", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    save_fig(fig, path)
