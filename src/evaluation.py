import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, f1_score,
    precision_score, recall_score, accuracy_score, confusion_matrix,
    precision_recall_curve
)


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "auc_pr": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "fail_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fail_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "fail_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "pass_f1": float(f1_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "pass_precision": float(precision_score(y_true, y_pred, pos_label=0, zero_division=0)),
        "pass_recall": float(recall_score(y_true, y_pred, pos_label=0, zero_division=0)),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    metrics["confusion_matrix"] = cm.tolist()
    metrics["tp"] = int(cm[0, 0])
    metrics["fn"] = int(cm[0, 1])
    metrics["fp"] = int(cm[1, 0])
    metrics["tn"] = int(cm[1, 1])

    return metrics


def find_best_threshold(y_true, y_prob, metric="f1"):
    best_t, best_score = 0.5, 0.0
    for t in np.arange(0.01, 1.0, 0.01):
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        if score > best_score:
            best_score = score
            best_t = t
    return float(best_t), float(best_score)


def print_metrics(metrics, model_name):
    print(f"\n{'='*50}")
    print(f"  {model_name} — Metrics (eligible dies only)")
    print(f"{'='*50}")
    print(f"  AUC-PR:          {metrics['auc_pr']:.4f}")
    print(f"  ROC-AUC:         {metrics['roc_auc']:.4f}")
    print(f"  Threshold:       {metrics['threshold']:.2f}")
    print(f"  Accuracy:        {metrics['accuracy']:.4f}")
    print(f"  Fail F1:         {metrics['fail_f1']:.4f}")
    print(f"  Fail Precision:  {metrics['fail_precision']:.4f}")
    print(f"  Fail Recall:     {metrics['fail_recall']:.4f}")
    print(f"  Pass F1:         {metrics['pass_f1']:.4f}")
    print(f"  Pass Precision:  {metrics['pass_precision']:.4f}")
    print(f"  Pass Recall:     {metrics['pass_recall']:.4f}")
    cm = metrics["confusion_matrix"]
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"                  Pred Fail  Pred Pass")
    print(f"  Actual Fail     {cm[0][0]:>8}   {cm[0][1]:>8}")
    print(f"  Actual Pass     {cm[1][0]:>8}   {cm[1][1]:>8}")
    print(f"{'='*50}\n")
