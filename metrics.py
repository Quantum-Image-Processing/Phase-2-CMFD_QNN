"""Image-level metrics + mean/std summary across seeds."""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             precision_recall_fscore_support, roc_auc_score)


def compute_metrics(y_true, scores, threshold):
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    y_pred = (scores >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    out = dict(accuracy=float(accuracy_score(y_true, y_pred)),
               balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
               precision=float(p), recall=float(r), f1=float(f1))
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, scores))
    except ValueError:
        out["roc_auc"] = float("nan")
    return out


def summarize(runs):
    keys = runs[0].keys()
    return {k: dict(mean=float(np.nanmean([r[k] for r in runs])),
                    std=float(np.nanstd([r[k] for r in runs]))) for k in keys}
