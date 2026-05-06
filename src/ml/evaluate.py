# src/ml/evaluate.py

from typing import Any

import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate_binary_classifier(
    y_true,
    y_proba,
    threshold: float,
) -> dict[str, Any]:
    """
    Evaluate binary classifier probabilities using a chosen operating threshold.
    """
    y_pred = (y_proba >= threshold).astype(int)

    labels = [0, 1]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()

    unique_classes = set(np.asarray(y_true).tolist())

    auc = None
    if len(unique_classes) == 2:
        auc = float(roc_auc_score(y_true, y_proba))

    return {
        "threshold": float(threshold),
        "auc": auc,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
    }