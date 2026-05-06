# src/ml/threshold.py

import numpy as np
from sklearn.metrics import recall_score

from ml.constants import RECALL_TARGET


def choose_threshold_for_recall(
    y_true,
    y_proba,
    min_recall: float = RECALL_TARGET,
) -> float:
    """
    Pick the highest threshold that still achieves recall >= min_recall.

    Why highest?
    - Lower thresholds catch more positives but may create many false positives.
    - The project rule requires recall >= 0.75.
    - So we choose the highest threshold that satisfies the recall constraint.
    """
    thresholds = np.linspace(0.0, 1.0, 1001)

    valid_thresholds: list[float] = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        recall = recall_score(y_true, y_pred, zero_division=0)

        if recall >= min_recall:
            valid_thresholds.append(float(threshold))

    if not valid_thresholds:
        raise ValueError(
            f"No threshold found that satisfies recall >= {min_recall}. "
            "This should be rare because threshold=0 usually predicts all positives."
        )

    return max(valid_thresholds)