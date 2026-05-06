# tests/ml/test_threshold.py

import numpy as np
from sklearn.metrics import recall_score

from ml.threshold import choose_threshold_for_recall


def test_choose_threshold_meets_minimum_recall() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_threshold_for_recall(
        y_true=y_true,
        y_proba=y_proba,
        min_recall=0.5,
    )

    y_pred = (y_proba >= threshold).astype(int)
    recall = recall_score(y_true, y_pred)

    assert recall >= 0.5


def test_choose_threshold_returns_highest_valid_threshold() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_threshold_for_recall(
        y_true=y_true,
        y_proba=y_proba,
        min_recall=1.0,
    )

    assert threshold <= 0.6
    assert threshold > 0.59