# src/ml/drift_reference.py

import json
from typing import Any

import numpy as np
import pandas as pd
import logging

from common.paths import REFERENCE_STATS_PATH, ensure_project_dirs
from ml.constants import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERIC_FEATURES, TARGET

logger = logging.getLogger(__name__)

def build_numeric_reference(
    df: pd.DataFrame,
    numeric_features: list[str],
    bins: int = 10,
) -> dict[str, Any]:
    """
    Build reference distributions for numeric features.

    These bin edges and proportions will later be used for PSI.
    """
    numeric_reference: dict[str, Any] = {}

    for feature in numeric_features:
        series = pd.to_numeric(df[feature], errors="coerce").dropna()

        if series.empty:
            numeric_reference[feature] = {
                "bin_edges": [],
                "proportions": [],
                "counts": [],
                "count": 0,
            }
            continue

        if series.nunique() == 1:
            value = float(series.iloc[0])
            numeric_reference[feature] = {
                "bin_edges": [value - 0.5, value + 0.5],
                "proportions": [1.0],
                "counts": [int(series.shape[0])],
                "count": int(series.shape[0]),
            }
            continue

        counts, bin_edges = np.histogram(series, bins=bins)

        total = counts.sum()
        proportions = counts / total if total > 0 else np.zeros_like(counts, dtype=float)

        numeric_reference[feature] = {
            "bin_edges": [float(edge) for edge in bin_edges],
            "proportions": [float(value) for value in proportions],
            "counts": [int(value) for value in counts],
            "count": int(series.shape[0]),
        }

    return numeric_reference


def build_categorical_reference(
    df: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, Any]:
    """
    Build reference distributions for categorical features.

    These counts/proportions will later be used for chi-square drift checks.
    """
    categorical_reference: dict[str, Any] = {}

    for feature in categorical_features:
        series = df[feature].fillna("__MISSING__").astype(str)

        counts = series.value_counts()
        proportions = series.value_counts(normalize=True)

        categorical_reference[feature] = {
            "counts": {
                str(category): int(count)
                for category, count in counts.items()
            },
            "proportions": {
                str(category): float(proportion)
                for category, proportion in proportions.items()
            },
            "count": int(series.shape[0]),
        }

    return categorical_reference


def build_target_reference(df: pd.DataFrame) -> dict[str, Any]:
    """
    Store the training target distribution.

    This is not the same as prediction output drift, but it is useful context.
    """
    target_series = df[TARGET]

    counts = target_series.value_counts()
    proportions = target_series.value_counts(normalize=True)

    return {
        "counts": {
            str(label): int(count)
            for label, count in counts.items()
        },
        "proportions": {
            str(label): float(proportion)
            for label, proportion in proportions.items()
        },
        "positive_rate": float(target_series.mean()),
        "count": int(target_series.shape[0]),
    }


def build_prediction_output_reference(
    df: pd.DataFrame,
    pipeline,
    threshold: float,
) -> dict[str, Any]:
    """
    Store the model's reference prediction distribution on the training split.

    This is what we compare future live prediction output distribution against.
    """
    X = df[MODEL_FEATURES]

    probabilities = pipeline.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    positive_rate = float(np.mean(predictions))
    negative_rate = float(1.0 - positive_rate)

    return {
        "threshold": float(threshold),
        "positive_prediction_rate": positive_rate,
        "negative_prediction_rate": negative_rate,
        "mean_probability": float(np.mean(probabilities)),
        "min_probability": float(np.min(probabilities)),
        "max_probability": float(np.max(probabilities)),
        "count": int(len(predictions)),
    }


def build_reference_stats(
    train_df: pd.DataFrame,
    pipeline,
    threshold: float,
) -> dict[str, Any]:
    """
    Build the full drift reference payload.
    """
    return {
        "source": "training_split",
        "numeric_features": build_numeric_reference(train_df, NUMERIC_FEATURES),
        "categorical_features": build_categorical_reference(train_df, CATEGORICAL_FEATURES),
        "target_distribution": build_target_reference(train_df),
        "output_distribution": build_prediction_output_reference(
            df=train_df,
            pipeline=pipeline,
            threshold=threshold,
        ),
    }


def write_reference_stats(
    train_df: pd.DataFrame,
    pipeline,
    threshold: float,
) -> None:
    """
    Write reference statistics to data/reference/reference_stats.json.
    """
    ensure_project_dirs()

    reference_payload = build_reference_stats(
        train_df=train_df,
        pipeline=pipeline,
        threshold=threshold,
    )

    REFERENCE_STATS_PATH.write_text(
        json.dumps(reference_payload, indent=2),
        encoding="utf-8",
    )

    logger.info("Reference drift stats written to: %s", REFERENCE_STATS_PATH)