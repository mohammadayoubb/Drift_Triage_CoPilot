# src/ml/train.py

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from common.paths import (
    METRICS_PATH,
    MODEL_PATH,
    THRESHOLD_PATH,
    ensure_project_dirs,
)
from ml.constants import (
    MLFLOW_MODEL_NAME,
    MODEL_FEATURES,
    RECALL_TARGET,
    TARGET,
)
from ml.data_loader import load_splits
from ml.drift_reference import write_reference_stats
from ml.evaluate import evaluate_binary_classifier
from ml.model_card import write_model_card
from ml.pipeline import build_model_pipeline
from ml.registry import log_and_register_model
from ml.threshold import choose_threshold_for_recall

logger = logging.getLogger(__name__)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split processed data into model features X and target y.
    """
    missing_features = sorted(set(MODEL_FEATURES) - set(df.columns))
    if missing_features:
        raise ValueError(f"Missing model features: {missing_features}")

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    X = df[MODEL_FEATURES]
    y = df[TARGET]

    return X, y


def save_json(payload: dict[str, Any], path: Path) -> None:
    """
    Save a dictionary payload as formatted JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def train_model() -> dict[str, Any]:
    """
    Full training flow:

    1. Load train/val/test splits.
    2. Fit sklearn pipeline on train.
    3. Tune threshold on validation.
    4. Evaluate on validation and test.
    5. Save local artifacts.
    6. Write model card.
    7. Write drift reference stats.
    8. Log and register model in MLflow.
    """
    logger.info("Starting training flow")
    ensure_project_dirs()

    logger.info("Loading train, validation, and test splits")
    train_df, val_df, test_df = load_splits()

    logger.info(
        "Loaded splits",
        extra={
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "test_rows": len(test_df),
        },
    )

    X_train, y_train = split_features_target(train_df)
    X_val, y_val = split_features_target(val_df)
    X_test, y_test = split_features_target(test_df)

    logger.info(
        "Prepared feature matrices",
        extra={
            "n_features": len(MODEL_FEATURES),
            "train_shape": X_train.shape,
            "val_shape": X_val.shape,
            "test_shape": X_test.shape,
        },
    )

    pipeline = build_model_pipeline()

    logger.info("Training model pipeline")
    pipeline.fit(X_train, y_train)

    logger.info("Predicting validation and test probabilities")
    val_proba = pipeline.predict_proba(X_val)[:, 1]
    test_proba = pipeline.predict_proba(X_test)[:, 1]

    logger.info("Choosing operating threshold", extra={"recall_target": RECALL_TARGET})
    threshold = choose_threshold_for_recall(
        y_true=y_val,
        y_proba=val_proba,
        min_recall=RECALL_TARGET,
    )

    logger.info("Chosen operating threshold: %.4f", threshold)

    logger.info("Evaluating validation metrics")
    val_metrics = evaluate_binary_classifier(
        y_true=y_val,
        y_proba=val_proba,
        threshold=threshold,
    )

    logger.info("Evaluating test metrics")
    test_metrics = evaluate_binary_classifier(
        y_true=y_test,
        y_proba=test_proba,
        threshold=threshold,
    )

    metrics_payload = {
        "model_name": MLFLOW_MODEL_NAME,
        "threshold_rule": f"highest threshold with recall >= {RECALL_TARGET}",
        "threshold": threshold,
        "validation": val_metrics,
        "test": test_metrics,
    }

    threshold_payload = {
        "threshold": threshold,
        "rule": f"highest threshold with recall >= {RECALL_TARGET}",
        "tuned_on": "validation",
    }

    logger.info(
        "Evaluation complete",
        extra={
            "validation_auc": val_metrics.get("auc"),
            "validation_f1": val_metrics.get("f1"),
            "validation_precision": val_metrics.get("precision"),
            "validation_recall": val_metrics.get("recall"),
            "test_auc": test_metrics.get("auc"),
            "test_f1": test_metrics.get("f1"),
            "test_precision": test_metrics.get("precision"),
            "test_recall": test_metrics.get("recall"),
        },
    )

    logger.info("Saving local artifacts")
    joblib.dump(pipeline, MODEL_PATH)
    save_json(metrics_payload, METRICS_PATH)
    save_json(threshold_payload, THRESHOLD_PATH)

    logger.info("Saved model artifact to %s", MODEL_PATH)
    logger.info("Saved metrics artifact to %s", METRICS_PATH)
    logger.info("Saved threshold artifact to %s", THRESHOLD_PATH)

    logger.info("Writing model card")
    write_model_card(metrics_payload)

    logger.info("Writing drift reference statistics")
    write_reference_stats(
        train_df=train_df,
        pipeline=pipeline,
        threshold=threshold,
    )

    logger.info("Logging and registering model in MLflow")
    run_id = log_and_register_model(
        pipeline=pipeline,
        metrics_payload=metrics_payload,
        threshold_payload=threshold_payload,
        X_train_sample=X_train.head(5),
    )

    metrics_payload["mlflow_run_id"] = run_id
    save_json(metrics_payload, METRICS_PATH)

    logger.info("Training flow completed successfully")
    logger.info("MLflow run ID: %s", run_id)
    logger.info("Final metrics payload:\n%s", json.dumps(metrics_payload, indent=2))

    return metrics_payload