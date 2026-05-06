"""
Registry handles:
MLflow experiment
MLflow metrics
MLflow artifacts
MLflow model registration
input schema artifact
"""

import json
import os
from typing import Any

import mlflow
import mlflow.sklearn
import pandas as pd
import logging

logger = logging.getLogger(__name__)

from common.paths import (
    INPUT_SCHEMA_PATH,
    METRICS_PATH,
    MODEL_CARD_PATH,
    THRESHOLD_PATH,
)
from ml.constants import MLFLOW_EXPERIMENT_NAME, MLFLOW_MODEL_NAME, MODEL_FEATURES


def write_input_schema(X_sample: pd.DataFrame) -> None:
    """
    Write a simple input schema artifact.

    This is not the FastAPI Pydantic schema yet.
    It is an artifact that documents the fields the model expects.
    """
    schema_payload = {
        "features": [
            {
                "name": column,
                "dtype": str(X_sample[column].dtype),
                "required": True,
            }
            for column in MODEL_FEATURES
        ]
    }

    INPUT_SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)

    INPUT_SCHEMA_PATH.write_text(
        json.dumps(schema_payload, indent=2),
        encoding="utf-8",
    )

    print(f"Input schema written to: {INPUT_SCHEMA_PATH}")


def log_metrics_to_mlflow(metrics_payload: dict[str, Any]) -> None:
    """
    Log validation/test metrics to the active MLflow run.
    """
    for split_name in ["validation", "test"]:
        split_metrics = metrics_payload[split_name]

        for metric_name, metric_value in split_metrics.items():
            if metric_name == "confusion_matrix":
                continue

            if metric_value is not None:
                mlflow.log_metric(f"{split_name}_{metric_name}", float(metric_value))

        confusion_matrix = split_metrics["confusion_matrix"]
        for cm_name, cm_value in confusion_matrix.items():
            mlflow.log_metric(f"{split_name}_cm_{cm_name}", float(cm_value))


def log_and_register_model(
    pipeline,
    metrics_payload: dict[str, Any],
    threshold_payload: dict[str, Any],
    X_train_sample: pd.DataFrame,
) -> str:
    """
    Log and register the fitted model pipeline in MLflow.

    Artifacts logged:
    - fitted sklearn pipeline
    - metrics.json
    - threshold.json
    - input_schema.json
    - model card
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    write_input_schema(X_train_sample)

    with mlflow.start_run(run_name="logistic-regression-baseline") as run:
        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "class_weight": "balanced",
                "preprocessor": "ColumnTransformer",
                "registered_model_name": MLFLOW_MODEL_NAME,
                "threshold_rule": threshold_payload["rule"],
                "threshold": threshold_payload["threshold"],
                "threshold_tuned_on": threshold_payload["tuned_on"],
            }
        )

        log_metrics_to_mlflow(metrics_payload)

        mlflow.log_artifact(str(METRICS_PATH), artifact_path="reports")
        mlflow.log_artifact(str(THRESHOLD_PATH), artifact_path="reports")
        mlflow.log_artifact(str(INPUT_SCHEMA_PATH), artifact_path="schema")
        mlflow.log_artifact(str(MODEL_CARD_PATH), artifact_path="model_card")

        model_info = mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=X_train_sample,
            registered_model_name=MLFLOW_MODEL_NAME,
        )

        run_id = run.info.run_id

    logger.info("MLflow model logged successfully")
    logger.info("MLflow run ID: %s", run_id)
    logger.info("MLflow model URI: %s", model_info.model_uri)
    logger.info("Registered model name: %s", MLFLOW_MODEL_NAME)

    return run_id