"""
register_model.py

This script registers the trained Bank Marketing pipeline in MLflow.

It logs the required artifact triple:
- model binary
- schema
- model card

It also logs metadata such as threshold, metrics, and model name.
"""

import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import joblib


ARTIFACT_DIR = Path("model_artifacts")

MODEL_PATH = ARTIFACT_DIR / "bank_marketing_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
SCHEMA_PATH = ARTIFACT_DIR / "model_schema.json"
MODEL_CARD_PATH = ARTIFACT_DIR / "model_card.json"

REGISTERED_MODEL_NAME = "bank_marketing_classifier"
MLFLOW_TRACKING_URI = "file:./mlruns"


def load_json(path: Path) -> dict:
    """
    Load a JSON file.
    """

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """
    Register model and artifacts in MLflow.
    """

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("bank_marketing_drift_triage")

    metadata = load_json(METADATA_PATH)
    pipeline = joblib.load(MODEL_PATH)

    with mlflow.start_run(run_name="register_bank_marketing_classifier") as run:
        run_id = run.info.run_id

        # Log model parameters
        mlflow.log_param("model_name", REGISTERED_MODEL_NAME)
        mlflow.log_param("model_type", metadata["model_type"])
        mlflow.log_param("operating_threshold", metadata["operating_threshold"])
        mlflow.log_param("threshold_rule", metadata["threshold_rule"])

        # Log metrics
        for metric_name, metric_value in metadata["test_metrics"].items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)

        # Log supporting artifacts
        mlflow.log_artifact(str(MODEL_PATH), artifact_path="raw_artifacts")
        mlflow.log_artifact(str(METADATA_PATH), artifact_path="raw_artifacts")
        mlflow.log_artifact(str(SCHEMA_PATH), artifact_path="raw_artifacts")
        mlflow.log_artifact(str(MODEL_CARD_PATH), artifact_path="raw_artifacts")

        # Log sklearn pipeline as MLflow model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL_NAME,
        )

        print("Registered model in MLflow")
        print(f"Run ID: {run_id}")
        print(f"Registered model name: {REGISTERED_MODEL_NAME}")
        print(f"Tracking URI: {MLFLOW_TRACKING_URI}")


if __name__ == "__main__":
    main()