# async_queue/tasks.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ARTIFACTS_ROOT = Path("artifacts")
MODEL_PATH = ARTIFACTS_ROOT / "models" / "bank_marketing_pipeline.joblib"
THRESHOLD_PATH = ARTIFACTS_ROOT / "reports" / "threshold.json"
TEST_DATA_PATH = Path("data") / "processed" / "test.csv"
REPLAY_REPORTS_DIR = ARTIFACTS_ROOT / "reports" / "replay_reports"

NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "pdays_was_999",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def run_replay_test(payload: dict) -> dict:
    """
    Replay the held-out test set through the current production model
    and write a timestamped report to artifacts/reports/replay_reports/.
    """
    investigation_id = payload.get("investigation_id", "unknown")
    logger.info("Running replay test for investigation %s", investigation_id)

    model = joblib.load(MODEL_PATH)
    with open(THRESHOLD_PATH, "r") as f:
        threshold = json.load(f)["threshold"]

    test_df = pd.read_csv(TEST_DATA_PATH)
    X_test = test_df[MODEL_FEATURES]
    y_true = test_df["y"].values

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    metrics = {
        "auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "n_samples": len(y_true),
    }

    REPLAY_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_path = REPLAY_REPORTS_DIR / f"replay_{timestamp}_{investigation_id[:8]}.json"
    report = {
        "investigation_id": investigation_id,
        "timestamp": timestamp,
        "metrics": metrics,
    }
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Replay test complete: AUC=%.4f recall=%.4f", metrics["auc"], metrics["recall"])
    return {"status": "completed", "metrics": metrics, "report_path": str(report_path)}


def run_retrain(payload: dict) -> dict:
    """
    Retrain the model on the current training split.
    Does NOT promote to Production — human approval is required for that.
    """
    investigation_id = payload.get("investigation_id", "unknown")
    logger.info("Starting retrain for investigation %s", investigation_id)

    try:
        from src.ml.train import train_model

        result = train_model()
        logger.info("Retrain complete: threshold=%.3f", result.get("threshold", 0))
        return {
            "status": "completed",
            "mlflow_run_id": result.get("mlflow_run_id"),
            "threshold": result.get("threshold"),
            "note": "New model candidate registered. Requires human approval to promote to Production.",
        }
    except Exception as exc:
        logger.exception("Retrain failed")
        return {"status": "failed", "error": str(exc)}


def run_rollback(payload: dict) -> dict:
    """
    Roll back to a previous registered model version in MLflow.
    Does NOT restart the service — that requires an ops action.
    """
    investigation_id = payload.get("investigation_id", "unknown")
    target_version = payload.get("target_version")
    logger.info(
        "Rollback requested: investigation=%s target_version=%s",
        investigation_id,
        target_version,
    )

    if not target_version:
        return {"status": "failed", "error": "target_version not specified in payload"}

    try:
        import mlflow

        client = mlflow.tracking.MlflowClient()
        model_name = "bank-marketing-classifier"

        client.transition_model_version_stage(
            name=model_name,
            version=str(target_version),
            stage="Production",
            archive_existing_versions=True,
        )

        logger.info("MLflow rollback complete to version %s", target_version)
        return {
            "status": "completed",
            "rolled_back_to_version": target_version,
            "note": "MLflow stage updated. Service restart required to load new artifact.",
        }
    except Exception as exc:
        logger.exception("Rollback failed")
        return {"status": "failed", "error": str(exc)}
