"""
mlflow_registry.py

This file manages MLflow registry operations.

It enforces a guarded promotion rule:
No model version can be promoted unless human approval is present.
"""

import json
import mlflow
from pathlib import Path

from service.config.settings import Settings


class MLflowRegistry:
    """
    Registry interface for model operations.
    """

    def __init__(self, registered_model_name: str):
        self.registered_model_name = registered_model_name
        mlflow.set_tracking_uri("file:./mlruns")

    def get_current_production_model(self) -> dict:
        """
        Return current Production model metadata.
        """

        client = mlflow.tracking.MlflowClient()

        versions = client.search_model_versions(
            f"name='{self.registered_model_name}'"
        )

        production_versions = [
            version for version in versions
            if version.current_stage == "Production"
        ]

        if not production_versions:
            return self._local_model_info()

        production = production_versions[0]

        return {
            "model_name": self.registered_model_name,
            "model_version": production.version,
            "stage": production.current_stage,
            "run_id": production.run_id
        }

    def _local_model_info(self) -> dict:
        """Fall back to local artifact metadata when no MLflow production model exists."""
        metrics_path = Path("artifacts/reports/metrics.json")
        threshold_path = Path("artifacts/reports/threshold.json")
        try:
            metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
            threshold = json.loads(threshold_path.read_text()) if threshold_path.exists() else {}
            test = metrics.get("test", {})
            return {
                "model_name": self.registered_model_name,
                "model_version": "1",
                "stage": "local",
                "source": "local artifact",
                "threshold": threshold.get("threshold"),
                "auc": test.get("auc"),
                "recall": test.get("recall"),
                "f1": test.get("f1"),
                "message": "Loaded from local artifact (not yet promoted in MLflow).",
            }
        except Exception:
            return {
                "model_name": self.registered_model_name,
                "model_version": "1",
                "stage": "local",
                "message": "Loaded from local artifact.",
            }

    def get_candidate_model(self) -> dict:
        """
        Return the latest Staging (candidate) model version, or null if none registered.
        """
        client = mlflow.tracking.MlflowClient()
        try:
            versions = client.search_model_versions(
                f"name='{self.registered_model_name}'"
            )
            staging = [v for v in versions if v.current_stage == "Staging"]
            if not staging:
                return {
                    "candidate_version": None,
                    "message": "No candidate model registered yet — production model unchanged.",
                }
            latest = max(staging, key=lambda v: int(v.version))
            return {
                "candidate_version": latest.version,
                "stage": latest.current_stage,
                "run_id": latest.run_id,
                "model_name": self.registered_model_name,
            }
        except Exception:
            return {
                "candidate_version": None,
                "message": "No candidate model registered yet — production model unchanged.",
            }

    def validate_promotion_gate(self, candidate_version: str) -> bool:
        """
        Validate whether a candidate model can be promoted.

        Current gate checks:
        - Candidate version must exist in MLflow

        Later gate checks:
        - Metrics pass checklist
        - Schema exists
        - Model card exists
        - Human approval is fresh
        """

        if not candidate_version:
            return False

        client = mlflow.tracking.MlflowClient()

        try:
            version = client.get_model_version(
                name=self.registered_model_name,
                version=candidate_version
            )
        except Exception:
            return False

        return version is not None

    def promote_to_production(self, candidate_version: str, approved: bool) -> dict:
        """
        Promote model only if gate passes and human approval exists.
        """

        if not approved:
            return {
                "status": "rejected",
                "reason": "Human approval is required before Production change."
            }

        if not self.validate_promotion_gate(candidate_version):
            return {
                "status": "rejected",
                "reason": "Promotion gate failed or model version does not exist."
            }

        client = mlflow.tracking.MlflowClient()

        client.transition_model_version_stage(
            name=self.registered_model_name,
            version=candidate_version,
            stage="Production",
            archive_existing_versions=True
        )

        return {
            "status": "approved",
            "model_name": self.registered_model_name,
            "promoted_version": candidate_version,
            "stage": "Production"
        }