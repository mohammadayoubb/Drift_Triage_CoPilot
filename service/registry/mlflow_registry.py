"""
mlflow_registry.py

This file manages MLflow registry operations.

It enforces a guarded promotion rule:
No model version can be promoted unless human approval is present.
"""

import mlflow

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
        Return current production model metadata.

        If no Production model exists yet, return a clear state.
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
            return {
                "model_name": self.registered_model_name,
                "model_version": None,
                "stage": "None",
                "message": "No Production model is currently set."
            }

        production = production_versions[0]

        return {
            "model_name": self.registered_model_name,
            "model_version": production.version,
            "stage": production.current_stage,
            "run_id": production.run_id
        }

    def validate_promotion_gate(self, candidate_version: str) -> bool:
        """
        Validate whether a candidate model can be promoted.
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