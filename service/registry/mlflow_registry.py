"""
mlflow_registry.py

This file defines the registry interface for model registration and promotion.

For now, it is a safe placeholder. Later, it will connect to MLflow and enforce
the promotion checklist before any model is moved to Production.
"""


class MLflowRegistry:
    """
    Registry interface for model operations.
    """

    def __init__(self, registered_model_name: str):
        self.registered_model_name = registered_model_name

    def get_current_production_model(self) -> dict:
        """
        Return current production model metadata.

        Placeholder until MLflow integration is added.
        """

        return {
            "model_name": self.registered_model_name,
            "model_version": "v1",
            "stage": "Production"
        }

    def validate_promotion_gate(self, candidate_version: str) -> bool:
        """
        Validate whether a candidate model is allowed to be promoted.

        This will later enforce:
        - model artifact exists
        - schema exists
        - model card exists
        - metrics pass checklist
        - human approval exists
        """

        return bool(candidate_version)

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
                "reason": "Promotion gate failed."
            }

        return {
            "status": "approved",
            "model_name": self.registered_model_name,
            "promoted_version": candidate_version,
            "stage": "Production"
        }