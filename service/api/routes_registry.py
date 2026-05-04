"""
routes_registry.py

This file defines registry-related API endpoints.

These endpoints expose safe registry information and guarded promotion behavior.
No model should be promoted to Production without passing the gate and receiving
human approval.
"""

from fastapi import APIRouter

from service.registry.mlflow_registry import MLflowRegistry

router = APIRouter()

registry = MLflowRegistry(
    registered_model_name="bank_marketing_classifier"
)


@router.get("/registry/production")
def get_production_model():
    """
    Return the current Production model metadata.
    """

    return registry.get_current_production_model()


@router.post("/registry/promote")
def promote_model(candidate_version: str, approved: bool = False):
    """
    Guarded promotion endpoint.

    In the final project, this should only be called after the agent receives
    human approval from the dashboard.
    """

    return registry.promote_to_production(
        candidate_version=candidate_version,
        approved=approved
    )