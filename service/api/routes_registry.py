"""
routes_registry.py

This file defines registry-related API endpoints.

It exposes:
- Current Production model status
- Guarded promotion endpoint
"""

from fastapi import APIRouter

from service.registry.mlflow_registry import MLflowRegistry
from service.config.settings import Settings

router = APIRouter()

registry = MLflowRegistry(
    registered_model_name=Settings.MODEL_NAME
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

    This endpoint must only allow Production changes after human approval.
    """

    return registry.promote_to_production(
        candidate_version=candidate_version,
        approved=approved
    )