"""
routes_registry.py

This file defines registry-related API endpoints.

It exposes:
- Current Production model status
- Guarded promotion endpoint

The promotion endpoint uses a structured request body instead of loose query
parameters.
"""

from fastapi import APIRouter

from service.registry.mlflow_registry import MLflowRegistry
from service.config.settings import Settings
from service.validation.registry_schema import PromotionRequest, PromotionResponse

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


@router.post("/registry/promote", response_model=PromotionResponse)
def promote_model(request: PromotionRequest):
    """
    Guarded promotion endpoint.

    Production promotion requires explicit approval information.
    """

    return registry.promote_to_production(
        candidate_version=request.candidate_version,
        approved=request.approved
    )