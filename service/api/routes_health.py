"""
routes_health.py

This file provides a simple health check endpoint.

Used by:
- Docker
- Monitoring systems
- Debugging
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Basic service health endpoint.
    """

    return {
        "status": "ok",
        "service": "model_service"
    }