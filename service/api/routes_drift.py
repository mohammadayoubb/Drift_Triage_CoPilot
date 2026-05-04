"""
routes_drift.py

This file exposes a local drift report endpoint.

It is useful for testing drift detection before connecting the agent webhook.
"""

from fastapi import APIRouter

router = APIRouter()


@router.get("/drift/status")
def drift_status():
    """
    Placeholder drift status endpoint.

    The full implementation will later load reference data and recent prediction
    data, run the drift monitor, and optionally notify the agent.
    """

    return {
        "status": "not_configured",
        "message": "Drift monitor endpoint is ready, but reference/live data wiring is not complete yet."
    }