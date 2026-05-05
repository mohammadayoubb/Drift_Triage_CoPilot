"""
routes_drift.py

This file exposes drift monitoring endpoints.

It allows the model platform to:
- Check current drift status
- Optionally create a drift event
- Optionally notify the agent when severity changes
"""

from fastapi import APIRouter

from service.drift.drift_service import DriftService

router = APIRouter()


@router.get("/drift/status")
def drift_status():
    """
    Build and return the current drift report without notifying the agent.
    """

    drift_service = DriftService()

    return drift_service.check_drift(notify_agent=False)


@router.post("/drift/check")
def check_drift_and_notify():
    """
    Build a drift report and notify the agent only if severity changed.

    This is the endpoint that represents the platform-to-agent drift trigger.
    """

    drift_service = DriftService()

    return drift_service.check_drift(notify_agent=True)