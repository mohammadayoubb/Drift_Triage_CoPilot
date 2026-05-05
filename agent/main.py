"""
main.py

This is the entry point for the agent service.

For now, it receives drift webhooks and creates investigations.
Later, this will be upgraded into a LangGraph supervisor agent.
"""

from fastapi import FastAPI

from agent.schemas.drift_event import DriftEvent
from agent.storage.investigation_store import InvestigationStore


app = FastAPI(
    title="Drift Triage Agent Service",
    description="Agent service for handling drift investigations.",
    version="0.1.0"
)

investigation_store = InvestigationStore()


@app.get("/health")
def health_check():
    """
    Basic health endpoint for the agent service.
    """

    return {
        "status": "ok",
        "service": "agent_service"
    }


@app.post("/drift")
def receive_drift_event(event: DriftEvent):
    """
    Receive drift event from the model platform and create an investigation.
    """

    investigation = {
        "investigation_id": event.event_id,
        "event_id": event.event_id,
        "model_name": event.model_name,
        "model_version": event.model_version,
        "severity": event.severity,
        "affected_features": event.affected_features,
        "status": "opened",
        "recommended_action": "pending_triage"
    }

    investigation_store.append(investigation)

    return {
        "status": "accepted",
        "investigation_id": investigation["investigation_id"]
    }