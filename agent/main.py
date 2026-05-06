"""
main.py

Agent service entrypoint.

Receives drift events from the model platform and runs the integrated
investigation workflow.
"""

from fastapi import FastAPI

from agent.schemas.drift_event import DriftEvent
from agent.integration.investigation_runner import InvestigationRunner


app = FastAPI(
    title="Drift Triage Agent Service",
    description="Agent service for handling drift investigations.",
    version="0.1.0",
)

runner = InvestigationRunner()


@app.get("/health")
def health_check():
    """
    Basic health endpoint.
    """

    return {
        "status": "ok",
        "service": "agent_service",
    }


@app.post("/drift")
def receive_drift_event(event: DriftEvent):
    """
    Receive drift event and run integrated investigation flow.
    """

    investigation = runner.run(event.model_dump())

    return {
        "status": "accepted",
        "investigation_id": investigation["investigation_id"],
        "recommended_action": investigation["recommended_action"],
        "human_approval_needed": investigation["human_approval_needed"],
    }