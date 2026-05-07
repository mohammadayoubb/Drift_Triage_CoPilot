"""
test_agent_api.py

Tests the LangGraph agent API endpoints (src/agent/).

The agent exposes:
- GET /health
- POST /webhooks/drift
- GET /investigations
"""

from fastapi.testclient import TestClient

from agent.main import app

client = TestClient(app)


def test_agent_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "drift-triage-agent"


def test_drift_webhook_creates_investigation() -> None:
    """Drift webhook should accept a valid event and return an investigation response."""

    payload = {
        "contract_version": "1.0",
        "event_id": "event-test-1",
        "event_time": "2026-05-06T07:00:00+00:00",
        "model_name": "bank_marketing_classifier",
        "model_version": "v1",
        "production_model_version": "v1",
        "severity": "HIGH",
        "previous_severity": "LOW",
        "affected_features": ["euribor3m"],
        "drift_type": "numeric",
        "numeric_drift_summary": {
            "euribor3m": {
                "psi": 0.2,
                "severity": "HIGH",
            }
        },
        "categorical_drift_summary": {},
        "output_drift_summary": {},
        "rolling_window_size": 40,
        "reference_stats_version": "reference_v1",
    }

    response = client.post("/webhooks/drift", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert "investigation_id" in body
    assert "thread_id" in body
    assert body["severity"] == "HIGH"
    assert body["status"] in ("action_recommended", "resolved")
    assert isinstance(body["requires_human_approval"], bool)
    assert body["recommended_action"] is not None
