"""
test_agent_api.py

Tests the current agent API endpoints.

The current agent exposes:
- GET /health
- POST /drift

The /drift endpoint accepts the current DriftEvent contract and returns an
investigation response.
"""

from fastapi.testclient import TestClient

from agent.main import app

client = TestClient(app)


def test_agent_health() -> None:
    """
    Agent health endpoint should return ok.
    """

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_drift_webhook_creates_investigation() -> None:
    """
    Drift webhook should accept a valid current drift event payload.
    """

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

    response = client.post("/drift", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "accepted"
    assert body["investigation_id"] == "event-test-1"
    assert body["recommended_action"] in [
        "enqueue_replay_test",
        "request_human_approval_for_retrain",
        "continue_monitoring",
    ]
    assert isinstance(body["human_approval_needed"], bool)