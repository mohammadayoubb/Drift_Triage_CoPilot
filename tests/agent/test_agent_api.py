# tests/agent/test_agent_api.py

from fastapi.testclient import TestClient

from agent.main import app

client = TestClient(app)


def test_agent_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_drift_webhook_creates_investigation() -> None:
    payload = {
        "event_type": "drift_report.severity_changed",
        "version": "v1",
        "severity": "warning",
        "created_at_utc": "2026-05-06T07:00:00+00:00",
        "recent_count": 40,
        "window_size": 40,
        "report": {
            "severity": "warning",
            "status": "ok",
            "numeric_drift": {
                "euribor3m": {
                    "psi": 0.2,
                    "severity": "warning",
                    "recent_count": 40,
                }
            },
            "categorical_drift": {},
            "output_drift": {
                "absolute_delta": 0.12,
                "severity": "warning",
            },
        },
    }

    response = client.post("/webhooks/drift", json=payload)

    assert response.status_code == 200

    body = response.json()

    assert body["investigation_id"].startswith("inv_")
    assert body["severity"] == "warning"
    assert body["recommended_action"]["action_type"] == "monitor_and_replay_test_set"
    assert body["comms_summary"] is not None