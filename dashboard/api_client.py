"""
api_client.py

Helper functions for the Streamlit dashboard.
"""

import os
import requests

MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://127.0.0.1:8000")
AGENT_SERVICE_URL = os.getenv("AGENT_SERVICE_URL", "http://127.0.0.1:8001")


def _get(url: str) -> dict:
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def _post(url: str, payload: dict | None = None, params: dict | None = None) -> dict:
    response = requests.post(url, json=payload, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


def get_health() -> dict:
    return _get(f"{MODEL_SERVICE_URL}/health")


def get_registry_state() -> dict:
    return _get(f"{MODEL_SERVICE_URL}/registry/production")


def get_drift_status() -> dict:
    return _get(f"{MODEL_SERVICE_URL}/drift/status")


def post_drift_check() -> dict:
    return _post(f"{MODEL_SERVICE_URL}/drift/check")


def get_queue_status() -> dict:
    return _get(f"{MODEL_SERVICE_URL}/queue/status")


def get_pending_approvals() -> dict:
    return _get(f"{MODEL_SERVICE_URL}/approvals/pending")


def submit_approval_decision(
    approval_id: str,
    approved: bool,
    approved_by: str,
    decision_reason: str,
) -> dict:
    return _post(
        f"{MODEL_SERVICE_URL}/approvals/decision",
        payload={
            "approval_id": approval_id,
            "approved": approved,
            "approved_by": approved_by,
            "decision_reason": decision_reason,
        },
    )


def post_predict(payload: dict) -> dict:
    return _post(f"{MODEL_SERVICE_URL}/predict", payload=payload)


def post_demo_reset() -> dict:
    return _post(f"{MODEL_SERVICE_URL}/demo/reset")


def get_investigations() -> dict:
    return _get(f"{AGENT_SERVICE_URL}/investigations")
