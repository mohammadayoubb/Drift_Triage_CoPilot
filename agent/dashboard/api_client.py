"""
api_client.py

This file contains helper functions for the Streamlit dashboard.

The dashboard reads state from:
- model service
- registry endpoints
- drift endpoints
- queue endpoints
- approval endpoints
"""

import os
import requests


MODEL_SERVICE_URL = os.getenv(
    "MODEL_SERVICE_URL",
    "http://127.0.0.1:8000"
)


def get_json(path: str) -> dict:
    """
    Send GET request to the model service.
    """

    response = requests.get(
        f"{MODEL_SERVICE_URL}{path}",
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def post_json(path: str, payload: dict | None = None, params: dict | None = None) -> dict:
    """
    Send POST request to the model service.
    """

    response = requests.post(
        f"{MODEL_SERVICE_URL}{path}",
        json=payload,
        params=params,
        timeout=10
    )
    response.raise_for_status()
    return response.json()


def get_health() -> dict:
    """
    Get model service health.
    """

    return get_json("/health")


def get_registry_state() -> dict:
    """
    Get current Production model state.
    """

    return get_json("/registry/production")


def get_drift_status() -> dict:
    """
    Get current drift report.
    """

    return get_json("/drift/status")


def get_queue_status() -> dict:
    """
    Get queue and DLQ depth.
    """

    return get_json("/queue/status")


def get_pending_approvals() -> dict:
    """
    Get pending human approvals.
    """

    return get_json("/approvals/pending")


def submit_approval_decision(
    approval_id: str,
    approved: bool,
    approved_by: str,
    decision_reason: str,
) -> dict:
    """
    Submit a human approval decision.
    """

    payload = {
        "approval_id": approval_id,
        "approved": approved,
        "approved_by": approved_by,
        "decision_reason": decision_reason,
    }

    return post_json("/approvals/decision", payload=payload)