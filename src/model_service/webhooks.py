# src/model_service/webhooks.py

import logging
import os
from typing import Any

import httpx

from common.paths import LAST_DRIFT_SEVERITY_PATH

logger = logging.getLogger(__name__)

DEFAULT_AGENT_WEBHOOK_URL = "http://127.0.0.1:8001/webhooks/drift"


def get_agent_webhook_url() -> str:
    return os.getenv("AGENT_WEBHOOK_URL", DEFAULT_AGENT_WEBHOOK_URL)


def load_last_severity() -> str | None:
    if not LAST_DRIFT_SEVERITY_PATH.exists():
        return None

    return LAST_DRIFT_SEVERITY_PATH.read_text(encoding="utf-8").strip()


def save_last_severity(severity: str) -> None:
    LAST_DRIFT_SEVERITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAST_DRIFT_SEVERITY_PATH.write_text(severity, encoding="utf-8")


def should_emit_webhook(current_severity: str) -> bool:
    """
    Emit only when severity changes.

    Example:
    none -> warning: emit
    warning -> warning: do not emit
    warning -> critical: emit
    critical -> none: emit
    """
    previous_severity = load_last_severity()

    if previous_severity != current_severity:
        logger.info(
            "Drift severity changed: previous=%s current=%s",
            previous_severity,
            current_severity,
        )
        return True

    logger.info("Drift severity unchanged: severity=%s", current_severity)
    return False


def build_drift_event(report: dict[str, Any]) -> dict[str, Any]:
    """
    Build the event payload sent to the agent.

    Later we will formalize this in contracts/drift_event.v1.json.
    """
    return {
        "event_type": "drift_report.severity_changed",
        "version": "v1",
        "severity": report["severity"],
        "created_at_utc": report["created_at_utc"],
        "recent_count": report["recent_count"],
        "window_size": report["window_size"],
        "report": report,
    }


def emit_drift_webhook_if_needed(report: dict[str, Any]) -> bool:
    """
    Emit a drift webhook if severity changed.

    Returns True if webhook was emitted successfully.
    Returns False if no webhook was needed or delivery failed.
    """
    current_severity = report.get("severity", "none")

    if not should_emit_webhook(current_severity):
        return False

    save_last_severity(current_severity)

    event_payload = build_drift_event(report)
    webhook_url = get_agent_webhook_url()

    try:
        response = httpx.post(
            webhook_url,
            json=event_payload,
            timeout=5.0,
        )
        response.raise_for_status()

    except httpx.HTTPError as exc:
        logger.warning(
            "Failed to emit drift webhook to %s: %s",
            webhook_url,
            exc,
        )
        return False

    logger.info(
        "Drift webhook emitted successfully: severity=%s url=%s",
        current_severity,
        webhook_url,
    )

    return True