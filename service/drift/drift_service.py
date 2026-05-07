"""
drift_service.py

This file coordinates drift reporting, severity-change detection, and optional
agent webhook notification.
"""

from service.config.settings import Settings
from service.drift.report_builder import DriftReportBuilder
from service.drift.event_builder import DriftEventBuilder
from service.drift.webhook_client import DriftWebhookClient
from service.storage.drift_state_store import DriftStateStore


class DriftService:
    """
    High-level drift service used by API routes or background jobs.
    """

    def __init__(self):
        self.report_builder = DriftReportBuilder(
            window_size=Settings.DRIFT_WINDOW_SIZE
        )
        self.event_builder = DriftEventBuilder()
        self.state_store = DriftStateStore()
        self.webhook_client = DriftWebhookClient(
            agent_webhook_url=Settings.AGENT_WEBHOOK_URL
        )

    def check_drift(self, notify_agent: bool = False) -> dict:
        """
        Build a drift report and optionally notify agent if severity changed.
        """

        report = self.report_builder.build()

        if report.get("status") != "ok":
            return {
                "report": report,
                "event_sent": False,
                "reason": "Drift report is not ready."
            }

        current_severity = report.get("severity")
        previous_severity = self.state_store.get_last_severity()

        severity_changed = current_severity != previous_severity

        if not severity_changed:
            return {
                "report": report,
                "event_sent": False,
                "reason": "Severity did not change."
            }

        event = self.event_builder.build_event(
            report=report,
            model_name=Settings.MODEL_NAME,
            model_version="v1",
            previous_severity=previous_severity
        )

        if not notify_agent:
            # GET /drift/status path — report only, never update state or send webhook
            return {
                "report": report,
                "event_sent": False,
                "event": event,
                "webhook_response": {"skipped": True, "reason": "notify_agent=False"},
            }

        # POST /drift/check path — attempt delivery first, save state only on success.
        # If saved before delivery and the webhook fails, severity gets stuck and
        # subsequent retries will see no change and never re-fire.
        try:
            webhook_response = self.webhook_client.send_drift_event(event)
            self.state_store.set_last_severity(current_severity)
            event_sent = True
        except Exception as exc:
            webhook_response = {
                "error": str(exc),
                "delivered": False,
                "hint": "Agent service unreachable. Severity state not saved — next check will retry.",
            }
            event_sent = False

        return {
            "report": report,
            "event_sent": event_sent,
            "event": event,
            "webhook_response": webhook_response,
        }