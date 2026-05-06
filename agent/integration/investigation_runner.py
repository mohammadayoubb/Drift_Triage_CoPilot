"""
investigation_runner.py

This file coordinates the integrated agent flow.

Flow:
drift event -> triage -> action decision -> approval request or queue dispatch
"""

from agent.storage.investigation_store import InvestigationStore
from agent.dispatch.queue_dispatcher import QueueDispatcher
from agent.approval.approval_client import ApprovalClient


class InvestigationRunner:
    """
    Runs the integrated investigation workflow.
    """

    def __init__(self):
        self.store = InvestigationStore()
        self.dispatcher = QueueDispatcher()
        self.approval_client = ApprovalClient()

    def run(self, event: dict) -> dict:
        """
        Process one drift event end-to-end.
        """

        severity = event.get("severity")
        investigation_id = event.get("event_id")
        affected_features = event.get("affected_features", [])

        if severity == "CRITICAL":
            recommended_action = "request_human_approval_for_retrain"
            human_approval_needed = True

            approval = self.approval_client.request_approval(
                investigation_id=investigation_id,
                action_type="retrain",
                reason="Critical drift detected. Human approval required before retrain/promotion.",
                target_model_version=None,
            )

            queue_result = None

        elif severity == "HIGH":
            recommended_action = "enqueue_replay_test"
            human_approval_needed = False

            queue_result = self.dispatcher.dispatch(
                job_type="replay_test",
                payload={
                    "investigation_id": investigation_id,
                    "model_name": event.get("model_name"),
                    "model_version": event.get("model_version"),
                    "affected_features": affected_features,
                },
                idempotency_key=f"replay:{investigation_id}",
            )

            approval = None

        else:
            recommended_action = "continue_monitoring"
            human_approval_needed = False
            queue_result = None
            approval = None

        investigation = {
            "investigation_id": investigation_id,
            "event_id": investigation_id,
            "model_name": event.get("model_name"),
            "model_version": event.get("model_version"),
            "severity": severity,
            "affected_features": affected_features,
            "status": "opened",
            "recommended_action": recommended_action,
            "human_approval_needed": human_approval_needed,
            "approval": approval,
            "queue_result": queue_result,
        }

        self.store.append(investigation)

        return investigation