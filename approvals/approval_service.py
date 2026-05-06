"""
approval_service.py

This file contains the human-in-the-loop approval workflow.

It creates approval requests, lists pending approvals, and records human
approval or rejection decisions.
"""

from uuid import uuid4

from approvals.approval_models import ApprovalRequest, ApprovalDecision
from approvals.approval_store import ApprovalStore


class ApprovalService:
    """
    Coordinates approval request and decision handling.
    """

    def __init__(self):
        self.store = ApprovalStore()

    def create_request(
        self,
        investigation_id: str,
        action_type: str,
        reason: str,
        payload: dict,
        target_model_version: str | None = None,
    ) -> dict:
        """
        Create a new pending approval request.
        """

        approval = ApprovalRequest(
            approval_id=str(uuid4()),
            investigation_id=investigation_id,
            action_type=action_type,
            target_model_version=target_model_version,
            reason=reason,
            payload=payload,
            status="pending",
        )

        self.store.append(approval.model_dump())

        return approval.model_dump()

    def list_pending(self) -> list[dict]:
        """
        Return all pending approval requests.
        """

        return self.store.pending()

    def decide(self, decision: ApprovalDecision) -> dict:
        """
        Record a human approval or rejection decision.

        This local JSONL implementation appends a decision record instead of
        rewriting old records. The dashboard can read the latest decision later.
        """

        decision_record = decision.model_dump()
        decision_record["status"] = "approved" if decision.approved else "rejected"

        self.store.append(decision_record)

        return decision_record