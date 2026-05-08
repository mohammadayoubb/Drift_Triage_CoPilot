"""
routes_approvals.py

This file exposes human-in-the-loop approval endpoints.

The dashboard will use these endpoints to:
- list pending approvals
- create approval requests
- approve or reject actions
"""

from fastapi import APIRouter

from approvals.approval_models import ApprovalDecision
from approvals.approval_service import ApprovalService

router = APIRouter()

approval_service = ApprovalService()


@router.get("/approvals/pending")
def list_pending_approvals():
    """
    Return pending approval requests.
    """

    return {
        "pending": approval_service.list_pending()
    }


@router.post("/approvals/request")
def create_approval_request(
    investigation_id: str,
    action_type: str,
    reason: str,
    target_model_version: str | None = None,
):
    """
    Create a pending approval request.
    """

    approval = approval_service.create_request(
        investigation_id=investigation_id,
        action_type=action_type,
        reason=reason,
        target_model_version=target_model_version,
        payload={
            "investigation_id": investigation_id,
            "action_type": action_type,
            "target_model_version": target_model_version,
        },
    )

    return approval


@router.post("/approvals/decision")
def submit_approval_decision(decision: ApprovalDecision):
    """
    Record a human approval/rejection decision.
    """

    return approval_service.decide(decision)


@router.get("/approvals/history")
def get_approval_history():
    """
    Return all approvals with their final resolved status (most recent first).
    Each record merges the original request with any decision fields.
    """

    return {"history": approval_service.list_history()}