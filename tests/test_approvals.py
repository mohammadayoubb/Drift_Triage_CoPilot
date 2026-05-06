"""
test_approvals.py

Tests human-in-the-loop approval schemas and local approval store behavior.
"""

from approvals.approval_models import ApprovalRequest, ApprovalDecision
from approvals.approval_store import ApprovalStore


def test_approval_request_valid():
    approval = ApprovalRequest(
        approval_id="a1",
        investigation_id="i1",
        action_type="retrain",
        target_model_version="1",
        reason="Critical drift requires review.",
        payload={"severity": "CRITICAL"},
    )

    assert approval.status == "pending"
    assert approval.action_type == "retrain"


def test_approval_decision_valid():
    decision = ApprovalDecision(
        approval_id="a1",
        approved=True,
        approved_by="demo_user",
        decision_reason="Approved for test.",
    )

    assert decision.approved is True
    assert decision.approved_by == "demo_user"


def test_approval_store_pending(tmp_path):
    store_path = tmp_path / "approvals.jsonl"
    store = ApprovalStore(path=str(store_path))

    store.append({"approval_id": "a1", "status": "pending"})
    store.append({"approval_id": "a2", "status": "approved"})

    pending = store.pending()

    assert len(pending) == 1
    assert pending[0]["approval_id"] == "a1"