"""
approval_service.py

This file contains the human-in-the-loop approval workflow.

It creates approval requests, lists pending approvals, and records human
approval or rejection decisions.
"""

import logging
from datetime import datetime, timezone
from uuid import uuid4

from approvals.approval_models import ApprovalRequest, ApprovalDecision
from approvals.approval_store import ApprovalStore
from async_queue.idempotency import IdempotencyStore
from async_queue.job_models import QueueJob
from async_queue.producer import QueueProducer

logger = logging.getLogger(__name__)

# Maps approval action_type → queue job_type
_JOB_TYPE_MAP: dict[str, str] = {
    "replay_test_set_then_open_retrain_candidate": "replay_test",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
        Returns the approval dict with a ``created`` boolean:
        - True  → freshly created
        - False → a pending approval for this investigation already existed
        """

        existing = self.store.find_pending_by_investigation_id(investigation_id, action_type=action_type)
        if existing:
            return {**existing, "created": False}

        approval = ApprovalRequest(
            approval_id=str(uuid4()),
            investigation_id=investigation_id,
            action_type=action_type,
            target_model_version=target_model_version,
            reason=reason,
            payload=payload,
            status="pending",
            created_at_utc=_utc_now(),
        )

        self.store.append(approval.model_dump())

        return {**approval.model_dump(), "created": True}

    def list_pending(self) -> list[dict]:
        """
        Return all pending approval requests.
        """

        return self.store.pending()

    def list_history(self) -> list[dict]:
        """
        Return all approvals with their final resolved status.
        """

        return self.store.history()

    def decide(self, decision: ApprovalDecision) -> dict:
        """
        Record a human approval or rejection decision and, when approved,
        dispatch the corresponding async job to Redis.

        Returns early with already_decided=True if the approval was already
        processed — prevents duplicate queue jobs on button double-click.
        """

        existing = self.store.find_decision_by_approval_id(decision.approval_id)
        if existing:
            logger.info(
                "Approval already decided: approval_id=%s status=%s",
                decision.approval_id,
                existing["status"],
            )
            return {
                "already_decided": True,
                "approval_id": decision.approval_id,
                "status": existing["status"],
                "message": f"Approval already {existing['status']}. No action taken.",
            }

        # Look up original request first so we have investigation_id and action_type
        # before deciding whether to queue a job. This lets us record job_queued
        # with the correct value (the JSONL store is append-only; we cannot patch
        # a record after writing it).
        original = self.store.find_by_approval_id(decision.approval_id)
        investigation_id = (original or {}).get("investigation_id", "unknown")
        action_type = (original or {}).get("action_type", "")

        job_queued = False

        if decision.approved:
            logger.info(
                "Approval approved: approval_id=%s investigation_id=%s action_type=%s",
                decision.approval_id,
                investigation_id,
                action_type,
            )

            job_type = _JOB_TYPE_MAP.get(action_type)
            if job_type:
                idempotency_key = f"{investigation_id}-replay-test"
                # Use "queued:" namespace to track enqueued-but-not-yet-processed
                # jobs. The worker uses the default "idempotency:" namespace to
                # track completed jobs. Keeping them separate prevents the worker
                # from skipping a job that was queued but not yet processed.
                idem = IdempotencyStore(namespace="queued")

                if idem.already_seen(idempotency_key):
                    logger.info(
                        "replay_test job already queued (idempotency): investigation_id=%s",
                        investigation_id,
                    )
                else:
                    job = QueueJob(
                        job_id=str(uuid4()),
                        job_type=job_type,
                        payload={
                            "approval_id": decision.approval_id,
                            "investigation_id": investigation_id,
                            "action_type": action_type,
                        },
                        idempotency_key=idempotency_key,
                    )
                    QueueProducer().enqueue(job)
                    idem.mark_seen(idempotency_key)
                    job_queued = True
                    logger.info(
                        "replay_test job queued: investigation_id=%s job_id=%s",
                        investigation_id,
                        job.job_id,
                    )
        else:
            logger.info(
                "Approval rejected: approval_id=%s investigation_id=%s",
                decision.approval_id,
                investigation_id,
            )
            logger.info(
                "No job queued: approval rejected for investigation_id=%s",
                investigation_id,
            )

        # Write the decision record with the actual job_queued value now that
        # we know whether the queue call succeeded.
        decision_record = decision.model_dump()
        decision_record["status"] = "approved" if decision.approved else "rejected"
        decision_record["decided_at_utc"] = _utc_now()
        decision_record["job_queued"] = job_queued
        self.store.append(decision_record)

        return decision_record