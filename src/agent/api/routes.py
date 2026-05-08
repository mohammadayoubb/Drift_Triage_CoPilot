# src/agent/api/routes.py

import logging
import os

import requests as http_client
from fastapi import APIRouter, HTTPException, status

from ..graph.runner import run_investigation_graph
from ..persistence.investigation_store import (
    create_investigation,
    get_investigation,
    list_investigations,
    update_investigation,
)
from ..schemas.drift_event import DriftEvent
from ..schemas.investigation import (
    DriftWebhookResponse,
    InvestigationListResponse,
    InvestigationRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_MODEL_SERVICE_URL = os.getenv("MODEL_SERVICE_URL", "http://model-service:8000")


_APPROVAL_REASON = (
    "Critical drift detected. Agent recommends replaying the test set "
    "and opening a retraining candidate if model behavior degraded."
)


def _request_approval(investigation: InvestigationRecord) -> None:
    """
    POST a pending approval request to the model service.
    Non-fatal: logs on failure but does not raise.
    Deduplication is enforced server-side; the ``created`` flag in the
    response tells us whether a new record was written.
    """
    action = investigation.recommended_action or {}
    action_type = action.get(
        "action_type", "replay_test_set_then_open_retrain_candidate"
    )

    url = f"{_MODEL_SERVICE_URL}/approvals/request"
    # Omit target_model_version entirely when null — passing None as a query
    # param would serialise to the string "None" in the URL.
    params = {
        "investigation_id": investigation.investigation_id,
        "action_type": action_type,
        "reason": _APPROVAL_REASON,
    }

    try:
        resp = http_client.post(url, params=params, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        if data.get("created", True):
            logger.info(
                "Approval request created: investigation_id=%s action_type=%s",
                investigation.investigation_id,
                action_type,
            )
        else:
            logger.info(
                "Approval request already exists: investigation_id=%s",
                investigation.investigation_id,
            )
    except Exception as exc:
        logger.warning(
            "Approval request failed (non-fatal): investigation_id=%s error=%s",
            investigation.investigation_id,
            exc,
        )


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "drift-triage-agent",
    }


@router.post("/webhooks/drift", response_model=DriftWebhookResponse)
def receive_drift_webhook(event: DriftEvent) -> DriftWebhookResponse:
    """
    Receive drift event from model service and open an investigation.
    """
    try:
        investigation = create_investigation(event)

        logger.info(
            "Investigation created: investigation_id=%s severity=%s",
            investigation.investigation_id,
            event.severity,
        )

        final_state = run_investigation_graph(
            event=event,
            investigation_id=investigation.investigation_id,
            thread_id=investigation.thread_id,
        )

        # When action_node calls interrupt() for CRITICAL drift, graph.invoke()
        # returns before the node's return statement executes.  The interrupt
        # payload (which carries recommended_action) is in __interrupt__, not in
        # the top-level state keys.
        interrupts = final_state.get("__interrupt__", ())
        if interrupts:
            iv = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
            update_kwargs = {
                "status": "approval_pending",
                "requires_human_approval": True,
                "recommended_action": iv.get("recommended_action") if isinstance(iv, dict) else None,
                "triage_result": final_state.get("triage_result"),
                "comms_summary": None,
            }
        else:
            update_kwargs = {
                "status": final_state.get("status", "action_recommended"),
                "triage_result": final_state.get("triage_result"),
                "recommended_action": final_state.get("recommended_action"),
                "comms_summary": final_state.get("comms_summary"),
                "requires_human_approval": final_state.get("requires_human_approval", False),
            }

        updated = update_investigation(
            investigation.investigation_id,
            update_kwargs,
        )

        if updated.requires_human_approval:
            _request_approval(updated)

        return DriftWebhookResponse(
            investigation_id=updated.investigation_id,
            thread_id=updated.thread_id,
            status=updated.status,
            severity=updated.severity,
            requires_human_approval=updated.requires_human_approval,
            recommended_action=updated.recommended_action,
            comms_summary=updated.comms_summary,
        )

    except Exception as exc:
        logger.exception("Failed to process drift webhook")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "drift_webhook_processing_failed",
                "message": "Agent failed to process drift webhook.",
                "detail": str(exc),
            },
        ) from exc


@router.get("/investigations", response_model=InvestigationListResponse)
def get_investigations() -> InvestigationListResponse:
    return InvestigationListResponse(
        investigations=list_investigations()
    )


@router.get("/investigations/{investigation_id}", response_model=InvestigationRecord)
def get_investigation_by_id(investigation_id: str) -> InvestigationRecord:
    try:
        return get_investigation(investigation_id)

    except KeyError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "investigation_not_found",
                "message": str(exc),
            },
        ) from exc