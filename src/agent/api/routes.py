# src/agent/api/routes.py

import logging

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