# src/agent/schemas/investigation.py

from typing import Any, Literal

from pydantic import BaseModel, Field

InvestigationStatus = Literal[
    "opened",
    "triaged",
    "action_recommended",
    "approval_pending",
    "resolved",
    "failed",
]


class InvestigationRecord(BaseModel):
    investigation_id: str
    thread_id: str
    status: InvestigationStatus
    severity: str
    created_at_utc: str
    updated_at_utc: str
    drift_event: dict[str, Any]
    triage_result: dict[str, Any] | None = None
    recommended_action: dict[str, Any] | None = None
    comms_summary: str | None = None
    requires_human_approval: bool = False


class DriftWebhookResponse(BaseModel):
    investigation_id: str
    thread_id: str
    status: InvestigationStatus
    severity: str
    requires_human_approval: bool
    recommended_action: dict[str, Any] | None = None
    comms_summary: str | None = None


class InvestigationListResponse(BaseModel):
    investigations: list[InvestigationRecord] = Field(default_factory=list)