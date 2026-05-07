# src/agent/graph/state.py

from typing import Any, Literal, TypedDict

AgentStatus = Literal[
    "opened",
    "triaged",
    "action_recommended",
    "approval_pending",
    "resolved",
    "failed",
]


class AgentState(TypedDict, total=False):
    investigation_id: str
    thread_id: str
    drift_event: dict[str, Any]
    severity: str

    # Supervisor routing target — set by supervisor_node, consumed by route_supervisor
    next: str

    triage_result: dict[str, Any]
    recommended_action: dict[str, Any]
    comms_summary: str

    requires_human_approval: bool
    status: AgentStatus
