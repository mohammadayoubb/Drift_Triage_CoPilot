# src/agent/graph/nodes/comms.py

from typing import Any

from agent.graph.state import AgentState


def comms_node(state: AgentState) -> dict[str, Any]:
    """
    Create a compact human-readable investigation summary.
    """
    investigation_id = state["investigation_id"]
    severity = state.get("severity", "none")
    triage_result = state.get("triage_result", {})
    recommended_action = state.get("recommended_action", {})

    top_numeric = triage_result.get("top_numeric_drift", [])
    top_categorical = triage_result.get("top_categorical_drift", [])

    numeric_summary = ", ".join(
        item["feature"] for item in top_numeric[:3]
    ) or "none"

    categorical_summary = ", ".join(
        item["feature"] for item in top_categorical[:3]
    ) or "none"

    comms_summary = (
        f"Investigation {investigation_id}: severity={severity}. "
        f"Top numeric drift: {numeric_summary}. "
        f"Top categorical drift: {categorical_summary}. "
        f"Recommended action: {recommended_action.get('action_type')}."
    )

    return {
        "comms_summary": comms_summary,
    }