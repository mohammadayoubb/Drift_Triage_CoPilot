# src/agent/graph/nodes/action.py

from typing import Any

from agent.graph.state import AgentState


def action_node(state: AgentState) -> dict[str, Any]:
    """
    Recommend the next action.

    Important:
    - This does not execute retrain/rollback yet.
    - Slow actions will later go through Redis worker.
    - Anything touching Production must require human approval.
    """
    severity = state.get("severity", "none")
    triage_result = state.get("triage_result", {})

    if severity == "critical":
        recommended_action = {
            "action_type": "replay_test_set_then_open_retrain_candidate",
            "priority": "high",
            "reason": "Critical drift detected. Replay test set and prepare retraining candidate.",
            "queue_task": "replay_test_set",
            "touches_production": False,
        }

        return {
            "recommended_action": recommended_action,
            "requires_human_approval": False,
            "status": "action_recommended",
        }

    if severity == "warning":
        recommended_action = {
            "action_type": "monitor_and_replay_test_set",
            "priority": "medium",
            "reason": "Warning-level drift detected. Replay test set before deciding on retrain.",
            "queue_task": "replay_test_set",
            "touches_production": False,
        }

        return {
            "recommended_action": recommended_action,
            "requires_human_approval": False,
            "status": "action_recommended",
        }

    recommended_action = {
        "action_type": "no_action",
        "priority": "low",
        "reason": "No meaningful drift detected.",
        "queue_task": None,
        "touches_production": False,
    }

    return {
        "recommended_action": recommended_action,
        "requires_human_approval": False,
        "status": "resolved",
    }