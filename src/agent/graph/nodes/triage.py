# src/agent/graph/nodes/triage.py

from typing import Any

from agent.graph.state import AgentState


def _top_numeric_drift(report: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    numeric_drift = report.get("numeric_drift", {})

    rows = []
    for feature, payload in numeric_drift.items():
        psi = payload.get("psi")

        if psi is None:
            continue

        rows.append(
            {
                "feature": feature,
                "metric": "psi",
                "value": psi,
                "severity": payload.get("severity", "none"),
            }
        )

    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _top_categorical_drift(report: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    categorical_drift = report.get("categorical_drift", {})

    rows = []
    for feature, payload in categorical_drift.items():
        chi_square = payload.get("chi_square")
        p_value = payload.get("p_value")

        if chi_square is None:
            continue

        rows.append(
            {
                "feature": feature,
                "metric": "chi_square",
                "value": chi_square,
                "p_value": p_value,
                "severity": payload.get("severity", "none"),
            }
        )

    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def triage_node(state: AgentState) -> dict[str, Any]:
    """
    Triage drift severity and identify the most important signals.

    This is deterministic for now so it is easy to test in CI.
    Later, an LLM can summarize the same structured report.
    """
    drift_event = state["drift_event"]
    report = drift_event.get("report", {})
    severity = drift_event.get("severity", "none")

    output_drift = report.get("output_drift", {})

    triage_result = {
        "severity": severity,
        "recent_count": drift_event.get("recent_count"),
        "window_size": drift_event.get("window_size"),
        "top_numeric_drift": _top_numeric_drift(report),
        "top_categorical_drift": _top_categorical_drift(report),
        "output_drift": output_drift,
    }

    return {
        "triage_result": triage_result,
        "status": "triaged",
    }