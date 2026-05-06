# tests/agent/test_agent_graph.py

from agent.graph.runner import run_investigation_graph
from agent.schemas.drift_event import DriftEvent


def make_critical_event() -> DriftEvent:
    return DriftEvent(
        event_type="drift_report.severity_changed",
        version="v1",
        severity="critical",
        created_at_utc="2026-05-06T07:00:00+00:00",
        recent_count=40,
        window_size=40,
        report={
            "severity": "critical",
            "status": "ok",
            "numeric_drift": {
                "euribor3m": {
                    "psi": 0.4,
                    "severity": "critical",
                    "recent_count": 40,
                }
            },
            "categorical_drift": {
                "job": {
                    "chi_square": 100.0,
                    "p_value": 0.001,
                    "severity": "critical",
                    "recent_count": 40,
                }
            },
            "output_drift": {
                "absolute_delta": 0.25,
                "severity": "critical",
            },
        },
    )


def test_agent_graph_recommends_action_for_critical_drift() -> None:
    event = make_critical_event()

    final_state = run_investigation_graph(
        event=event,
        investigation_id="test-investigation",
        thread_id="test-thread-critical",
    )

    assert final_state["status"] == "action_recommended"
    assert final_state["triage_result"]["severity"] == "critical"
    assert (
        final_state["recommended_action"]["action_type"]
        == "replay_test_set_then_open_retrain_candidate"
    )
    assert "comms_summary" in final_state