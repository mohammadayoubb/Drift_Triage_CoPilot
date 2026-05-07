# tests/agent/test_trajectory_snapshots.py
"""
Regression tests for LangGraph agent routing logic.

Runs without any LLM API key — both the LLM calls (triage, comms) and
the HIL interrupt (action) are mocked so the full graph executes in CI.
"""

from unittest.mock import MagicMock, patch

from agent.graph.supervisor import build_supervisor_graph

# ── helpers ──────────────────────────────────────────────────────────────────

_LLM_MOCK = MagicMock()
_LLM_MOCK.return_value.invoke.return_value.content = "Mocked LLM analysis for CI."

_INTERRUPT_APPROVED = {"approved": True}


def _build_initial_state(severity: str) -> dict:
    return {
        "investigation_id": f"test-{severity}",
        "thread_id": f"thread-{severity}",
        "drift_event": {
            "severity": severity,
            "numeric_drift_summary": {},
            "categorical_drift_summary": {},
            "output_drift_summary": {},
            "affected_features": [],
            "rolling_window_size": 100,
        },
        "severity": severity,
        "status": "opened",
        "requires_human_approval": False,
    }


def _run(severity: str) -> dict:
    """
    Run the graph for a given severity with:
      - LLM calls patched to return a fixed string (no API key needed)
      - interrupt() patched to immediately return approval (no human needed)
    """
    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": f"thread-{severity}"}}

    with (
        patch("agent.graph.nodes.triage._llm_analysis", return_value="Mocked triage analysis."),
        patch("agent.graph.nodes.comms._llm_summary", return_value="Mocked comms summary for critical severity."),
        patch("agent.graph.nodes.action.interrupt", return_value=_INTERRUPT_APPROVED),
    ):
        return graph.invoke(_build_initial_state(severity), config=config)


# ── snapshot tests ────────────────────────────────────────────────────────────

def test_critical_trajectory():
    state = _run("critical")
    assert state["status"] == "resolved"
    assert state["recommended_action"]["action_type"] == "replay_test_set_then_open_retrain_candidate"
    assert state["recommended_action"]["queue_task"] == "replay_test"
    assert state["recommended_action"]["touches_production"] is True
    assert state["requires_human_approval"] is True
    assert "comms_summary" in state


def test_high_trajectory():
    state = _run("high")
    assert state["status"] == "resolved"
    assert state["recommended_action"]["action_type"] == "monitor_and_replay_test_set"
    assert state["recommended_action"]["queue_task"] == "replay_test"
    assert state["requires_human_approval"] is False
    assert "comms_summary" in state


def test_low_trajectory():
    state = _run("low")
    assert state["status"] == "resolved"
    assert state["recommended_action"]["action_type"] == "no_action"
    assert state["recommended_action"]["queue_task"] is None
    assert state["requires_human_approval"] is False
    assert "comms_summary" in state


def test_supervisor_routes_through_all_nodes():
    """Verify all three sub-agents are visited during a critical investigation."""
    state = _run("critical")
    assert "triage_result" in state
    assert state["triage_result"]["severity"] == "critical"
    assert "recommended_action" in state
    assert "comms_summary" in state
    assert "critical" in state["comms_summary"].lower()


def test_critical_interrupt_is_called():
    """Verify interrupt() is invoked exactly once for CRITICAL drift."""
    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "thread-interrupt-check"}}

    with (
        patch("agent.graph.nodes.triage._llm_analysis", return_value=""),
        patch("agent.graph.nodes.comms._llm_summary", return_value=""),
        patch("agent.graph.nodes.action.interrupt", return_value=_INTERRUPT_APPROVED) as mock_interrupt,
    ):
        graph.invoke(_build_initial_state("critical"), config=config)

    mock_interrupt.assert_called_once()
    call_kwargs = mock_interrupt.call_args[0][0]
    assert call_kwargs["investigation_id"] == "test-critical"
    assert "replay" in call_kwargs["reason"].lower()


def test_high_interrupt_is_not_called():
    """HIGH drift should NOT trigger the HIL interrupt."""
    graph = build_supervisor_graph()
    config = {"configurable": {"thread_id": "thread-high-no-interrupt"}}

    with (
        patch("agent.graph.nodes.triage._llm_analysis", return_value=""),
        patch("agent.graph.nodes.comms._llm_summary", return_value=""),
        patch("agent.graph.nodes.action.interrupt", return_value=_INTERRUPT_APPROVED) as mock_interrupt,
    ):
        graph.invoke(_build_initial_state("high"), config=config)

    mock_interrupt.assert_not_called()
