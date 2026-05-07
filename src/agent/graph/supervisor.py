# src/agent/graph/supervisor.py

from typing import Any, Literal

from langgraph.graph import END, START, StateGraph

from .nodes.action import action_node
from .nodes.comms import comms_node
from .nodes.triage import triage_node
from .state import AgentState
from ..persistence.checkpoints import get_checkpointer


def supervisor_node(state: AgentState) -> dict[str, Any]:
    """
    Central dispatcher. Decides which sub-agent to call next based on
    the current investigation status. Deterministic and regression-testable
    without an LLM key.
    """
    status = state.get("status", "opened")

    if status == "opened":
        return {"next": "triage"}
    if status == "triaged":
        return {"next": "action"}
    if status == "action_recommended":
        return {"next": "comms"}
    return {"next": END}


def route_supervisor(state: AgentState) -> Literal["triage", "action", "comms"] | str:
    return state.get("next", END)


def build_supervisor_graph() -> Any:
    """
    Supervisor topology:
      START -> supervisor -> [triage | action | comms] -> supervisor -> ... -> END

    The supervisor node decides which sub-agent to invoke next.
    Each sub-agent routes back to the supervisor after completion.
    """
    builder = StateGraph(AgentState)

    builder.add_node("supervisor", supervisor_node)
    builder.add_node("triage", triage_node)
    builder.add_node("action", action_node)
    builder.add_node("comms", comms_node)

    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "triage": "triage",
            "action": "action",
            "comms": "comms",
            END: END,
        },
    )
    builder.add_edge("triage", "supervisor")
    builder.add_edge("action", "supervisor")
    builder.add_edge("comms", "supervisor")

    return builder.compile(checkpointer=get_checkpointer())
