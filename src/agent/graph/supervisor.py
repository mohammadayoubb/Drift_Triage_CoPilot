# src/agent/graph/supervisor.py

from typing import Any

from langgraph.graph import END, START, StateGraph

from agent.graph.nodes.action import action_node
from agent.graph.nodes.comms import comms_node
from agent.graph.nodes.triage import triage_node
from agent.graph.state import AgentState
from agent.persistence.checkpoints import get_checkpointer


def build_supervisor_graph() -> Any:
    """
    Build and compile the LangGraph supervisor.

    Topology:
    START -> triage -> action -> comms -> END

    This is a supervisor-style deterministic skeleton:
    - triage node analyzes drift
    - action node recommends next operational action
    - comms node writes a human-readable summary

    Later, each node can call an LLM or specialized subgraph.
    """
    builder = StateGraph(AgentState)

    builder.add_node("triage", triage_node)
    builder.add_node("action", action_node)
    builder.add_node("comms", comms_node)

    builder.add_edge(START, "triage")
    builder.add_edge("triage", "action")
    builder.add_edge("action", "comms")
    builder.add_edge("comms", END)

    return builder.compile(checkpointer=get_checkpointer())