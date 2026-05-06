# src/agent/graph/runner.py

import logging
from functools import lru_cache
from typing import Any

from agent.graph.supervisor import build_supervisor_graph
from agent.schemas.drift_event import DriftEvent

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_graph():
    return build_supervisor_graph()


def run_investigation_graph(
    *,
    event: DriftEvent,
    investigation_id: str,
    thread_id: str,
) -> dict[str, Any]:
    """
    Run the supervisor graph for one investigation.

    thread_id is important: LangGraph checkpoints are keyed by thread.
    """
    graph = get_graph()

    initial_state = {
        "investigation_id": investigation_id,
        "thread_id": thread_id,
        "drift_event": event.model_dump(),
        "severity": event.severity,
        "status": "opened",
        "requires_human_approval": False,
    }

    config = {
        "configurable": {
            "thread_id": thread_id,
        }
    }

    logger.info(
        "Running agent graph: investigation_id=%s thread_id=%s severity=%s",
        investigation_id,
        thread_id,
        event.severity,
    )

    final_state = graph.invoke(initial_state, config=config)

    logger.info(
        "Agent graph completed: investigation_id=%s status=%s",
        investigation_id,
        final_state.get("status"),
    )

    return final_state