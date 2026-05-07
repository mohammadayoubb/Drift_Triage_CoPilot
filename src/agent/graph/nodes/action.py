# src/agent/graph/nodes/action.py

import logging
import uuid
from pathlib import Path
from typing import Any

from langgraph.types import interrupt

from ..state import AgentState

_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "action.md").read_text()

logger = logging.getLogger(__name__)


def _dispatch_queue_task(job_type: str, investigation_id: str) -> None:
    """Enqueue a job to the Redis async queue with an idempotency key."""
    try:
        from async_queue.job_models import QueueJob
        from async_queue.producer import QueueProducer

        job = QueueJob(
            job_id=str(uuid.uuid4()),
            job_type=job_type,
            payload={"investigation_id": investigation_id},
            idempotency_key=f"{investigation_id}-{job_type}",
        )
        QueueProducer().enqueue(job)
    except Exception as exc:
        logger.warning("Queue dispatch failed (non-fatal): %s", exc)


def action_node(state: AgentState) -> dict[str, Any]:
    """
    Recommend and dispatch the next operational action.

    CRITICAL drift: dispatches replay_test to the queue AND requires human
    approval (touches_production=True) before proceeding with retrain/promote.
    The graph pauses here via interrupt() until a human approves in the dashboard.

    HIGH drift: dispatches replay_test but does NOT touch Production.

    LOW/MEDIUM: no action.
    """
    severity = state.get("severity", "none")
    investigation_id = state.get("investigation_id", "unknown")

    if severity in ("critical", "CRITICAL"):
        recommended_action = {
            "action_type": "replay_test_set_then_open_retrain_candidate",
            "priority": "high",
            "reason": (
                "Critical drift detected across numeric and categorical features. "
                "Replay test set dispatched immediately. "
                "Retrain and production promotion require human approval."
            ),
            "queue_task": "replay_test",
            "touches_production": True,
        }
        _dispatch_queue_task("replay_test", investigation_id)

        # Pause the graph — human must approve in the dashboard before
        # any production-touching step (retrain + promote) proceeds.
        approval = interrupt({
            "reason": "Critical drift: replay test dispatched. Approve to proceed with retrain.",
            "investigation_id": investigation_id,
            "recommended_action": recommended_action,
        })

        # If human rejected, downgrade to monitor-only
        if isinstance(approval, dict) and not approval.get("approved", True):
            recommended_action = {
                **recommended_action,
                "action_type": "rejected_by_human",
                "reason": "Human reviewer rejected the retrain/promote action.",
                "touches_production": False,
            }

    elif severity in ("high", "HIGH"):
        recommended_action = {
            "action_type": "monitor_and_replay_test_set",
            "priority": "medium",
            "reason": "High drift detected. Replay test set before deciding on retrain.",
            "queue_task": "replay_test",
            "touches_production": False,
        }
        _dispatch_queue_task("replay_test", investigation_id)

    else:
        recommended_action = {
            "action_type": "no_action",
            "priority": "low",
            "reason": "No meaningful drift detected. Continue monitoring.",
            "queue_task": None,
            "touches_production": False,
        }

    return {
        "recommended_action": recommended_action,
        "requires_human_approval": recommended_action["touches_production"],
        "status": "action_recommended",
    }
