# src/agent/graph/nodes/comms.py

import logging
import os
from pathlib import Path
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "comms.md").read_text()


def _llm_summary(state: AgentState) -> str:
    """
    Call gpt-4o-mini to produce a stakeholder-facing investigation summary.
    Returns empty string if OPENAI_API_KEY is not set or the call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        triage = state.get("triage_result", {})
        action = state.get("recommended_action", {})
        user_content = (
            f"Severity: {state.get('severity', 'unknown')}\n"
            f"Affected features: {triage.get('affected_features', [])}\n"
            f"Triage analysis: {triage.get('llm_analysis', 'N/A')}\n"
            f"Recommended action: {action.get('action_type', 'N/A')}\n"
            f"Reason: {action.get('reason', 'N/A')}\n"
            f"Requires human approval: {state.get('requires_human_approval', False)}"
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=api_key)
        response = llm.invoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=user_content),
        ])
        return response.content
    except Exception as exc:
        logger.warning("Comms LLM call failed (non-fatal): %s", exc)
        return ""


def comms_node(state: AgentState) -> dict[str, Any]:
    """
    Write the investigation summary for the dashboard.
    Uses gpt-4o-mini when OPENAI_API_KEY is set; falls back to a
    deterministic template summary otherwise.
    Sets status=resolved so the supervisor routes to END.
    """
    investigation_id = state["investigation_id"]
    severity = state.get("severity", "none")
    triage_result = state.get("triage_result", {})
    recommended_action = state.get("recommended_action", {})

    # Try LLM first
    comms_summary = _llm_summary(state)

    if not comms_summary:
        # Deterministic fallback — always works, no API key needed
        numeric_summary = ", ".join(
            item["feature"] for item in triage_result.get("top_numeric_drift", [])[:3]
        ) or "none"
        categorical_summary = ", ".join(
            item["feature"] for item in triage_result.get("top_categorical_drift", [])[:3]
        ) or "none"
        comms_summary = (
            f"Investigation {investigation_id}: severity={severity}. "
            f"Top numeric drift: {numeric_summary}. "
            f"Top categorical drift: {categorical_summary}. "
            f"Recommended action: {recommended_action.get('action_type')}."
        )

    return {
        "comms_summary": comms_summary,
        "status": "resolved",
    }
