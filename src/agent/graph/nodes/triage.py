# src/agent/graph/nodes/triage.py

import logging
import os
from pathlib import Path
from typing import Any

from ..state import AgentState

logger = logging.getLogger(__name__)

_PROMPT = (Path(__file__).parent.parent.parent / "prompts" / "triage.md").read_text()


def _top_numeric_drift(drift_event: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for feature, payload in drift_event.get("numeric_drift_summary", {}).items():
        psi = payload.get("psi")
        if psi is None:
            continue
        rows.append({
            "feature": feature,
            "metric": "psi",
            "value": psi,
            "severity": payload.get("severity", "none"),
        })
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _top_categorical_drift(drift_event: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    for feature, payload in drift_event.get("categorical_drift_summary", {}).items():
        chi_square = payload.get("chi_square")
        if chi_square is None:
            continue
        rows.append({
            "feature": feature,
            "metric": "chi_square",
            "value": chi_square,
            "p_value": payload.get("p_value"),
            "severity": payload.get("severity", "none"),
        })
    return sorted(rows, key=lambda item: item["value"], reverse=True)[:limit]


def _llm_analysis(triage_data: dict[str, Any]) -> str:
    """
    Call gpt-4o-mini with the triage prompt and drift data.
    Returns empty string if OPENAI_API_KEY is not set or the call fails.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return ""
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage

        user_content = (
            f"Severity: {triage_data['severity']}\n"
            f"Affected features: {triage_data['affected_features']}\n"
            f"Top numeric drift (PSI): {triage_data['top_numeric_drift']}\n"
            f"Top categorical drift (chi-square): {triage_data['top_categorical_drift']}\n"
            f"Output drift: {triage_data['output_drift']}"
        )
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
        response = llm.invoke([
            SystemMessage(content=_PROMPT),
            HumanMessage(content=user_content),
        ])
        return response.content
    except Exception as exc:
        logger.warning("Triage LLM call failed (non-fatal): %s", exc)
        return ""


def triage_node(state: AgentState) -> dict[str, Any]:
    """
    Triage drift severity and identify the most important signals.
    Calls gpt-4o-mini for an enhanced analysis if OPENAI_API_KEY is set.
    Falls back to deterministic output if the key is absent or the call fails.
    """
    drift_event = state["drift_event"]
    severity = drift_event.get("severity", state.get("severity", "none"))

    triage_data: dict[str, Any] = {
        "severity": severity,
        "rolling_window_size": drift_event.get("rolling_window_size"),
        "affected_features": drift_event.get("affected_features", []),
        "top_numeric_drift": _top_numeric_drift(drift_event),
        "top_categorical_drift": _top_categorical_drift(drift_event),
        "output_drift": drift_event.get("output_drift_summary", {}),
    }

    analysis = _llm_analysis(triage_data)
    if analysis:
        triage_data["llm_analysis"] = analysis

    return {
        "triage_result": triage_data,
        "status": "triaged",
    }
