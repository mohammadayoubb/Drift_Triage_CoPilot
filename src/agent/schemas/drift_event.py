# src/agent/schemas/drift_event.py

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

DriftSeverity = Literal["none", "warning", "critical"]


class DriftEvent(BaseModel):
    """
    Drift event sent by the model service to the agent.

    This mirrors contracts/drift_event.v1.json.
    """

    model_config = ConfigDict(extra="allow")

    event_type: str = Field(default="drift_report.severity_changed")
    version: str = Field(default="v1")
    severity: DriftSeverity
    created_at_utc: str
    recent_count: int = Field(..., ge=0)
    window_size: int = Field(..., ge=1)
    report: dict[str, Any]