# src/agent/schemas/drift_event.py

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DriftEvent(BaseModel):
    """
    Drift event sent by the model service to the agent webhook.

    Field names mirror service/drift/event_builder.py.
    Severity values mirror service/drift/drift_monitor.py: LOW / MEDIUM / HIGH / CRITICAL.
    """

    model_config = ConfigDict(extra="allow")

    contract_version: str = "1.0"
    event_id: str = ""
    event_time: str = ""
    model_name: str = ""
    model_version: str = ""
    production_model_version: str = ""
    severity: str  # LOW | MEDIUM | HIGH | CRITICAL (produced by drift_monitor)
    previous_severity: str | None = None
    affected_features: list[str] = Field(default_factory=list)
    drift_type: str = "none"
    numeric_drift_summary: dict[str, Any] = Field(default_factory=dict)
    categorical_drift_summary: dict[str, Any] = Field(default_factory=dict)
    output_drift_summary: dict[str, Any] = Field(default_factory=dict)
    rolling_window_size: int | None = None
    reference_stats_version: str = ""
