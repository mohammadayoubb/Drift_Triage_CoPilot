"""
drift_event.py

This file defines the schema for drift events received from the model platform.

The model service sends this event when drift severity changes.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any


class DriftEvent(BaseModel):
    """
    Contract for platform-to-agent drift events.
    """

    model_config = ConfigDict(extra="forbid")

    contract_version: str
    event_id: str
    event_time: str

    model_name: str
    model_version: str
    production_model_version: str

    severity: str
    previous_severity: str | None = None

    affected_features: list[str]
    drift_type: str

    numeric_drift_summary: dict[str, Any]
    categorical_drift_summary: dict[str, Any]
    output_drift_summary: dict[str, Any]

    rolling_window_size: int | None = None
    reference_stats_version: str