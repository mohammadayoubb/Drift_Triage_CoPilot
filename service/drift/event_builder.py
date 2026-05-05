"""
event_builder.py

This file builds drift event payloads.

A drift event is the contract sent from the model platform to the agent when
drift severity changes.
"""

from service.utils.id_generator import generate_event_id
from service.utils.time_utils import get_utc_timestamp


class DriftEventBuilder:
    """
    Builds versioned drift event payloads for the agent.
    """

    def build_event(
        self,
        report: dict,
        model_name: str,
        model_version: str,
        previous_severity: str | None = None,
    ) -> dict:
        """
        Create a drift event from a drift report.
        """

        return {
            "contract_version": "1.0",
            "event_id": generate_event_id(),
            "event_time": get_utc_timestamp(),
            "model_name": model_name,
            "model_version": model_version,
            "production_model_version": model_version,
            "severity": report.get("severity"),
            "previous_severity": previous_severity,
            "affected_features": report.get("affected_features", []),
            "drift_type": self._infer_drift_type(report),
            "numeric_drift_summary": report.get("numeric_results", {}),
            "categorical_drift_summary": report.get("categorical_results", {}),
            "output_drift_summary": report.get("output_results", {}),
            "rolling_window_size": report.get("window_size"),
            "reference_stats_version": "reference_v1",
        }

    def _infer_drift_type(self, report: dict) -> str:
        """
        Infer drift type from affected features.
        """

        has_numeric = bool(report.get("numeric_results"))
        has_categorical = bool(report.get("categorical_results"))

        if has_numeric and has_categorical:
            return "mixed"

        if has_numeric:
            return "numeric"

        if has_categorical:
            return "categorical"

        return "none"