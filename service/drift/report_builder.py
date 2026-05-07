"""
report_builder.py

This file builds a drift report using reference data and recent live prediction
records saved by the prediction store.
"""

import pandas as pd

from service.drift.drift_monitor import DriftMonitor
from service.storage.prediction_store import PredictionStore
from service.storage.reference_store import ReferenceStore


NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "pdays_was_999",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]


class DriftReportBuilder:
    """
    Builds drift reports from stored prediction records.
    """

    def __init__(
        self,
        reference_store: ReferenceStore | None = None,
        prediction_store: PredictionStore | None = None,
        window_size: int = 100,
    ):
        self.reference_store = reference_store or ReferenceStore()
        self.prediction_store = prediction_store or PredictionStore()
        self.window_size = window_size

        self.monitor = DriftMonitor(
            numeric_features=NUMERIC_FEATURES,
            categorical_features=CATEGORICAL_FEATURES,
        )

    def build(self) -> dict:
        """
        Build a drift report from reference data and recent prediction records.
        """

        reference_df = self.reference_store.load()
        records = self.prediction_store.read_all()

        if len(records) < self.window_size:
            return {
                "status": "insufficient_data",
                "message": "Not enough prediction records to compute drift.",
                "records_available": len(records),
                "records_required": self.window_size,
            }

        recent_records = records[-self.window_size:]

        current_features = [
            record["features"]
            for record in recent_records
        ]

        current_df = pd.DataFrame(current_features)

        # pdays_was_999 is a derived feature not stored in prediction logs;
        # recompute it from pdays so the drift monitor can compare against reference.
        if "pdays_was_999" not in current_df.columns and "pdays" in current_df.columns:
            current_df["pdays_was_999"] = (current_df["pdays"] == 999).astype(int)

        report = self.monitor.run_report(
            reference_df=reference_df,
            current_df=current_df,
        )

        report["status"] = "ok"
        report["window_size"] = self.window_size
        report["records_available"] = len(records)

        return report