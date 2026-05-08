"""
drift_monitor.py

This file coordinates drift detection.

It compares reference data against recent/live prediction data using:
- PSI for numeric features
- Chi-square tests for categorical features
- Output distribution comparison
"""

from service.drift.psi import calculate_psi
from service.drift.chi_square import calculate_chi_square


class DriftMonitor:
    """
    Runs drift checks for numeric, categorical, and output distributions.
    """

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        psi_medium_threshold: float = 0.1,
        psi_high_threshold: float = 0.25,
    ):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.psi_medium_threshold = psi_medium_threshold
        self.psi_high_threshold = psi_high_threshold

    def evaluate_numeric_drift(self, reference_df, current_df) -> dict:
        """
        Evaluate PSI drift for each numeric feature.
        """

        results = {}

        for feature in self.numeric_features:
            psi_value = calculate_psi(
                reference_df[feature],
                current_df[feature]
            )

            if psi_value >= self.psi_high_threshold:
                severity = "HIGH"
            elif psi_value >= self.psi_medium_threshold:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            drifted = severity in ("MEDIUM", "HIGH", "SEVERE")
            results[feature] = {
                "psi": psi_value,
                "severity": severity,
                "drifted": bool(drifted),
                "is_drifted": bool(drifted),
                "drift_detected": bool(drifted),
            }

        return results

    def evaluate_categorical_drift(self, reference_df, current_df) -> dict:
        """
        Evaluate chi-square drift for each categorical feature.
        """

        results = {}

        for feature in self.categorical_features:
            results[feature] = calculate_chi_square(
                reference_df[feature],
                current_df[feature]
            )

        return results

    def summarize_severity(self, numeric_results: dict, categorical_results: dict) -> str:
        """
        Produce an overall drift severity label.
        """

        has_high_numeric = any(
            result["severity"] == "HIGH"
            for result in numeric_results.values()
        )

        has_medium_numeric = any(
            result["severity"] == "MEDIUM"
            for result in numeric_results.values()
        )

        has_categorical_drift = any(
            result["drift_detected"]
            for result in categorical_results.values()
        )

        if has_high_numeric and has_categorical_drift:
            return "CRITICAL"

        if has_high_numeric:
            return "HIGH"

        if has_medium_numeric or has_categorical_drift:
            return "MEDIUM"

        return "LOW"

    def run_report(self, reference_df, current_df) -> dict:
        """
        Run full drift report.
        """

        numeric_results = self.evaluate_numeric_drift(reference_df, current_df)
        categorical_results = self.evaluate_categorical_drift(reference_df, current_df)

        severity = self.summarize_severity(
            numeric_results=numeric_results,
            categorical_results=categorical_results
        )

        affected_features = []

        for feature, result in numeric_results.items():
            if result["severity"] in ["MEDIUM", "HIGH"]:
                affected_features.append(feature)

        for feature, result in categorical_results.items():
            if result["drift_detected"]:
                affected_features.append(feature)

        return {
            "severity": severity,
            "affected_features": affected_features,
            "numeric_results": numeric_results,
            "categorical_results": categorical_results
        }