"""
predictor.py

This file handles model inference.

It supports:
- sklearn pipelines (with predict_proba)
- MLflow pyfunc models (predict only)
"""

import pandas as pd


class ModelPredictor:
    """
    Handles predictions and threshold logic.
    """

    def __init__(self, pipeline, threshold, model_name, model_version):
        self.pipeline = pipeline
        self.threshold = threshold
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, input_data: dict) -> dict:
        """
        Run prediction and apply threshold.
        """

        input_df = pd.DataFrame([input_data])

        # --- CASE 1: sklearn pipeline (local model)
        if hasattr(self.pipeline, "predict_proba"):
            probability_yes = float(
                self.pipeline.predict_proba(input_df)[:, 1][0]
            )

        # --- CASE 2: MLflow pyfunc model
        else:
            prediction = self.pipeline.predict(input_df)

            # MLflow may return:
            # - probability
            # - class label
            # We assume probability here (as logged)
            probability_yes = float(prediction[0])

        # Apply threshold
        predicted_label = "yes" if probability_yes >= self.threshold else "no"

        return {
            "prediction": predicted_label,
            "probability_yes": probability_yes,
            "threshold_used": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }