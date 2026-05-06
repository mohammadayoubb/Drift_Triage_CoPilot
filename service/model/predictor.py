"""
predictor.py

This file handles model inference.

It converts validated API input into the exact feature column names used during
training, then runs prediction and applies the operating threshold.
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

        # Convert API-safe field names back to original training column names.
        row = {
            "age": input_data["age"],
            "campaign": input_data["campaign"],
            "pdays": input_data["pdays"],
            "previous": input_data["previous"],
            "emp.var.rate": input_data["emp_var_rate"],
            "cons.price.idx": input_data["cons_price_idx"],
            "cons.conf.idx": input_data["cons_conf_idx"],
            "euribor3m": input_data["euribor3m"],
            "nr.employed": input_data["nr_employed"],
            "pdays_was_999": input_data["pdays_was_999"],
            "job": input_data["job"],
            "marital": input_data["marital"],
            "education": input_data["education"],
            "default": input_data["default"],
            "housing": input_data["housing"],
            "loan": input_data["loan"],
            "contact": input_data["contact"],
            "month": input_data["month"],
            "day_of_week": input_data["day_of_week"],
            "poutcome": input_data["poutcome"],
        }

        input_df = pd.DataFrame([row])

        if hasattr(self.pipeline, "predict_proba"):
            probability_yes = float(
                self.pipeline.predict_proba(input_df)[:, 1][0]
            )
        else:
            prediction = self.pipeline.predict(input_df)
            probability_yes = float(prediction[0])

        predicted_label = "yes" if probability_yes >= self.threshold else "no"

        return {
            "prediction": predicted_label,
            "probability_yes": probability_yes,
            "threshold_used": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }