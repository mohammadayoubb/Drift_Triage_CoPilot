"""
predictor.py

This file contains the prediction logic for the model service.

It receives validated input data, converts it into the same column format used
during training, gets the model probability, applies the operating threshold,
and returns a clean prediction result.
"""

import pandas as pd


class ModelPredictor:
    """
    Wraps the trained model pipeline and prediction threshold.
    """

    def __init__(self, pipeline, threshold: float, model_name: str, model_version: str):
        # Trained sklearn pipeline loaded from model artifacts or MLflow
        self.pipeline = pipeline

        # Operating threshold chosen during validation
        self.threshold = threshold

        # Registry metadata
        self.model_name = model_name
        self.model_version = model_version

    def predict(self, request_data: dict) -> dict:
        """
        Run prediction for one validated request.
        """

        # Convert API-safe field names back to original training column names
        row = {
            "age": request_data["age"],
            "campaign": request_data["campaign"],
            "pdays": request_data["pdays"],
            "previous": request_data["previous"],
            "emp.var.rate": request_data["emp_var_rate"],
            "cons.price.idx": request_data["cons_price_idx"],
            "cons.conf.idx": request_data["cons_conf_idx"],
            "euribor3m": request_data["euribor3m"],
            "nr.employed": request_data["nr_employed"],
            "pdays_was_999": request_data["pdays_was_999"],
            "job": request_data["job"],
            "marital": request_data["marital"],
            "education": request_data["education"],
            "default": request_data["default"],
            "housing": request_data["housing"],
            "loan": request_data["loan"],
            "contact": request_data["contact"],
            "month": request_data["month"],
            "day_of_week": request_data["day_of_week"],
            "poutcome": request_data["poutcome"],
        }

        # sklearn expects a DataFrame with the same feature names used in training
        input_df = pd.DataFrame([row])

        # Get probability of positive class: y = yes
        probability_yes = float(self.pipeline.predict_proba(input_df)[:, 1][0])

        # Apply threshold selected during validation
        prediction = "yes" if probability_yes >= self.threshold else "no"

        return {
            "prediction": prediction,
            "probability_yes": probability_yes,
            "threshold_used": self.threshold,
            "model_name": self.model_name,
            "model_version": self.model_version,
        }