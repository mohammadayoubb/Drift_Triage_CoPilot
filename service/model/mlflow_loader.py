"""
mlflow_loader.py

This file loads the Production model from MLflow registry.

This replaces direct file loading and makes the system registry-driven.
"""

import mlflow.pyfunc


class MLflowModelLoader:
    """
    Loads model from MLflow Production stage.
    """

    def __init__(self, model_name: str):
        self.model_name = model_name

        self.model = None
        self.model_version = None

    def load(self):
        """
        Load Production model from MLflow.
        """

        model_uri = f"models:/{self.model_name}/Production"

        self.model = mlflow.pyfunc.load_model(model_uri)

        # NOTE: MLflow pyfunc doesn't directly expose version easily
        # For now, we assume version is "Production"
        self.model_version = "Production"

        return self