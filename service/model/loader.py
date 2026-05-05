"""
loader.py

Unified model loader.

Currently supports:
- MLflow (primary)
- Local fallback (for development)
"""

import joblib
import json

from service.config.settings import Settings
from service.model.mlflow_loader import MLflowModelLoader


class ModelLoader:
    """
    Loads model either from MLflow or local artifacts.
    """

    def __init__(self):
        self.pipeline = None
        self.threshold = None
        self.model_name = Settings.MODEL_NAME
        self.model_version = None

    def load(self):
        """
        Try MLflow first, fallback to local files.
        """

        try:
            # Try MLflow Production model
            mlflow_loader = MLflowModelLoader(self.model_name).load()

            self.pipeline = mlflow_loader.model
            self.model_version = mlflow_loader.model_version

            # Load threshold from local metadata (still needed)
            with open(Settings.METADATA_PATH, "r") as f:
                metadata = json.load(f)

            self.threshold = metadata["operating_threshold"]

            print("Loaded model from MLflow")

        except Exception as e:
            print("MLflow load failed, falling back to local model:", e)

            self.pipeline = joblib.load(Settings.MODEL_PATH)

            with open(Settings.METADATA_PATH, "r") as f:
                metadata = json.load(f)

            self.threshold = metadata["operating_threshold"]
            self.model_version = "local"

        return self