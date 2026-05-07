"""
loader.py

Unified model loader.

For now, this loads the local sklearn pipeline to preserve predict_proba support.
MLflow registration still exists, but serving uses the local artifact until we
add a custom MLflow probability wrapper.
"""

import joblib
import json

from service.config.settings import Settings


class ModelLoader:
    """
    Loads the local sklearn model artifact.
    """

    def __init__(self):
        self.pipeline = None
        self.threshold = None
        self.model_name = Settings.MODEL_NAME
        self.model_version = "v1"

    def load(self):
        """
        Load model and metadata from local artifacts.
        """

        self.pipeline = joblib.load(Settings.MODEL_PATH)

        with open(Settings.METADATA_PATH, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        self.threshold = metadata["threshold"]

        print("Loaded local sklearn model")

        return self