"""
loader.py

This file is responsible for loading the trained model pipeline and metadata.

In the real system, this will load from MLflow. For now, it loads from local artifacts.
"""

import joblib
import json


class ModelLoader:
    """
    Loads model pipeline and metadata into memory.
    """

    def __init__(self, model_path: str, metadata_path: str):
        self.model_path = model_path
        self.metadata_path = metadata_path

        self.pipeline = None
        self.threshold = None
        self.model_name = None
        self.model_version = None

    def load(self):
        """
        Load model and metadata from disk.
        """

        # Load trained pipeline
        self.pipeline = joblib.load(self.model_path)

        # Load metadata (threshold, model info, etc.)
        with open(self.metadata_path, "r") as f:
            metadata = json.load(f)

        self.threshold = metadata["operating_threshold"]
        self.model_name = "bank_marketing_classifier"
        self.model_version = "v1"

        return self