"""
settings.py

Central configuration file for the model service.

All configurable values should live here instead of being hardcoded.
"""

import os


class Settings:
    """
    Application settings loaded from environment variables or defaults.
    """

    # Model artifact paths
    MODEL_PATH = os.getenv(
        "MODEL_PATH",
        "model_artifacts/bank_marketing_pipeline.joblib"
    )

    METADATA_PATH = os.getenv(
        "METADATA_PATH",
        "model_artifacts/model_metadata.json"
    )

    # Drift webhook (agent endpoint)
    AGENT_WEBHOOK_URL = os.getenv(
    "AGENT_WEBHOOK_URL",
    "http://127.0.0.1:8001/drift"
)

    # Drift window size
    DRIFT_WINDOW_SIZE = int(
        os.getenv("DRIFT_WINDOW_SIZE", "100")
    )

    # Service info
    SERVICE_NAME = "model_service"
    MODEL_NAME = "bank_marketing_classifier"