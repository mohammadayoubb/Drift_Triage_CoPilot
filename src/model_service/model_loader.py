# src/model_service/model_loader.py

import json
import logging
from functools import lru_cache
from typing import Any

import joblib

from common.paths import MODEL_PATH, THRESHOLD_PATH
from ml.constants import MLFLOW_MODEL_NAME

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def load_model() -> Any:
    """
    Load the trained sklearn pipeline from disk.

    Cached so the API does not reload the model on every request.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. "
            "Run: uv run python scripts/train_register_model.py"
        )

    logger.info("Loading model artifact from %s", MODEL_PATH)

    model = joblib.load(MODEL_PATH)

    logger.info("Model artifact loaded successfully")

    return model


@lru_cache(maxsize=1)
def load_threshold() -> float:
    """
    Load the operating threshold chosen during training.
    """
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold artifact not found at {THRESHOLD_PATH}. "
            "Run: uv run python scripts/train_register_model.py"
        )

    logger.info("Loading threshold artifact from %s", THRESHOLD_PATH)

    threshold_payload = json.loads(
        THRESHOLD_PATH.read_text(encoding="utf-8")
    )

    threshold = float(threshold_payload["threshold"])

    logger.info("Threshold loaded successfully: %.4f", threshold)

    return threshold


def get_model_status() -> dict[str, bool]:
    """
    Used by /health to report whether required artifacts exist.
    """
    return {
        "model_loaded": MODEL_PATH.exists(),
        "threshold_loaded": THRESHOLD_PATH.exists(),
    }


def get_model_name() -> str:
    return MLFLOW_MODEL_NAME