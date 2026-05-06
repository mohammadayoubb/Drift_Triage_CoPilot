# src/model_service/prediction_log.py

import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from common.paths import PREDICTIONS_LOG_PATH
from ml.constants import MODEL_FEATURES

logger = logging.getLogger(__name__)


def append_prediction_log(
    *,
    request_id: str,
    model_name: str,
    X: pd.DataFrame,
    probability: float,
    prediction: int,
    threshold: float,
    path: Path = PREDICTIONS_LOG_PATH,
) -> None:
    """
    Append one prediction record to the local prediction log.

    This is our temporary persistence layer. Later, this should move to Postgres.
    """
    if X.shape[0] != 1:
        raise ValueError(f"Expected one-row prediction DataFrame, got shape={X.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)

    feature_payload = X.iloc[0].to_dict()

    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
        "model_name": model_name,
        **feature_payload,
        "probability": float(probability),
        "prediction": int(prediction),
        "threshold": float(threshold),
    }

    record_df = pd.DataFrame([record])

    file_exists = path.exists()

    record_df.to_csv(
        path,
        mode="a",
        header=not file_exists,
        index=False,
    )

    logger.info("Prediction logged: request_id=%s path=%s", request_id, path)


def load_prediction_log(path: Path = PREDICTIONS_LOG_PATH) -> pd.DataFrame:
    """
    Load all prediction records.
    """
    if not path.exists():
        logger.warning("Prediction log does not exist yet: %s", path)
        return pd.DataFrame()

    return pd.read_csv(path)


def load_recent_predictions(
    window_size: int = 200,
    path: Path = PREDICTIONS_LOG_PATH,
) -> pd.DataFrame:
    """
    Load the most recent prediction records for drift monitoring.
    """
    df = load_prediction_log(path)

    if df.empty:
        return df

    missing_features = sorted(set(MODEL_FEATURES) - set(df.columns))
    if missing_features:
        raise ValueError(f"Prediction log is missing model features: {missing_features}")

    return df.tail(window_size).copy()