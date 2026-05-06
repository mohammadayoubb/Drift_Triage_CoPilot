# scripts/simulate_drift.py

import argparse
import logging
import random
import time
from typing import Any

import httpx
import pandas as pd

from common.logging import configure_logging
from common.paths import TEST_DATA_PATH
from ml.constants import MODEL_FEATURES

logger = logging.getLogger(__name__)

API_BASE_URL = "http://127.0.0.1:8000"

# The API should not receive engineered features or target.
REQUEST_FEATURES = [
    feature
    for feature in MODEL_FEATURES
    if feature != "pdays_was_999"
]


def to_json_safe(value: Any) -> Any:
    """
    Convert pandas/numpy values into JSON-safe Python values.
    """
    if pd.isna(value):
        return None

    if hasattr(value, "item"):
        return value.item()

    return value


def row_to_payload(row: pd.Series) -> dict[str, Any]:
    """
    Convert one processed dataset row into API request payload.
    """
    payload = {
        feature: to_json_safe(row[feature])
        for feature in REQUEST_FEATURES
    }

    return payload


def make_drifted_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """
    Intentionally shift one numeric group and one categorical group.

    This simulates a changed live population, not one accidental weird row.
    """
    drifted = payload.copy()

    # Numeric macroeconomic shift
    drifted["euribor3m"] = 0.6
    drifted["cons.price.idx"] = 96.5
    drifted["cons.conf.idx"] = -20.0

    # Categorical distribution shift
    drifted["job"] = "student"
    drifted["contact"] = "cellular"
    drifted["month"] = "mar"

    return drifted


def send_prediction(client: httpx.Client, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(
        f"{API_BASE_URL}/predict",
        json=payload,
        timeout=10.0,
    )
    response.raise_for_status()
    return response.json()


def run_simulation(mode: str, n: int, sleep_seconds: float) -> None:
    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Test split not found at {TEST_DATA_PATH}. "
            "Run: uv run python scripts/make_splits.py"
        )

    test_df = pd.read_csv(TEST_DATA_PATH)

    if test_df.empty:
        raise ValueError("Test split is empty.")

    rows = test_df.sample(
        n=n,
        replace=True,
        random_state=42 if mode == "normal" else 99,
    )

    logger.info("Starting simulation: mode=%s n=%s", mode, n)

    successes = 0

    with httpx.Client() as client:
        for index, (_, row) in enumerate(rows.iterrows(), start=1):
            payload = row_to_payload(row)

            if mode == "drift":
                payload = make_drifted_payload(payload)

            result = send_prediction(client, payload)

            successes += 1

            logger.info(
                "Sent prediction %s/%s: request_id=%s probability=%.4f prediction=%s",
                index,
                n,
                result["request_id"],
                result["probability"],
                result["prediction"],
            )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

    logger.info("Simulation complete: mode=%s successes=%s", mode, successes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Send normal or drifted prediction traffic to the model service."
    )

    parser.add_argument(
        "--mode",
        choices=["normal", "drift"],
        required=True,
        help="normal sends realistic test rows; drift sends intentionally shifted rows.",
    )

    parser.add_argument(
        "--n",
        type=int,
        default=40,
        help="Number of prediction requests to send.",
    )

    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Seconds to sleep between requests.",
    )

    return parser.parse_args()


def main() -> None:
    configure_logging()

    args = parse_args()

    run_simulation(
        mode=args.mode,
        n=args.n,
        sleep_seconds=args.sleep,
    )


if __name__ == "__main__":
    main()