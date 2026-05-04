"""
prediction_logger.py

This file handles prediction logging.

Every prediction should be stored so the drift monitor can later compare
recent live data against the training/reference distribution.
"""

import json
from datetime import datetime, timezone
from pathlib import Path


class PredictionLogger:
    """
    Simple file-based logger for prediction records.

    This is acceptable for the first local version.
    Later, this should move to Postgres for persistence across services.
    """

    def __init__(self, log_path: str = "data/predictions.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: dict) -> None:
        """
        Append one prediction record to a JSONL file.
        """

        record["logged_at"] = datetime.now(timezone.utc).isoformat()

        with self.log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")