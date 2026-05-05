"""
prediction_store.py

This file stores prediction records in JSONL format.

Each prediction is saved so the drift monitor can later compare recent live data
against the reference training distribution.
"""

import json
from pathlib import Path


class PredictionStore:
    """
    Stores prediction records in a local JSONL file.

    This is a simple first version. Later, this can be replaced with Postgres.
    """

    def __init__(self, path: str = "data/predictions.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict) -> None:
        """
        Save one prediction record.
        """

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")

    def read_all(self) -> list[dict]:
        """
        Read all saved prediction records.
        """

        if not self.path.exists():
            return []

        records = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line))

        return records