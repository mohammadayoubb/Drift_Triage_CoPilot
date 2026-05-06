"""
investigation_store.py

This file stores drift investigations created by the agent.

This is a local JSONL version for now. Later, LangGraph checkpoints and Postgres
will provide durable agent state.
"""

import json
from pathlib import Path


class InvestigationStore:
    """
    Stores investigation records locally.
    """

    def __init__(self, path: str = "data/investigations.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, investigation: dict) -> None:
        """
        Save one investigation record.
        """

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(investigation) + "\n")

    def read_all(self) -> list[dict]:
        """
        Read all saved investigations.
        """

        if not self.path.exists():
            return []

        records = []

        with self.path.open("r", encoding="utf-8") as file:
            for line in file:
                records.append(json.loads(line))

        return records