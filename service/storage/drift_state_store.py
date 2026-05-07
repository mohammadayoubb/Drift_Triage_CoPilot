"""
drift_state_store.py

This file stores the last known drift severity.

The platform should only send a webhook to the agent when drift severity changes.
"""

import json
from pathlib import Path


class DriftStateStore:
    """
    Stores last drift state locally.

    This is a simple first version. Later, this can move to Postgres.
    """

    def __init__(self, path: str = "data/drift_state.json"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def get_last_severity(self) -> str | None:
        """
        Return the last stored severity, if available.
        """

        if not self.path.exists():
            return "LOW"

        with self.path.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)

        return data.get("last_severity")

    def set_last_severity(self, severity: str) -> None:
        """
        Store latest severity.
        """

        with self.path.open("w", encoding="utf-8") as file:
            json.dump({"last_severity": severity}, file, indent=2)