"""
approval_store.py

This file stores human approval requests locally.

The dashboard will later read pending approvals from here.
"""

import json
from pathlib import Path


class ApprovalStore:
    """
    Local JSONL approval store.
    """

    def __init__(self, path: str = "data/approvals.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, approval: dict) -> None:
        """
        Save one approval request.
        """

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(approval) + "\n")

    def read_all(self) -> list[dict]:
        """
        Read all approval records.
        """

        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def pending(self) -> list[dict]:
        """
        Return only pending approvals.
        """

        return [
            approval for approval in self.read_all()
            if approval.get("status") == "pending"
        ]