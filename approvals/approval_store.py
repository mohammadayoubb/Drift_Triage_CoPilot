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
        Save one approval request or decision record.
        """

        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(approval) + "\n")

    def read_all(self) -> list[dict]:
        """
        Read all approval records (requests and decisions).
        """

        if not self.path.exists():
            return []

        with self.path.open("r", encoding="utf-8") as file:
            return [json.loads(line) for line in file]

    def _decisions(self) -> dict[str, dict]:
        """
        Return a mapping of approval_id -> decision record for every approval
        that has been approved or rejected.

        Decision records are distinguished from request records by the absence
        of the ``investigation_id`` field.
        """
        decided: dict[str, dict] = {}
        for record in self.read_all():
            if (
                "investigation_id" not in record
                and record.get("status") in ("approved", "rejected")
            ):
                aid = record.get("approval_id")
                if aid:
                    decided[aid] = record
        return decided

    def pending(self) -> list[dict]:
        """
        Return only pending approval requests that have not yet been decided.
        """
        decided_ids = set(self._decisions().keys())
        return [
            record for record in self.read_all()
            if record.get("status") == "pending"
            and "investigation_id" in record
            and record.get("approval_id") not in decided_ids
        ]

    def find_by_approval_id(self, approval_id: str) -> dict | None:
        """
        Return the original approval request for a given approval_id, or None.
        Distinguishes request records from decision records by the presence of
        ``investigation_id`` (decision records don't have it).
        """

        for record in self.read_all():
            if (
                record.get("approval_id") == approval_id
                and "investigation_id" in record
            ):
                return record
        return None

    def find_decision_by_approval_id(self, approval_id: str) -> dict | None:
        """
        Return the decision record for a given approval_id if one exists,
        or None if the approval is still pending.
        """
        for record in self.read_all():
            if (
                record.get("approval_id") == approval_id
                and "investigation_id" not in record
                and record.get("status") in ("approved", "rejected")
            ):
                return record
        return None

    def find_pending_by_investigation_id(
        self,
        investigation_id: str,
        action_type: str | None = None,
    ) -> dict | None:
        """
        Return the first undecided pending approval for a given investigation,
        or None.  Used to prevent duplicate approval requests for the same
        investigation.

        Pass action_type to restrict the match to a specific action — this
        prevents a pending workflow approval from blocking creation of a
        promote_candidate_model approval for the same investigation.
        """
        decided_ids = set(self._decisions().keys())
        for record in self.read_all():
            if (
                record.get("status") == "pending"
                and "investigation_id" in record
                and record.get("investigation_id") == investigation_id
                and (action_type is None or record.get("action_type") == action_type)
                and record.get("approval_id") not in decided_ids
            ):
                return record
        return None

    def history(self) -> list[dict]:
        """
        Return all approval requests with their final resolved status,
        most recent first.

        Each entry contains the original request fields merged with any
        decision fields (status, approved_by, decision_reason, decided_at_utc,
        job_queued).
        """
        decisions = self._decisions()
        requests: list[dict] = []
        for record in self.read_all():
            if "investigation_id" not in record:
                continue
            merged = record.copy()
            decision = decisions.get(record.get("approval_id", ""))
            if decision:
                merged["status"] = decision["status"]
                merged["approved_by"] = decision.get("approved_by")
                merged["decision_reason"] = decision.get("decision_reason")
                merged["decided_at_utc"] = decision.get("decided_at_utc")
                merged["job_queued"] = decision.get("job_queued", False)
            requests.append(merged)
        return list(reversed(requests))
