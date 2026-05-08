"""
dlq.py

This file manages the dead-letter queue.

Jobs that fail too many times are moved to the DLQ so they are not lost and can
be inspected from the dashboard later.
"""

import json

from async_queue.redis_client import get_redis_client


DLQ_NAME = "drift_jobs_dlq"


class DeadLetterQueue:
    """
    Stores permanently failed jobs.
    """

    def __init__(self):
        self.redis_client = get_redis_client()

    def push(self, failed_job: dict) -> None:
        """
        Add a failed job to the DLQ as a JSON string.
        """

        self.redis_client.rpush(DLQ_NAME, json.dumps(failed_job))

    def list_failed(self) -> list[dict]:
        """
        Return all failed jobs in the DLQ as decoded dicts.
        """

        items = []
        for raw in self.redis_client.lrange(DLQ_NAME, 0, -1):
            try:
                items.append(json.loads(raw))
            except Exception:
                items.append({"raw": str(raw)})
        return items