"""
dlq.py

This file manages the dead-letter queue.

Jobs that fail too many times are moved to the DLQ so they are not lost and can
be inspected from the dashboard later.
"""

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
        Add a failed job to the DLQ.
        """

        self.redis_client.rpush(
            DLQ_NAME,
            str(failed_job)
        )

    def list_failed(self) -> list[str]:
        """
        Return all failed jobs currently in the DLQ.
        """

        return self.redis_client.lrange(DLQ_NAME, 0, -1)