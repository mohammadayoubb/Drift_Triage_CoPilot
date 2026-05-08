"""
status.py

This file provides queue status helpers.

The dashboard and API can use this later to show:
- queue depth
- running jobs
- DLQ depth
"""

import json

from async_queue.redis_client import get_redis_client
from async_queue.producer import QUEUE_NAME
from async_queue.dlq import DLQ_NAME
from async_queue.worker import COMPLETED_QUEUE_NAME, RUNNING_HASH_NAME

_RECENT_N = 10


def _try_json(raw) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        return {"raw": str(raw)}


class QueueStatus:
    """
    Reads queue, running-jobs, completed-jobs, and DLQ status from Redis.
    """

    def __init__(self):
        self.redis_client = get_redis_client()

    def get_status(self) -> dict:
        """
        Return queue depth, running jobs, completed-jobs depth, DLQ depth, and
        recent items for completed jobs and the DLQ (most-recent-first).
        """
        recent_completed = [
            _try_json(item)
            for item in reversed(
                self.redis_client.lrange(COMPLETED_QUEUE_NAME, -_RECENT_N, -1)
            )
        ]
        recent_dlq = [
            _try_json(item)
            for item in reversed(
                self.redis_client.lrange(DLQ_NAME, -5, -1)
            )
        ]

        running_raw = self.redis_client.hgetall(RUNNING_HASH_NAME)
        running_jobs = [_try_json(v) for v in (running_raw or {}).values()]

        return {
            "queue": QUEUE_NAME,
            "queue_depth": self.redis_client.llen(QUEUE_NAME),
            "running_depth": len(running_jobs),
            "running_jobs": running_jobs,
            "dlq": DLQ_NAME,
            "dlq_depth": self.redis_client.llen(DLQ_NAME),
            "completed_depth": self.redis_client.llen(COMPLETED_QUEUE_NAME),
            "recent_completed_jobs": recent_completed,
            "recent_dlq_jobs": recent_dlq,
        }
