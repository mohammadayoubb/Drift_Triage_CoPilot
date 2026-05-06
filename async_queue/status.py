"""
status.py

This file provides queue status helpers.

The dashboard and API can use this later to show:
- queue depth
- DLQ depth
"""

from async_queue.redis_client import get_redis_client
from async_queue.producer import QUEUE_NAME
from async_queue.dlq import DLQ_NAME


class QueueStatus:
    """
    Reads queue and DLQ status from Redis.
    """

    def __init__(self):
        self.redis_client = get_redis_client()

    def get_status(self) -> dict:
        """
        Return queue depth and DLQ depth.
        """

        return {
            "queue": QUEUE_NAME,
            "queue_depth": self.redis_client.llen(QUEUE_NAME),
            "dlq": DLQ_NAME,
            "dlq_depth": self.redis_client.llen(DLQ_NAME),
        }