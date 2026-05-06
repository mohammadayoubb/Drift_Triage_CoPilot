"""
idempotency.py

This file handles idempotency for queue jobs.

Idempotency prevents the same logical job from being executed multiple times,
even if retries or duplicate requests happen.
"""

from async_queue.redis_client import get_redis_client


class IdempotencyStore:
    """
    Stores idempotency keys in Redis.
    """

    def __init__(self):
        self.redis_client = get_redis_client()

    def already_seen(self, key: str) -> bool:
        """
        Return True if this idempotency key already exists.
        """

        return self.redis_client.exists(f"idempotency:{key}") == 1

    def mark_seen(self, key: str) -> None:
        """
        Mark an idempotency key as seen.
        """

        self.redis_client.set(f"idempotency:{key}", "seen")