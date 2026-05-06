"""
redis_client.py

This file creates a Redis client used by the queue producer and worker.

Redis stores async jobs that should not block the agent, such as replay tests,
retraining, and rollback actions.
"""

import os
import redis


def get_redis_client():
    """
    Create and return a Redis client.

    Defaults to localhost for local development.
    Later, Docker Compose will provide the Redis hostname.
    """

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    return redis.from_url(
        redis_url,
        decode_responses=True
    )