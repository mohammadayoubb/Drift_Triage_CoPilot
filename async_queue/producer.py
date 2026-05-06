"""
producer.py

This file pushes async jobs into Redis.

The agent will use this producer to dispatch slow tools like replay test,
retrain, and rollback without blocking the investigation flow.
"""

import json

from async_queue.job_models import QueueJob
from async_queue.redis_client import get_redis_client


QUEUE_NAME = "drift_jobs"


class QueueProducer:
    """
    Sends jobs to the Redis queue.
    """

    def __init__(self):
        self.redis_client = get_redis_client()

    def enqueue(self, job: QueueJob) -> dict:
        """
        Add a job to the Redis queue.
        """

        self.redis_client.rpush(
            QUEUE_NAME,
            job.model_dump_json()
        )

        return {
            "status": "queued",
            "queue": QUEUE_NAME,
            "job_id": job.job_id,
            "job_type": job.job_type,
        }