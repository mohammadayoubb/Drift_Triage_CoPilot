"""
queue_dispatcher.py

This file lets the agent dispatch slow tool jobs to Redis.

The agent should not execute slow tasks directly. It should enqueue replay,
retrain, and rollback jobs through this dispatcher.
"""

from uuid import uuid4

from async_queue.job_models import QueueJob
from async_queue.producer import QueueProducer


class QueueDispatcher:
    """
    Dispatches slow agent actions to Redis.
    """

    def __init__(self):
        self.producer = QueueProducer()

    def dispatch(self, job_type: str, payload: dict, idempotency_key: str) -> dict:
        """
        Create and enqueue a queue job.
        """

        job = QueueJob(
            job_id=str(uuid4()),
            job_type=job_type,
            payload=payload,
            idempotency_key=idempotency_key,
        )

        return self.producer.enqueue(job)