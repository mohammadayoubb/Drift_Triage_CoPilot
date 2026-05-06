"""
worker.py

This file runs the Redis queue worker.

The worker:
- pulls jobs from Redis
- checks idempotency
- executes the matching task
- sends failed jobs to the DLQ
"""

import json
import time

from async_queue.redis_client import get_redis_client
from async_queue.idempotency import IdempotencyStore
from async_queue.dlq import DeadLetterQueue
from async_queue.tasks import run_replay_test, run_retrain, run_rollback


QUEUE_NAME = "drift_jobs"


class QueueWorker:
    """
    Redis-backed worker for slow tool execution.
    """

    def __init__(self):
        self.redis_client = get_redis_client()
        self.idempotency_store = IdempotencyStore()
        self.dlq = DeadLetterQueue()

    def handle_job(self, job: dict) -> dict:
        """
        Execute a job based on job_type.
        """

        job_type = job["job_type"]
        payload = job.get("payload", {})

        if job_type == "replay_test":
            return run_replay_test(payload)

        if job_type == "retrain":
            return run_retrain(payload)

        if job_type == "rollback":
            return run_rollback(payload)

        raise ValueError(f"Unknown job_type: {job_type}")

    def run_once(self) -> dict:
        """
        Process one job from the queue.
        """

        raw_job = self.redis_client.lpop(QUEUE_NAME)

        if raw_job is None:
            return {
                "status": "empty",
                "message": "No jobs available.",
            }

        try:
            job = json.loads(raw_job)
            idempotency_key = job["idempotency_key"]

            if self.idempotency_store.already_seen(idempotency_key):
                return {
                    "status": "skipped",
                    "reason": "Duplicate idempotency key.",
                    "job_id": job["job_id"],
                }

            result = self.handle_job(job)
            self.idempotency_store.mark_seen(idempotency_key)

            return {
                "status": "completed",
                "job_id": job["job_id"],
                "result": result,
            }

        except Exception as error:
            failed_job = {
                "raw_job": raw_job,
                "error": str(error),
            }
            self.dlq.push(failed_job)

            return {
                "status": "failed",
                "error": str(error),
            }

    def run_forever(self, sleep_seconds: float = 1.0) -> None:
        """
        Continuously process jobs.
        """

        while True:
            result = self.run_once()
            print(result)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    QueueWorker().run_forever()