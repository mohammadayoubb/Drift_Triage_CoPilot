"""
worker.py

Redis queue worker with exponential backoff retry and dead-letter queue support.
"""

import json
import time

from async_queue.dlq import DeadLetterQueue
from async_queue.idempotency import IdempotencyStore
from async_queue.redis_client import get_redis_client
from async_queue.tasks import run_replay_test, run_retrain, run_rollback

QUEUE_NAME = "drift_jobs"
MAX_RETRIES = 3
BASE_DELAY_SECONDS = 2


class QueueWorker:
    """
    Redis-backed worker for slow tool execution.
    """

    def __init__(self):
        self.redis_client = get_redis_client()
        self.idempotency_store = IdempotencyStore()
        self.dlq = DeadLetterQueue()

    def handle_job(self, job: dict) -> dict:
        job_type = job["job_type"]
        payload = job.get("payload", {})

        if job_type == "replay_test":
            return run_replay_test(payload)
        if job_type == "retrain":
            return run_retrain(payload)
        if job_type == "rollback":
            return run_rollback(payload)

        raise ValueError(f"Unknown job_type: {job_type}")

    def _execute_with_retry(self, job: dict) -> dict | None:
        """Execute a job with exponential backoff. Sends to DLQ after MAX_RETRIES."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            try:
                return self.handle_job(job)
            except Exception as exc:
                last_exc = exc
                if attempt < MAX_RETRIES - 1:
                    delay = BASE_DELAY_SECONDS ** attempt
                    print(f"[worker] Attempt {attempt + 1} failed, retrying in {delay}s: {exc}")
                    time.sleep(delay)

        print(f"[worker] Job failed after {MAX_RETRIES} attempts, sending to DLQ: {last_exc}")
        self.dlq.push(json.dumps(job))
        return None

    def run_once(self) -> dict:
        raw_job = self.redis_client.lpop(QUEUE_NAME)

        if raw_job is None:
            return {"status": "empty", "message": "No jobs available."}

        try:
            job = json.loads(raw_job)
            idempotency_key = job["idempotency_key"]

            if self.idempotency_store.already_seen(idempotency_key):
                return {
                    "status": "skipped",
                    "reason": "Duplicate idempotency key.",
                    "job_id": job["job_id"],
                }

            result = self._execute_with_retry(job)
            if result is not None:
                self.idempotency_store.mark_seen(idempotency_key)

            return {
                "status": "completed" if result is not None else "failed",
                "job_id": job["job_id"],
                "result": result,
            }

        except Exception as error:
            self.dlq.push({"raw_job": raw_job, "error": str(error)})
            return {"status": "failed", "error": str(error)}

    def run_forever(self, sleep_seconds: float = 1.0) -> None:
        while True:
            result = self.run_once()
            print(result)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    QueueWorker().run_forever()
