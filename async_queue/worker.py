"""
worker.py

Redis queue worker with exponential backoff retry and dead-letter queue support.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone

from async_queue.dlq import DeadLetterQueue
from async_queue.idempotency import IdempotencyStore
from async_queue.redis_client import get_redis_client
from async_queue.tasks import run_replay_test, run_retrain, run_rollback

logger = logging.getLogger(__name__)

QUEUE_NAME = "drift_jobs"
COMPLETED_QUEUE_NAME = "drift_jobs_completed"
RUNNING_HASH_NAME = "drift_jobs_running"
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
        """
        Execute a job with exponential backoff.
        Both raised exceptions and task-returned {"status": "failed"} trigger retries.
        Sends a structured DLQ record after MAX_RETRIES.
        """
        last_error: str | None = None

        for attempt in range(MAX_RETRIES):
            result = None
            try:
                result = self.handle_job(job)
            except Exception as exc:
                last_error = str(exc)

            # Explicit failure result counts the same as a raised exception
            if result is not None and result.get("status") != "failed":
                return result  # success

            if result is not None:
                last_error = result.get("error") or "Task returned status=failed"

            if attempt < MAX_RETRIES - 1:
                delay = BASE_DELAY_SECONDS ** attempt
                logger.warning(
                    "Job attempt %d/%d failed, retrying in %ds: job_id=%s error=%s",
                    attempt + 1,
                    MAX_RETRIES,
                    delay,
                    job.get("job_id"),
                    last_error,
                )
                time.sleep(delay)

        dlq_record = {
            "job_id": job.get("job_id"),
            "job_type": job.get("job_type"),
            "investigation_id": job.get("payload", {}).get("investigation_id", "unknown"),
            "error": last_error or "Unknown error after max retries",
            "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        logger.error(
            "Job failed after %d attempts, sending to DLQ: job_id=%s error=%s",
            MAX_RETRIES,
            job.get("job_id"),
            last_error,
        )
        self.dlq.push(dlq_record)
        return None

    def _enqueue_retrain_job(self, investigation_id: str, result: dict) -> None:
        """
        After a successful replay_test, automatically enqueue a retrain job.
        Passes replay_metrics through the job payload so the promotion approval
        can show both sets of metrics side-by-side.
        Non-fatal — logs on failure but does not raise.
        """
        try:
            from async_queue.producer import QueueProducer
            from async_queue.job_models import QueueJob
            from async_queue.idempotency import IdempotencyStore
            from uuid import uuid4

            replay_metrics = (result or {}).get("metrics") or {}
            idempotency_key = f"{investigation_id}-retrain"
            idem = IdempotencyStore(namespace="queued")

            if idem.already_seen(idempotency_key):
                logger.info(
                    "retrain job already queued (idempotency): investigation_id=%s",
                    investigation_id,
                )
                return

            job = QueueJob(
                job_id=str(uuid4()),
                job_type="retrain",
                payload={
                    "investigation_id": investigation_id,
                    "replay_metrics": replay_metrics,
                },
                idempotency_key=idempotency_key,
            )
            QueueProducer().enqueue(job)
            idem.mark_seen(idempotency_key)
            logger.info(
                "retrain job enqueued after replay_test: investigation_id=%s job_id=%s",
                investigation_id,
                job.job_id,
            )
        except Exception as exc:
            logger.warning("Could not enqueue retrain job: %s", exc)

    def _maybe_create_promotion_approval(
        self,
        investigation_id: str,
        result: dict,
        job_payload: dict,
    ) -> None:
        """
        After a successful retrain, create a promote_candidate_model approval so
        a human can decide whether to promote the candidate to Production.
        Includes both replay and retrain metrics in the payload for the UI.
        Non-fatal — logs on failure but does not raise.
        """
        try:
            from approvals.approval_service import ApprovalService

            replay_metrics = (job_payload or {}).get("replay_metrics") or {}
            retrain_metrics = (result or {}).get("metrics") or {}
            model_version = (result or {}).get("model_version")

            ApprovalService().create_request(
                investigation_id=investigation_id,
                action_type="promote_candidate_model",
                reason=(
                    "Candidate model trained and registered in Staging. "
                    "Human approval is required before promoting to Production."
                ),
                payload={
                    "investigation_id": investigation_id,
                    "source_job_type": "retrain",
                    "replay_metrics": replay_metrics,
                    "retrain_metrics": retrain_metrics,
                    "model_version": model_version,
                },
            )
            logger.info(
                "Promotion approval created after retrain: investigation_id=%s model_version=%s",
                investigation_id,
                model_version,
            )
        except Exception as exc:
            logger.warning("Could not create promotion approval: %s", exc)

    def run_once(self) -> dict:
        raw_job = self.redis_client.lpop(QUEUE_NAME)

        if raw_job is None:
            return {"status": "empty", "message": "No jobs available."}

        job_id = None
        job_type = None
        try:
            job = json.loads(raw_job)
            job_id = job.get("job_id")
            job_type = job.get("job_type")
            idempotency_key = job["idempotency_key"]
            investigation_id = job.get("payload", {}).get("investigation_id", "unknown")

            if self.idempotency_store.already_seen(idempotency_key):
                logger.info(
                    "Worker skipped duplicate processed job: job_id=%s idempotency_key=%s",
                    job_id,
                    idempotency_key,
                )
                return {
                    "status": "skipped",
                    "reason": "Duplicate: job already processed.",
                    "job_id": job_id,
                }

            logger.info(
                "Worker started job: job_id=%s job_type=%s investigation_id=%s",
                job_id,
                job_type,
                investigation_id,
            )

            # Mark as running so the dashboard can see in-progress jobs.
            # Use a hash keyed by job_id so cleanup is O(1) regardless of
            # how many concurrent jobs exist.
            if job_id:
                running_record_json = json.dumps({
                    "job_id": job_id,
                    "job_type": job_type,
                    "investigation_id": investigation_id,
                    "started_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
                self.redis_client.hset(RUNNING_HASH_NAME, job_id, running_record_json)

            try:
                result = self._execute_with_retry(job)
            finally:
                # Always remove from running — whether the job succeeded, was
                # DLQ'd, or an unexpected exception propagated out.
                if job_id:
                    self.redis_client.hdel(RUNNING_HASH_NAME, job_id)

            # _execute_with_retry returns None only when it DLQ'd the job.
            # A non-None result that reached here has already passed the
            # status != "failed" guard inside _execute_with_retry.
            completed = result is not None
            if completed:
                self.idempotency_store.mark_seen(idempotency_key)
                logger.info(
                    "Worker completed job: job_id=%s job_type=%s investigation_id=%s",
                    job_id,
                    job_type,
                    investigation_id,
                )
                completed_record = {
                    "job_id": job_id,
                    "job_type": job_type,
                    "investigation_id": investigation_id,
                    "status": "completed",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "result_summary": {
                        k: v for k, v in result.items()
                        if k in ("status", "metrics", "report_path", "mlflow_run_id",
                                 "threshold", "rolled_back_to_version", "error")
                    },
                }
                self.redis_client.rpush(COMPLETED_QUEUE_NAME, json.dumps(completed_record))

                if job_type == "replay_test":
                    self._enqueue_retrain_job(investigation_id, result)
                elif job_type == "retrain":
                    self._maybe_create_promotion_approval(
                        investigation_id, result, job.get("payload", {})
                    )

            else:
                logger.warning(
                    "Worker job exhausted retries and was sent to DLQ: "
                    "job_id=%s job_type=%s investigation_id=%s",
                    job_id,
                    job_type,
                    investigation_id,
                )

            outcome = "completed" if completed else "failed"
            return {
                "status": outcome,
                "job_id": job_id,
                "result": result,
            }

        except Exception as error:
            logger.exception(
                "Worker unexpected exception: job_id=%s job_type=%s error=%s",
                job_id,
                job_type,
                error,
            )
            self.dlq.push({
                "job_id": job_id,
                "job_type": job_type,
                "investigation_id": None,
                "error": str(error),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            return {"status": "failed", "error": str(error)}

    def run_forever(self, sleep_seconds: float = 1.0) -> None:
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        logger.info(
            "Worker starting — queue=%s redis=%s",
            QUEUE_NAME,
            redis_url,
        )
        while True:
            result = self.run_once()
            if result.get("status") not in ("empty", "skipped"):
                logger.info("Worker tick: %s", result)
            time.sleep(sleep_seconds)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    QueueWorker().run_forever()
