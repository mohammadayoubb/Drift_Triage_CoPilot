"""
test_worker.py

Tests queue worker task routing without requiring Redis.
"""

from async_queue.worker import QueueWorker


def test_worker_handles_replay_job():
    worker = QueueWorker()

    result = worker.handle_job({
        "job_type": "replay_test",
        "payload": {"model_version": "v1"},
    })

    assert result["status"] == "completed"
    assert result["task"] == "replay_test"


def test_worker_rejects_unknown_job_type():
    worker = QueueWorker()

    try:
        worker.handle_job({
            "job_type": "bad_job_type",
            "payload": {},
        })
    except ValueError as error:
        assert "Unknown job_type" in str(error)
    else:
        raise AssertionError("Expected ValueError for unknown job type")