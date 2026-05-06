"""
test_queue_models.py

This test verifies async queue job schemas.

It ensures queue jobs are structured and reject unexpected fields.
"""

import pytest
from pydantic import ValidationError

from async_queue.job_models import QueueJob


def test_queue_job_valid():
    """
    A valid queue job should be accepted.
    """

    job = QueueJob(
        job_id="j1",
        job_type="replay_test",
        payload={"model_version": "v1"},
        idempotency_key="replay-v1",
    )

    assert job.status == "queued"
    assert job.job_type == "replay_test"


def test_queue_job_rejects_extra_fields():
    """
    QueueJob should reject unknown fields.
    """

    with pytest.raises(ValidationError):
        QueueJob(
            job_id="j1",
            job_type="replay_test",
            payload={},
            idempotency_key="k1",
            unexpected="bad",
        )