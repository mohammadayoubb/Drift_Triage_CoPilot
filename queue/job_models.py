"""
job_models.py

This file defines the shared job schemas for the async queue system.

Slow actions like replay test, retrain, and rollback are represented as
structured jobs before being pushed to Redis.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any


class QueueJob(BaseModel):
    """
    Standard async job contract.

    Every queued job must include:
    - job_id: unique job identifier
    - job_type: replay_test, retrain, or rollback
    - payload: job-specific data
    - idempotency_key: prevents duplicate execution
    """

    model_config = ConfigDict(extra="forbid")

    job_id: str
    job_type: str
    payload: dict[str, Any]
    idempotency_key: str
    status: str = "queued"