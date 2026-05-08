"""
approval_models.py

This file defines approval request schemas.

Production-changing actions such as promote or rollback must create an approval
request before execution.
"""

from pydantic import BaseModel, ConfigDict
from typing import Any


class ApprovalRequest(BaseModel):
    """
    Human approval request contract.
    """

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    investigation_id: str
    action_type: str
    target_model_version: str | None = None
    reason: str
    payload: dict[str, Any]
    status: str = "pending"
    created_at_utc: str = ""


class ApprovalDecision(BaseModel):
    """
    Human approval decision contract.
    """

    model_config = ConfigDict(extra="forbid")

    approval_id: str
    approved: bool
    approved_by: str
    decision_reason: str | None = None