"""
registry_schema.py

This file defines request/response schemas for registry operations.

It ensures promotion requests are structured and contain explicit approval
information before any Production change is attempted.
"""

from pydantic import BaseModel, ConfigDict


class PromotionRequest(BaseModel):
    """
    Request schema for promoting a model version to Production.
    """

    model_config = ConfigDict(extra="forbid")

    candidate_version: str
    approved: bool
    approved_by: str | None = None
    approval_reason: str | None = None


class PromotionResponse(BaseModel):
    """
    Response schema for promotion attempts.
    """

    status: str
    model_name: str | None = None
    promoted_version: str | None = None
    stage: str | None = None
    reason: str | None = None