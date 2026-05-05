"""
routes_test_payloads.py

This file exposes test payload endpoints.

These endpoints help during development and demos by returning valid example
requests that can be sent to the prediction endpoint.
"""

from fastapi import APIRouter

from service.utils.sample_payload import VALID_SAMPLE_PAYLOAD

router = APIRouter()


@router.get("/test-payload")
def get_test_payload():
    """
    Return a valid sample payload for prediction testing.
    """

    return VALID_SAMPLE_PAYLOAD