"""
time_utils.py

This file provides standardized timestamp generation.

All timestamps should be in UTC ISO format for consistency across services.
"""

from datetime import datetime, timezone


def get_utc_timestamp() -> str:
    """
    Return current UTC timestamp in ISO format.
    """

    return datetime.now(timezone.utc).isoformat()