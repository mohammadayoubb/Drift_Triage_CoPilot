"""
id_generator.py

This file generates unique IDs for requests, events, and investigations.

Using consistent IDs is critical for:
- Logging
- Debugging
- Agent investigations
- Traceability across services
"""

from uuid import uuid4


def generate_request_id() -> str:
    """
    Generate a unique request ID.
    """
    return str(uuid4())


def generate_event_id() -> str:
    """
    Generate a unique drift event ID.
    """
    return str(uuid4())