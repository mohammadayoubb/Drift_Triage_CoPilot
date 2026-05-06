"""
routes_queue.py

This file exposes queue status endpoints.

The dashboard can later use this to show queue depth and DLQ depth.
"""

from fastapi import APIRouter

from async_queue.status import QueueStatus

router = APIRouter()


@router.get("/queue/status")
def queue_status():
    """
    Return current Redis queue and DLQ status.
    """

    return QueueStatus().get_status()