"""
routes_demo.py

Demo reset endpoint — archives all runtime state for a clean demo run.
Old files are moved to artifacts/archive/<timestamp>/ so nothing is lost.
"""

import json
import shutil
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter

from async_queue.redis_client import get_redis_client

router = APIRouter()

_ARCHIVE_FILES = [
    Path("data/predictions.jsonl"),
    Path("data/approvals.jsonl"),
    Path("artifacts/reports/latest_drift_report.json"),
    Path("artifacts/reports/predictions_log.csv"),
    Path("artifacts/agent_state/investigations.json"),
]

_ARCHIVE_DIRS = [
    Path("artifacts/reports/drift_reports"),
    Path("artifacts/reports/replay_reports"),
]

_DRIFT_STATE = Path("data/drift_state.json")

# Redis keys cleared on every reset (delete works for any key type)
_REDIS_LISTS = [
    "drift_jobs",
    "drift_jobs_dlq",
    "drift_jobs_completed",
    "drift_jobs_running",
]

# Key patterns cleared on every reset (prefix-matched via Redis KEYS)
_REDIS_KEY_PATTERNS = [
    "queued:*",
    "idempotency:*",
]


@router.post("/demo/reset")
def demo_reset():
    """
    Archive all runtime state and reset to a clean baseline for demo.

    Files are moved to artifacts/archive/<timestamp>/ — nothing is deleted.
    Redis queues, DLQ, completed-jobs list, and idempotency keys are cleared.
    Drift state is reset to LOW so the first drift check correctly fires
    the webhook when severity rises to CRITICAL.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(f"artifacts/archive/reset_{timestamp}")
    archive_root.mkdir(parents=True, exist_ok=True)

    # ── Archive files ─────────────────────────────────────────────────────────
    archived = []
    for path in _ARCHIVE_FILES:
        if path.exists():
            dest = archive_root / path.name
            shutil.move(str(path), str(dest))
            archived.append(str(path))

    for dir_path in _ARCHIVE_DIRS:
        if dir_path.exists() and any(dir_path.iterdir()):
            dest = archive_root / dir_path.name
            shutil.copytree(str(dir_path), str(dest))
            shutil.rmtree(dir_path)
        dir_path.mkdir(parents=True, exist_ok=True)

    # ── Reset drift baseline ──────────────────────────────────────────────────
    _DRIFT_STATE.parent.mkdir(parents=True, exist_ok=True)
    _DRIFT_STATE.write_text(json.dumps({"last_severity": "LOW"}, indent=2), encoding="utf-8")

    # ── Clear Redis runtime state ─────────────────────────────────────────────
    cleared_redis: list[str] = []
    try:
        rc = get_redis_client()

        for list_name in _REDIS_LISTS:
            if rc.exists(list_name):
                rc.delete(list_name)
                cleared_redis.append(list_name)

        for pattern in _REDIS_KEY_PATTERNS:
            matched = rc.keys(pattern)
            if matched:
                rc.delete(*matched)
                cleared_redis.extend(matched)
    except Exception as exc:
        cleared_redis.append(f"redis_error: {exc}")

    return {
        "status": "reset",
        "message": (
            f"Runtime state archived to {archive_root}. "
            "Redis queues and idempotency keys cleared. "
            "Drift baseline reset to LOW."
        ),
        "archive_path": str(archive_root),
        "archived": archived,
        "cleared_redis": cleared_redis,
    }
