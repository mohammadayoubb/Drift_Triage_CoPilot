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

router = APIRouter()

_ARCHIVE_FILES = [
    Path("data/predictions.jsonl"),
    Path("artifacts/reports/latest_drift_report.json"),
    Path("artifacts/reports/predictions_log.csv"),
    Path("artifacts/agent_state/investigations.json"),
]

_ARCHIVE_DIRS = [
    Path("artifacts/reports/drift_reports"),
    Path("artifacts/reports/replay_reports"),
]

_DRIFT_STATE = Path("data/drift_state.json")


@router.post("/demo/reset")
def demo_reset():
    """
    Archive all runtime state and reset to a clean baseline for demo.

    Files are moved to artifacts/archive/<timestamp>/ — nothing is deleted.
    Drift state is reset to LOW so the first drift check correctly fires
    the webhook when severity rises to CRITICAL.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_root = Path(f"artifacts/archive/reset_{timestamp}")
    archive_root.mkdir(parents=True, exist_ok=True)

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

    # Write LOW as the baseline severity so the next drift check correctly
    # detects LOW → CRITICAL as a change and fires the webhook.
    _DRIFT_STATE.parent.mkdir(parents=True, exist_ok=True)
    _DRIFT_STATE.write_text(json.dumps({"last_severity": "LOW"}, indent=2), encoding="utf-8")

    return {
        "status": "archived",
        "message": f"Runtime state archived to {archive_root}. Drift baseline reset to LOW.",
        "archive_path": str(archive_root),
        "archived": archived,
    }
