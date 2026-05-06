# src/agent/persistence/investigation_store.py

import json
import logging
from datetime import datetime, timezone
from threading import Lock
from typing import Any
from uuid import uuid4

from common.paths import INVESTIGATIONS_PATH, ensure_project_dirs
from agent.schemas.drift_event import DriftEvent
from agent.schemas.investigation import InvestigationRecord

logger = logging.getLogger(__name__)

_STORE_LOCK = Lock()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_store() -> dict[str, Any]:
    ensure_project_dirs()

    if not INVESTIGATIONS_PATH.exists():
        return {}

    return json.loads(INVESTIGATIONS_PATH.read_text(encoding="utf-8"))


def _write_store(payload: dict[str, Any]) -> None:
    ensure_project_dirs()

    INVESTIGATIONS_PATH.write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def create_investigation(event: DriftEvent) -> InvestigationRecord:
    investigation_id = f"inv_{uuid4().hex[:12]}"
    thread_id = investigation_id
    now = utc_now_iso()

    record = InvestigationRecord(
        investigation_id=investigation_id,
        thread_id=thread_id,
        status="opened",
        severity=event.severity,
        created_at_utc=now,
        updated_at_utc=now,
        drift_event=event.model_dump(),
    )

    with _STORE_LOCK:
        store = _read_store()
        store[investigation_id] = record.model_dump()
        _write_store(store)

    logger.info(
        "Investigation created: investigation_id=%s severity=%s",
        investigation_id,
        event.severity,
    )

    return record


def update_investigation(
    investigation_id: str,
    updates: dict[str, Any],
) -> InvestigationRecord:
    with _STORE_LOCK:
        store = _read_store()

        if investigation_id not in store:
            raise KeyError(f"Investigation not found: {investigation_id}")

        current = store[investigation_id]
        current.update(updates)
        current["updated_at_utc"] = utc_now_iso()

        store[investigation_id] = current
        _write_store(store)

    return InvestigationRecord(**current)


def get_investigation(investigation_id: str) -> InvestigationRecord:
    store = _read_store()

    if investigation_id not in store:
        raise KeyError(f"Investigation not found: {investigation_id}")

    return InvestigationRecord(**store[investigation_id])


def list_investigations() -> list[InvestigationRecord]:
    store = _read_store()

    return [
        InvestigationRecord(**record)
        for record in sorted(
            store.values(),
            key=lambda item: item.get("created_at_utc", ""),
            reverse=True,
        )
    ]