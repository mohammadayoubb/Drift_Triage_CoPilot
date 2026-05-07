# src/agent/persistence/checkpoints.py

import atexit
import logging
import os
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver

logger = logging.getLogger(__name__)

_CHECKPOINTER: Any | None = None
_CHECKPOINTER_CONTEXT: Any | None = None


def get_checkpointer() -> Any:
    """
    Return a LangGraph checkpointer.

    Default:
    - memory checkpointer for local development/tests.

    Optional:
    - postgres checkpointer when AGENT_CHECKPOINTER=postgres.
    """
    global _CHECKPOINTER, _CHECKPOINTER_CONTEXT

    if _CHECKPOINTER is not None:
        return _CHECKPOINTER

    mode = os.getenv("AGENT_CHECKPOINTER", "memory").lower()

    if mode == "postgres":
        database_url = os.getenv("DATABASE_URL")

        if not database_url:
            raise RuntimeError(
                "AGENT_CHECKPOINTER=postgres requires DATABASE_URL to be set."
            )

        # PostgresSaver uses psycopg3 directly (not SQLAlchemy), so strip any
        # driver prefix: "postgresql+psycopg://..." → "postgresql://..."
        if "://" in database_url:
            scheme, rest = database_url.split("://", 1)
            database_url = f"{scheme.split('+')[0]}://{rest}"

        from langgraph.checkpoint.postgres import PostgresSaver

        _CHECKPOINTER_CONTEXT = PostgresSaver.from_conn_string(database_url)
        _CHECKPOINTER = _CHECKPOINTER_CONTEXT.__enter__()

        # Creates checkpoint tables if they do not exist.
        _CHECKPOINTER.setup()

        atexit.register(_close_checkpointer_context)

        logger.info("Using Postgres LangGraph checkpointer")
        return _CHECKPOINTER

    logger.info("Using in-memory LangGraph checkpointer")
    _CHECKPOINTER = InMemorySaver()
    return _CHECKPOINTER


def _close_checkpointer_context() -> None:
    global _CHECKPOINTER_CONTEXT

    if _CHECKPOINTER_CONTEXT is not None:
        _CHECKPOINTER_CONTEXT.__exit__(None, None, None)
        _CHECKPOINTER_CONTEXT = None