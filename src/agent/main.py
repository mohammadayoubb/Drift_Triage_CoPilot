# src/agent/main.py

import logging

from fastapi import FastAPI

from .api.routes import router
from common.logging import configure_logging

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drift Triage Copilot - Agent",
    version="0.1.0",
    description="Receives drift events and opens LangGraph investigations.",
)

app.include_router(router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Drift Triage Copilot - Agent",
        "docs": "/docs",
        "health": "/health",
        "webhook": "/webhooks/drift",
        "investigations": "/investigations",
    }