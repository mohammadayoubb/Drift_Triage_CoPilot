"""
main.py

This is the entry point for the FastAPI model service.

It connects all API routes into one FastAPI application.
"""

from fastapi import FastAPI

from service.api.routes_predictions import router as predictions_router
from service.api.routes_health import router as health_router
from service.api.routes_registry import router as registry_router
from service.api.routes_drift import router as drift_router
from service.api.routes_test_payloads import router as test_payloads_router
from service.api.routes_queue import router as queue_router
from service.api.routes_approvals import router as approvals_router
from service.api.routes_demo import router as demo_router

app = FastAPI(
    title="Bank Marketing Model Service",
    description="FastAPI service for serving the Bank Marketing classifier.",
    version="0.1.0"
)

# Register routes
app.include_router(health_router, tags=["health"])
app.include_router(predictions_router, tags=["predictions"])
app.include_router(registry_router, tags=["registry"])
app.include_router(drift_router, tags=["drift"])
app.include_router(test_payloads_router, tags=["testing"])
app.include_router(queue_router, tags=["queue"])
app.include_router(approvals_router, tags=["approvals"])
app.include_router(demo_router, tags=["demo"])