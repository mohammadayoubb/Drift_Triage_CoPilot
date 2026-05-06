# src/model_service/main.py

import logging

from fastapi import FastAPI, HTTPException, Query, status

from common.logging import configure_logging
from model_service.drift import build_drift_report
from model_service.model_loader import get_model_status
from model_service.predict import predict_subscription
from model_service.schemas import HealthResponse, PredictionRequest, PredictionResponse
from model_service.webhooks import emit_drift_webhook_if_needed

configure_logging()

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Drift Triage Copilot - Model Service",
    version="0.1.0",
    description="Serves and monitors the Bank Marketing subscription classifier.",
)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Drift Triage Copilot - Model Service",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
        "drift_report": "/drift/report",
    }


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    status_payload = get_model_status()

    return HealthResponse(
        status="ok",
        model_loaded=status_payload["model_loaded"],
        threshold_loaded=status_payload["threshold_loaded"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return predict_subscription(request)

    except FileNotFoundError as exc:
        logger.exception("Required model artifact missing")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "model_artifact_missing",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception("Prediction failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "prediction_failed",
                "message": "Prediction failed. Check service logs for details.",
            },
        ) from exc


@app.get("/drift/report")
def drift_report(
    window_size: int = Query(default=200, ge=30, le=10_000),
    emit_webhook: bool = Query(default=True),
) -> dict:
    """
    Build a drift report from recent predictions.

    If emit_webhook=true, the service emits a webhook when severity changes.
    """
    try:
        report = build_drift_report(window_size=window_size)

        webhook_emitted = False
        if emit_webhook and report.get("status") == "ok":
            webhook_emitted = emit_drift_webhook_if_needed(report)

        return {
            "report": report,
            "webhook_emitted": webhook_emitted,
        }

    except FileNotFoundError as exc:
        logger.exception("Reference stats missing")

        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "reference_stats_missing",
                "message": str(exc),
            },
        ) from exc

    except Exception as exc:
        logger.exception("Drift report failed")

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "drift_report_failed",
                "message": "Drift report failed. Check service logs for details.",
            },
        ) from exc