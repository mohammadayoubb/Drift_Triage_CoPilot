"""
routes_predictions.py

This file defines the prediction API endpoint.

It validates requests, runs predictions, stores prediction records, and returns
a structured response.
"""

from fastapi import APIRouter

from service.validation.request_schema import PredictionRequest, PredictionResponse
from service.model.predictor import ModelPredictor
from service.model.loader import ModelLoader
from service.storage.prediction_store import PredictionStore
from service.utils.id_generator import generate_request_id
from service.utils.time_utils import get_utc_timestamp
from service.config.settings import Settings

router = APIRouter()

# Load model once when this module starts
# NEW (MLflow-based loader)
loader = ModelLoader().load()
predictor = ModelPredictor(
    pipeline=loader.pipeline,
    threshold=loader.threshold,
    model_name=loader.model_name,
    model_version=loader.model_version
)

prediction_store = PredictionStore()


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    """

    request_id = generate_request_id()
    timestamp = get_utc_timestamp()

    # API-safe names for predictor
    request_data = request.model_dump(by_alias=False)

    # Original training column names for logging and drift monitoring
    logged_features = request.model_dump(by_alias=True)

    result = predictor.predict(request_data)

    prediction_store.append({
        "request_id": request_id,
        "timestamp": timestamp,
        "features": logged_features,
        "prediction": result["prediction"],
        "probability_yes": result["probability_yes"],
        "threshold_used": result["threshold_used"],
        "model_name": result["model_name"],
        "model_version": result["model_version"]
    })

    result["request_id"] = request_id

    return result