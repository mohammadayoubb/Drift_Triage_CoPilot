"""
routes_predictions.py

This file defines the prediction API endpoint.

It:
- Receives request
- Validates input (via Pydantic)
- Calls predictor
- Returns structured response
"""

from fastapi import APIRouter
from uuid import uuid4

from service.validation.request_schema import PredictionRequest, PredictionResponse
from service.model.predictor import ModelPredictor
from service.model.loader import ModelLoader

router = APIRouter()

# Load model once at startup
loader = ModelLoader(
    model_path="model_artifacts/bank_marketing_pipeline.joblib",
    metadata_path="model_artifacts/model_metadata.json"
).load()

predictor = ModelPredictor(
    pipeline=loader.pipeline,
    threshold=loader.threshold,
    model_name=loader.model_name,
    model_version=loader.model_version
)


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Main prediction endpoint.
    """

    # Generate unique request ID (useful for logging and debugging)
    request_id = str(uuid4())

    # Convert validated request into dictionary
    request_data = request.model_dump(by_alias=False)

    # Run prediction
    result = predictor.predict(request_data)

    # Attach request ID
    result["request_id"] = request_id

    return result