# src/model_service/predict.py

import logging
from uuid import uuid4

from model_service.model_loader import get_model_name, load_model, load_threshold
from model_service.prediction_log import append_prediction_log
from model_service.schemas import PredictionRequest, PredictionResponse

logger = logging.getLogger(__name__)


def predict_subscription(request: PredictionRequest) -> PredictionResponse:
    """
    Run one prediction using the trained sklearn pipeline.
    """
    request_id = str(uuid4())

    logger.info("Starting prediction request: %s", request_id)

    model = load_model()
    threshold = load_threshold()
    model_name = get_model_name()

    X = request.to_model_dataframe()

    probability = float(model.predict_proba(X)[:, 1][0])
    prediction = int(probability >= threshold)
    label = "yes" if prediction == 1 else "no"

    append_prediction_log(
        request_id=request_id,
        model_name=model_name,
        X=X,
        probability=probability,
        prediction=prediction,
        threshold=threshold,
    )

    logger.info(
        "Prediction completed: request_id=%s probability=%.4f prediction=%s threshold=%.4f",
        request_id,
        probability,
        prediction,
        threshold,
    )

    return PredictionResponse(
        request_id=request_id,
        model_name=model_name,
        probability=probability,
        prediction=prediction,
        threshold=threshold,
        label=label,
    )