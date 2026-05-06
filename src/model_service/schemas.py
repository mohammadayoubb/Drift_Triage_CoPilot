# Ussed aliases for the following fields due to python conflicts
# emp.var.rate
# cons.price.idx
# cons.conf.idx
# nr.employed

from typing import Any

import pandas as pd    
from pydantic import BaseModel, ConfigDict, Field

from ml.constants import MODEL_FEATURES


class PredictionRequest(BaseModel):
    """
    Incoming request body for one bank marketing prediction.

    We do not ask the client for:
    - duration: leakage column, forbidden
    - y: target, unknown at prediction time
    - pdays_was_999: engineered internally from pdays
    """

    model_config = ConfigDict(
        populate_by_name=True,
        extra="forbid",
    )

    age: int = Field(..., ge=0, le=120)

    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str

    campaign: int = Field(..., ge=0)
    pdays: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)
    poutcome: str

    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    def to_model_dataframe(self) -> pd.DataFrame:
        """
        Convert the validated request into a one-row DataFrame
        with the exact feature names and order expected by the sklearn pipeline.
        """
        payload: dict[str, Any] = self.model_dump(by_alias=True)

        payload["pdays_was_999"] = int(payload["pdays"] == 999)

        return pd.DataFrame([payload], columns=MODEL_FEATURES)


class PredictionResponse(BaseModel):
    request_id: str
    model_name: str
    probability: float
    prediction: int
    threshold: float
    label: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    threshold_loaded: bool