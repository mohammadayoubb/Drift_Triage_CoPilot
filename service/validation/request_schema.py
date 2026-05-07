"""
request_schema.py

This file defines the request and response schemas for the prediction API.

It ensures that every prediction request matches the exact feature schema used
during training. Invalid requests are rejected before reaching the model.
"""

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """
    Schema for one prediction request.

    This includes all 20 model input features:
    - 10 numeric features
    - 10 categorical features

    Extra fields are forbidden so leakage columns like `duration`
    or target columns like `y` cannot be accidentally sent.
    """

    # Reject any fields not explicitly defined below
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    # Numeric features
    age: int = Field(..., ge=0)
    campaign: int = Field(..., ge=0)
    pdays: int = Field(..., ge=0)
    previous: int = Field(..., ge=0)

    # Dataset column names contain dots, so aliases preserve original names
    emp_var_rate: float = Field(..., alias="emp.var.rate")
    cons_price_idx: float = Field(..., alias="cons.price.idx")
    cons_conf_idx: float = Field(..., alias="cons.conf.idx")
    euribor3m: float
    nr_employed: float = Field(..., alias="nr.employed")

    # Categorical features
    job: str
    marital: str
    education: str
    default: str
    housing: str
    loan: str
    contact: str
    month: str
    day_of_week: str
    poutcome: str


class PredictionResponse(BaseModel):
    """
    Schema for the prediction response returned by the API.
    """

    request_id: str
    prediction: str
    probability_yes: float
    threshold_used: float
    model_name: str
    model_version: str