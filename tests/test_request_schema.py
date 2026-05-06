"""
test_request_schema.py

This test verifies the prediction request schema.

It ensures valid requests pass and forbidden extra fields are rejected.
"""

import pytest
from pydantic import ValidationError

from service.validation.request_schema import PredictionRequest


VALID_PAYLOAD = {
    "age": 35,
    "campaign": 1,
    "pdays": 999,
    "previous": 0,
    "emp.var.rate": 1.1,
    "cons.price.idx": 93.994,
    "cons.conf.idx": -36.4,
    "euribor3m": 4.857,
    "nr.employed": 5191.0,
    "pdays_was_999": 1,
    "job": "admin.",
    "marital": "married",
    "education": "university.degree",
    "default": "no",
    "housing": "yes",
    "loan": "no",
    "contact": "cellular",
    "month": "may",
    "day_of_week": "mon",
    "poutcome": "nonexistent",
}


def test_valid_prediction_request_passes():
    """
    A valid payload should create a PredictionRequest.
    """

    request = PredictionRequest(**VALID_PAYLOAD)

    assert request.age == 35
    assert request.emp_var_rate == 1.1
    assert request.nr_employed == 5191.0


def test_extra_field_is_rejected():
    """
    Extra fields such as duration should be rejected.
    """

    payload = dict(VALID_PAYLOAD)
    payload["duration"] = 120

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)


def test_target_field_is_rejected():
    """
    Target field y should not be accepted in prediction input.
    """

    payload = dict(VALID_PAYLOAD)
    payload["y"] = "yes"

    with pytest.raises(ValidationError):
        PredictionRequest(**payload)