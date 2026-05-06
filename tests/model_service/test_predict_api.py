# tests/model_service/test_predict_api.py

from fastapi.testclient import TestClient

from model_service.main import app

client = TestClient(app)


def valid_payload() -> dict:
    return {
        "age": 35,
        "job": "admin.",
        "marital": "married",
        "education": "university.degree",
        "default": "no",
        "housing": "yes",
        "loan": "no",
        "contact": "cellular",
        "month": "may",
        "day_of_week": "mon",
        "campaign": 1,
        "pdays": 999,
        "previous": 0,
        "poutcome": "nonexistent",
        "emp.var.rate": 1.1,
        "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4,
        "euribor3m": 4.857,
        "nr.employed": 5191.0,
    }


def test_health_endpoint_returns_status() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    payload = response.json()

    assert payload["status"] == "ok"
    assert "model_loaded" in payload
    assert "threshold_loaded" in payload


def test_predict_rejects_duration_leakage_field() -> None:
    payload = valid_payload()
    payload["duration"] = 120

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_returns_probability_and_label() -> None:
    response = client.post("/predict", json=valid_payload())

    assert response.status_code == 200

    payload = response.json()

    assert "request_id" in payload
    assert payload["model_name"] == "bank-marketing-classifier"
    assert 0 <= payload["probability"] <= 1
    assert payload["prediction"] in [0, 1]
    assert payload["label"] in ["yes", "no"]
    assert 0 <= payload["threshold"] <= 1