# tests/model_service/test_prediction_log.py

import pandas as pd

from model_service.prediction_log import append_prediction_log, load_recent_predictions


def test_append_prediction_log_creates_file(tmp_path) -> None:
    path = tmp_path / "predictions_log.csv"

    X = pd.DataFrame(
        [
            {
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
        ]
    )

    append_prediction_log(
        request_id="test-request",
        model_name="test-model",
        X=X,
        probability=0.25,
        prediction=0,
        threshold=0.5,
        path=path,
    )

    df = load_recent_predictions(window_size=10, path=path)

    assert path.exists()
    assert len(df) == 1
    assert df.iloc[0]["request_id"] == "test-request"
    assert df.iloc[0]["prediction"] == 0