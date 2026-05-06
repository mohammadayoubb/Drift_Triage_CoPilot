# tests/ml/test_pipeline.py

import pandas as pd

from ml.constants import MODEL_FEATURES
from ml.pipeline import build_model_pipeline


def make_sample_X() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [30, 45, 28, 60],
            "campaign": [1, 2, 1, 3],
            "pdays": [999, 6, 999, 3],
            "previous": [0, 1, 0, 2],
            "emp.var.rate": [1.1, -1.8, 1.1, -2.9],
            "cons.price.idx": [93.994, 92.893, 93.994, 92.201],
            "cons.conf.idx": [-36.4, -46.2, -36.4, -31.4],
            "euribor3m": [4.857, 1.313, 4.857, 0.883],
            "nr.employed": [5191.0, 5099.1, 5191.0, 5076.2],
            "pdays_was_999": [1, 0, 1, 0],
            "job": ["admin.", "technician", "unknown", "blue-collar"],
            "marital": ["single", "married", "single", "divorced"],
            "education": ["university.degree", "high.school", "unknown", "basic.9y"],
            "default": ["no", "unknown", "no", "no"],
            "housing": ["yes", "no", "yes", "no"],
            "loan": ["no", "yes", "no", "no"],
            "contact": ["cellular", "telephone", "cellular", "telephone"],
            "month": ["may", "jun", "jul", "aug"],
            "day_of_week": ["mon", "tue", "wed", "thu"],
            "poutcome": ["nonexistent", "success", "failure", "nonexistent"],
        }
    )


def test_pipeline_can_fit_and_predict_probabilities() -> None:
    X = make_sample_X()
    y = [0, 1, 0, 1]

    assert list(X.columns) == MODEL_FEATURES

    pipeline = build_model_pipeline()
    pipeline.fit(X, y)

    probabilities = pipeline.predict_proba(X)

    assert probabilities.shape == (4, 2)
    assert all(0 <= probability <= 1 for probability in probabilities[:, 1])