# tests/ml/test_model_fidelity.py
"""
Verifies the saved model produces bit-exact predictions on the test set.
Run tests/ml/generate_fixture.py once to generate the reference .npy file,
then commit it so this test can pass in CI.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ARTIFACTS_ROOT = Path("artifacts")
MODEL_PATH = ARTIFACTS_ROOT / "models" / "bank_marketing_pipeline.joblib"
THRESHOLD_PATH = ARTIFACTS_ROOT / "reports" / "threshold.json"
TEST_DATA_PATH = Path("data") / "processed" / "test.csv"
FIXTURE_PATH = Path("tests") / "ml" / "fixtures" / "reference_proba.npy"

NUMERIC_FEATURES = [
    "age", "campaign", "pdays", "previous", "emp.var.rate",
    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed", "pdays_was_999",
]
CATEGORICAL_FEATURES = [
    "job", "marital", "education", "default", "housing", "loan",
    "contact", "month", "day_of_week", "poutcome",
]
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@pytest.fixture(scope="module")
def model_and_threshold():
    if not MODEL_PATH.exists():
        pytest.skip("Model artifact not found — run scripts/train_register_model.py first")
    model = joblib.load(MODEL_PATH)
    threshold = json.load(open(THRESHOLD_PATH))["threshold"]
    return model, threshold


@pytest.fixture(scope="module")
def test_data():
    if not TEST_DATA_PATH.exists():
        pytest.skip("Test split not found — run scripts/make_splits.py first")
    return pd.read_csv(TEST_DATA_PATH)


def test_model_recall_meets_threshold(model_and_threshold, test_data):
    """Recall on test set must be >= 0.74 (assignment requires >= 0.75 on val)."""
    from sklearn.metrics import recall_score

    model, threshold = model_and_threshold
    y_true = test_data["y"].values
    y_proba = model.predict_proba(test_data[MODEL_FEATURES])[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    recall = recall_score(y_true, y_pred)
    assert recall >= 0.74, f"Test recall {recall:.3f} is below acceptable range"


def test_model_fidelity(model_and_threshold, test_data):
    """Model must produce bit-exact probabilities — verifies no silent retraining."""
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture not found at {FIXTURE_PATH}. "
            "Generate with: python tests/ml/generate_fixture.py"
        )
    model, _ = model_and_threshold
    y_proba = model.predict_proba(test_data[MODEL_FEATURES])[:, 1]
    reference = np.load(FIXTURE_PATH)
    np.testing.assert_allclose(
        y_proba,
        reference,
        rtol=1e-12,
        atol=1e-12,
        err_msg="Model probabilities differ from reference fixture — unexpected retraining?",
    )
