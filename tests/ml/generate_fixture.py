#!/usr/bin/env python
"""
Run once to generate the fidelity test fixture, then commit the .npy file.
Usage: python tests/ml/generate_fixture.py
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ARTIFACTS_ROOT = Path("artifacts")
MODEL_PATH = ARTIFACTS_ROOT / "models" / "bank_marketing_pipeline.joblib"
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

model = joblib.load(MODEL_PATH)
test_df = pd.read_csv(TEST_DATA_PATH)
y_proba = model.predict_proba(test_df[MODEL_FEATURES])[:, 1]

FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
np.save(FIXTURE_PATH, y_proba)
print(f"Fixture saved to {FIXTURE_PATH} ({len(y_proba)} samples)")
