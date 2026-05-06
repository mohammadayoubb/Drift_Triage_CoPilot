# src/ml/constants.py

TARGET = "y"

RANDOM_STATE = 42
RECALL_TARGET = 0.75

MLFLOW_EXPERIMENT_NAME = "drift-triage-bank-marketing"
MLFLOW_MODEL_NAME = "bank-marketing-classifier"

RAW_COLUMNS = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "y",
]

LEAKAGE_COLUMNS = ["duration"]

NUMERIC_FEATURES = [
    "age",
    "campaign",
    "pdays",
    "previous",
    "emp.var.rate",
    "cons.price.idx",
    "cons.conf.idx",
    "euribor3m",
    "nr.employed",
    "pdays_was_999",
]

CATEGORICAL_FEATURES = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES