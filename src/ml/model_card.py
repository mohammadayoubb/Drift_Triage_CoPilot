# src/ml/model_card.py

import json
import platform
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from typing import Any
import logging
logger = logging.getLogger(__name__)


from common.hashing import sha256_file
from common.paths import (
    MODEL_CARD_PATH,
    MODEL_PATH,
    RAW_BANK_DATA_PATH,
    ensure_project_dirs,
)
from ml.constants import (
    CATEGORICAL_FEATURES,
    LEAKAGE_COLUMNS,
    MLFLOW_MODEL_NAME,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    RECALL_TARGET,
    TARGET,
)


def safe_package_version(package_name: str) -> str:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not-installed"


def build_environment_fingerprint() -> dict[str, str]:
    return {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "pandas": safe_package_version("pandas"),
        "numpy": safe_package_version("numpy"),
        "scikit_learn": safe_package_version("scikit-learn"),
        "mlflow": safe_package_version("mlflow"),
        "joblib": safe_package_version("joblib"),
    }


def write_model_card(metrics_payload: dict[str, Any]) -> None:
    ensure_project_dirs()

    raw_data_hash = sha256_file(RAW_BANK_DATA_PATH)
    model_hash = sha256_file(MODEL_PATH) if MODEL_PATH.exists() else "model-not-saved-yet"
    environment = build_environment_fingerprint()

    threshold = metrics_payload.get("threshold")
    validation_metrics = metrics_payload.get("validation", {})
    test_metrics = metrics_payload.get("test", {})

    numeric_features_json = json.dumps(NUMERIC_FEATURES, indent=2)
    categorical_features_json = json.dumps(CATEGORICAL_FEATURES, indent=2)
    model_features_json = json.dumps(MODEL_FEATURES, indent=2)
    validation_metrics_json = json.dumps(validation_metrics, indent=2)
    test_metrics_json = json.dumps(test_metrics, indent=2)
    env_fingerprint_json = json.dumps(environment, indent=2)

    content = f"""# Bank Marketing Model Card

## Overview

- **Model name:** `{MLFLOW_MODEL_NAME}`
- **Task:** Binary classification
- **Target:** `{TARGET}`
- **Positive class:** client subscribed to a term deposit
- **Created at UTC:** {datetime.now(timezone.utc).isoformat()}

## Dataset

- **Dataset:** UCI Bank Marketing / bank-additional-full.csv
- **Raw data SHA256:** `{raw_data_hash}`
- **Split:** stratified 60/20/20
- **Random state:** 42

## Important preprocessing decisions

- Dropped leakage columns: `{LEAKAGE_COLUMNS}`
- Treated `pdays == 999` as a sentinel by adding `pdays_was_999`
- Kept `"unknown"` as a valid category
- Converted target values: `yes -> 1`, `no -> 0`

## Features

### Numeric features

```text
{numeric_features_json}
```

### Categorical features

```text
{categorical_features_json}
```

### Final model features

```text
{model_features_json}
```

## Model

- **Pipeline:** sklearn Pipeline
- **Preprocessor:** ColumnTransformer
- **Classifier:** LogisticRegression
- **Class imbalance handling:** class_weight="balanced"

## Operating threshold

- **Threshold:** {threshold}
- **Rule:** highest threshold where validation recall >= {RECALL_TARGET}
- **Threshold tuned on:** validation set

## Validation metrics

```json
{validation_metrics_json}
```

## Test metrics

```json
{test_metrics_json}
```

## Artifact fingerprints

- **Model artifact path:** `{MODEL_PATH}`
- **Model artifact SHA256:** `{model_hash}`

## Environment fingerprint

```json
{env_fingerprint_json}
```

## Known limitations

- This model predicts campaign subscription likelihood from historical bank marketing data.
- It should not be interpreted as a general customer-value model.
- The live API must validate inputs before prediction.
- Drift monitoring is required because economic features like `euribor3m` and `cons.price.idx` may shift over time.
"""

    MODEL_CARD_PATH.write_text(content, encoding="utf-8")

    logger.info("Model card written to %s", MODEL_CARD_PATH)