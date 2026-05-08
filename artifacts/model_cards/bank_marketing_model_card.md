# Bank Marketing Model Card

## Overview

- **Model name:** `bank-marketing-classifier`
- **Task:** Binary classification
- **Target:** `y`
- **Positive class:** client subscribed to a term deposit
- **Created at UTC:** 2026-05-08T13:40:24.306065+00:00

## Dataset

- **Dataset:** UCI Bank Marketing / bank-additional-full.csv
- **Raw data SHA256:** `233e260d5d1d506a2c10381da5b8c2f75f2c08d1b373ca7b825e39ebe1bd30df`
- **Split:** stratified 60/20/20
- **Random state:** 42

## Important preprocessing decisions

- Dropped leakage columns: `['duration']`
- Treated `pdays == 999` as a sentinel by adding `pdays_was_999`
- Kept `"unknown"` as a valid category
- Converted target values: `yes -> 1`, `no -> 0`

## Features

### Numeric features

```text
[
  "age",
  "campaign",
  "pdays",
  "previous",
  "emp.var.rate",
  "cons.price.idx",
  "cons.conf.idx",
  "euribor3m",
  "nr.employed",
  "pdays_was_999"
]
```

### Categorical features

```text
[
  "job",
  "marital",
  "education",
  "default",
  "housing",
  "loan",
  "contact",
  "month",
  "day_of_week",
  "poutcome"
]
```

### Final model features

```text
[
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
  "job",
  "marital",
  "education",
  "default",
  "housing",
  "loan",
  "contact",
  "month",
  "day_of_week",
  "poutcome"
]
```

## Model

- **Pipeline:** sklearn Pipeline
- **Preprocessor:** ColumnTransformer
- **Classifier:** LogisticRegression
- **Class imbalance handling:** class_weight="balanced"

## Operating threshold

- **Threshold:** 0.385
- **Rule:** highest threshold where validation recall >= 0.75
- **Threshold tuned on:** validation set

## Validation metrics

```json
{
  "threshold": 0.385,
  "auc": 0.8017301081418935,
  "f1": 0.3700159489633174,
  "precision": 0.2455892731122089,
  "recall": 0.75,
  "positive_prediction_rate": 0.34401553775188154,
  "confusion_matrix": {
    "tn": 5172,
    "fp": 2138,
    "fn": 232,
    "tp": 696
  }
}
```

## Test metrics

```json
{
  "threshold": 0.385,
  "auc": 0.8012724656823436,
  "f1": 0.37082554100988513,
  "precision": 0.24653641207815274,
  "recall": 0.7478448275862069,
  "positive_prediction_rate": 0.3417091527069677,
  "confusion_matrix": {
    "tn": 5189,
    "fp": 2121,
    "fn": 234,
    "tp": 694
  }
}
```

## Artifact fingerprints

- **Model artifact path:** `/app/artifacts/models/bank_marketing_pipeline.joblib`
- **Model artifact SHA256:** `324d29eebe4492fddd0a6977143baefc7f2ea330e60745ab15ae0de492e03b95`

## Environment fingerprint

```json
{
  "python": "3.12.13 (main, Apr 22 2026, 02:08:00) [GCC 14.2.0]",
  "platform": "Linux-6.6.87.2-microsoft-standard-WSL2-x86_64-with-glibc2.41",
  "pandas": "2.3.3",
  "numpy": "2.4.4",
  "scikit_learn": "1.8.0",
  "mlflow": "3.12.0",
  "joblib": "1.5.3"
}
```

## Known limitations

- This model predicts campaign subscription likelihood from historical bank marketing data.
- It should not be interpreted as a general customer-value model.
- The live API must validate inputs before prediction.
- Drift monitoring is required because economic features like `euribor3m` and `cons.price.idx` may shift over time.
