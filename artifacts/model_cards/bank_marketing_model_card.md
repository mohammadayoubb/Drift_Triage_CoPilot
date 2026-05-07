# Bank Marketing Model Card

## Overview

- **Model name:** `bank-marketing-classifier`
- **Task:** Binary classification
- **Target:** `y`
- **Positive class:** client subscribed to a term deposit
- **Created at UTC:** 2026-05-07T08:06:40.646273+00:00

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

- **Model artifact path:** `C:\Users\LEGION\Desktop\AIE\Assignments\Week5\Code\drift-triage-copilot\artifacts\models\bank_marketing_pipeline.joblib`
- **Model artifact SHA256:** `d7c371be39efaa9fcfbf1b61dc57a605703f180349d85091e2f0b12968d8771f`

## Environment fingerprint

```json
{
  "python": "3.12.3 (tags/v3.12.3:f6650f9, Apr  9 2024, 14:05:25) [MSC v.1938 64 bit (AMD64)]",
  "platform": "Windows-11-10.0.26200-SP0",
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
