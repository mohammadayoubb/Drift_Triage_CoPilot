# checks if the raw DataFrame is healthy before we clean or split it

import pandas as pd

from ml.constants import RAW_COLUMNS, TARGET


def validate_raw_bank_data(df: pd.DataFrame) -> None:
    actual_columns = set(df.columns)
    expected_columns = set(RAW_COLUMNS)

    missing_columns = sorted(expected_columns - actual_columns)
    extra_columns = sorted(actual_columns - expected_columns)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if extra_columns:
        print(f"WARNING: Extra columns found and ignored later: {extra_columns}")

    if df.empty:
        raise ValueError("Dataset is empty.")

    target_values = set(df[TARGET].dropna().unique())
    expected_target_values = {"yes", "no"}

    if target_values != expected_target_values:
        raise ValueError(
            f"Unexpected target values in '{TARGET}'. "
            f"Expected {expected_target_values}, got {target_values}"
        )

    print("--- RAW DATA VALIDATION ---")
    print(f"Shape: {df.shape}")
    print("\nTarget counts:")
    print(df[TARGET].value_counts())

    print("\nTarget balance:")
    print(df[TARGET].value_counts(normalize=True))

    print("\nMissing values:")
    print(df.isna().sum().sort_values(ascending=False))

    print("\n'unknown' value counts:")
    print((df == "unknown").sum().sort_values(ascending=False).head(10))

    print("\npdays == 999 count:")
    print((df["pdays"] == 999).sum())

    print("\nValidation passed.")