# drops leaking columnss

import pandas as pd

from ml.constants import LEAKAGE_COLUMNS, MODEL_FEATURES, TARGET


def clean_bank_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    df["pdays_was_999"] = (df["pdays"] == 999).astype(int)

    df[TARGET] = df[TARGET].map({"yes": 1, "no": 0})

    if df[TARGET].isna().any():
        raise ValueError("Target mapping failed. Expected only 'yes' and 'no'.")

    df = df.drop(columns=LEAKAGE_COLUMNS)

    final_columns = MODEL_FEATURES + [TARGET]
    df = df[final_columns]

    return df