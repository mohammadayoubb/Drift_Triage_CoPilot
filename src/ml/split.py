import pandas as pd
from sklearn.model_selection import train_test_split

from ml.constants import RANDOM_STATE, TARGET


def split_train_val_test(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df, temp_df = train_test_split(
        df,
        test_size=0.40,
        random_state=RANDOM_STATE,
        stratify=df[TARGET],
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=temp_df[TARGET],
    )

    return train_df, val_df, test_df


def print_split_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    total = len(train_df) + len(val_df) + len(test_df)

    print("--- SPLIT REPORT ---")
    for name, split_df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:
        print(f"\n{name.upper()}")
        print(f"Rows: {len(split_df)}")
        print(f"Percent: {len(split_df) / total:.2%}")
        print("Target counts:")
        print(split_df[TARGET].value_counts())
        print("Target balance:")
        print(split_df[TARGET].value_counts(normalize=True))