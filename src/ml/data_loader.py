# Loads Data and handless separation issues

from pathlib import Path

import pandas as pd

from common.paths import RAW_BANK_DATA_PATH, TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH


def read_bank_csv(path: Path = RAW_BANK_DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: uv run python scripts/fetch_data.py"
        )

    df = pd.read_csv(path, sep=";")

    if df.shape[1] == 1:
        df = pd.read_csv(path)

    return df


def save_splits(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    val_df.to_csv(VAL_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(TRAIN_DATA_PATH),
        pd.read_csv(VAL_DATA_PATH),
        pd.read_csv(TEST_DATA_PATH),
    )