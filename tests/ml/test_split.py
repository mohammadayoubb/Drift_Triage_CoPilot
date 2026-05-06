# tests/ml/test_split.py

import pandas as pd

from ml.split import split_train_val_test


def test_split_sizes_are_60_20_20() -> None:
    df = pd.DataFrame(
        {
            "feature": range(100),
            "y": [0] * 80 + [1] * 20,
        }
    )

    train_df, val_df, test_df = split_train_val_test(df)

    assert len(train_df) == 60
    assert len(val_df) == 20
    assert len(test_df) == 20