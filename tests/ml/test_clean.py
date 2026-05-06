# tests/ml/test_clean.py

import pandas as pd

from ml.clean import clean_bank_data


def test_clean_drops_duration_and_maps_target() -> None:
    raw_df = pd.DataFrame(
        {
            "age": [30, 40],
            "job": ["admin.", "unknown"],
            "marital": ["single", "married"],
            "education": ["university.degree", "high.school"],
            "default": ["no", "unknown"],
            "housing": ["yes", "no"],
            "loan": ["no", "yes"],
            "contact": ["cellular", "telephone"],
            "month": ["may", "jun"],
            "day_of_week": ["mon", "tue"],
            "duration": [120, 240],
            "campaign": [1, 2],
            "pdays": [999, 6],
            "previous": [0, 1],
            "poutcome": ["nonexistent", "success"],
            "emp.var.rate": [1.1, -1.8],
            "cons.price.idx": [93.994, 92.893],
            "cons.conf.idx": [-36.4, -46.2],
            "euribor3m": [4.857, 1.313],
            "nr.employed": [5191.0, 5099.1],
            "y": ["no", "yes"],
        }
    )

    clean_df = clean_bank_data(raw_df)

    assert "duration" not in clean_df.columns
    assert clean_df["y"].tolist() == [0, 1]
    assert clean_df["pdays_was_999"].tolist() == [1, 0]
    assert "unknown" in clean_df["job"].values