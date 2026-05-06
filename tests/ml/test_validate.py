# tests/ml/test_validate.py

import pandas as pd
import pytest

from ml.constants import RAW_COLUMNS
from ml.validate import validate_raw_bank_data


def test_validate_rejects_missing_columns() -> None:
    df = pd.DataFrame({"age": [1], "y": ["yes"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_bank_data(df)


def test_validate_rejects_bad_target_values() -> None:
    data = {column: ["dummy"] for column in RAW_COLUMNS}
    data["age"] = [30]
    data["duration"] = [100]
    data["campaign"] = [1]
    data["pdays"] = [999]
    data["previous"] = [0]
    data["emp.var.rate"] = [1.1]
    data["cons.price.idx"] = [93.9]
    data["cons.conf.idx"] = [-36.4]
    data["euribor3m"] = [4.8]
    data["nr.employed"] = [5191.0]
    data["y"] = ["maybe"]

    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="Unexpected target values"):
        validate_raw_bank_data(df)