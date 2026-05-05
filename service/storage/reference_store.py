"""
reference_store.py

This file loads reference data used for drift comparison.

The reference data normally comes from the training split and represents the
baseline distribution the live data is compared against.
"""

import pandas as pd


class ReferenceStore:
    """
    Loads reference data for drift monitoring.
    """

    def __init__(self, reference_path: str = "data/reference.csv"):
        self.reference_path = reference_path

    def load(self) -> pd.DataFrame:
        """
        Load reference data as a pandas DataFrame.
        """

        return pd.read_csv(self.reference_path)