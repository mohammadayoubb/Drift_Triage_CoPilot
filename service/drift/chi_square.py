"""
chi_square.py

This file contains chi-square drift logic for categorical features.

It compares category counts from the reference data against recent/live data.
"""

import pandas as pd
from scipy.stats import chisquare


def calculate_chi_square(reference_values, current_values) -> dict:
    """
    Calculate chi-square statistic and p-value for one categorical feature.
    """

    reference_counts = pd.Series(reference_values).value_counts()
    current_counts = pd.Series(current_values).value_counts()

    # Align categories so both distributions have the same labels
    all_categories = reference_counts.index.union(current_counts.index)

    reference_counts = reference_counts.reindex(all_categories, fill_value=0)
    current_counts = current_counts.reindex(all_categories, fill_value=0)

    # Scale expected frequencies to match current sample size
    expected = reference_counts / max(reference_counts.sum(), 1)
    expected = expected * current_counts.sum()

    # Avoid zero expected values
    expected = expected.replace(0, 1e-6)

    statistic, p_value = chisquare(
        f_obs=current_counts,
        f_exp=expected
    )

    p_value = float(p_value)
    statistic = float(statistic)
    drifted = p_value < 0.05

    if p_value < 0.001:
        severity = "HIGH"
    elif p_value < 0.05:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "chi_square": statistic,
        "chi_square_statistic": statistic,
        "p_value": p_value,
        "drift_detected": bool(drifted),
        "is_drifted": bool(drifted),
        "severity": severity,
    }