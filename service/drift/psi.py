"""
psi.py

This file contains the PSI calculation used for numeric drift detection.

PSI = Population Stability Index.
It compares a reference distribution against a recent/live distribution.
"""

import numpy as np


def calculate_psi(reference_values, current_values, buckets: int = 10) -> float:
    """
    Calculate PSI for one numeric feature.
    """

    reference_values = np.asarray(reference_values)
    current_values = np.asarray(current_values)

    # Create bucket boundaries from reference distribution
    breakpoints = np.percentile(
        reference_values,
        np.linspace(0, 100, buckets + 1)
    )

    # Avoid duplicate breakpoints causing invalid bins
    breakpoints = np.unique(breakpoints)

    if len(breakpoints) < 2:
        return 0.0

    reference_counts, _ = np.histogram(reference_values, bins=breakpoints)
    current_counts, _ = np.histogram(current_values, bins=breakpoints)

    reference_percents = reference_counts / max(reference_counts.sum(), 1)
    current_percents = current_counts / max(current_counts.sum(), 1)

    # Prevent division by zero and log(0)
    epsilon = 1e-6
    reference_percents = np.clip(reference_percents, epsilon, None)
    current_percents = np.clip(current_percents, epsilon, None)

    psi = np.sum(
        (current_percents - reference_percents)
        * np.log(current_percents / reference_percents)
    )

    return float(psi)