# tests/model_service/test_drift.py

from model_service.drift import (
    calculate_psi,
    combine_severities,
    severity_from_output_delta,
    severity_from_psi,
)


def test_calculate_psi_is_zero_when_distributions_match() -> None:
    psi = calculate_psi(
        expected_proportions=[0.5, 0.5],
        actual_proportions=[0.5, 0.5],
    )

    assert psi == 0


def test_severity_from_psi() -> None:
    assert severity_from_psi(0.01) == "none"
    assert severity_from_psi(0.15) == "warning"
    assert severity_from_psi(0.30) == "critical"


def test_output_delta_severity() -> None:
    assert severity_from_output_delta(0.01) == "none"
    assert severity_from_output_delta(0.12) == "warning"
    assert severity_from_output_delta(0.25) == "critical"


def test_combine_severities() -> None:
    assert combine_severities(["none", "none"]) == "none"
    assert combine_severities(["none", "warning"]) == "warning"
    assert combine_severities(["warning", "critical"]) == "critical"