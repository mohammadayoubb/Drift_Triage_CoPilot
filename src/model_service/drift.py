# src/model_service/drift.py

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import chisquare

from common.paths import (
    DRIFT_REPORT_INDEX_PATH,
    DRIFT_REPORT_PATH,
    DRIFT_REPORTS_DIR,
    PROJECT_ROOT,
    REFERENCE_STATS_PATH,
)
from ml.constants import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from model_service.prediction_log import load_recent_predictions

logger = logging.getLogger(__name__)

DriftSeverity = Literal["none", "warning", "critical"]

LOCAL_TIMEZONE_NAME = "Asia/Beirut"
LOCAL_TIMEZONE = ZoneInfo(LOCAL_TIMEZONE_NAME)

PSI_WARNING_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25

CATEGORICAL_WARNING_P_VALUE = 0.05
CATEGORICAL_CRITICAL_P_VALUE = 0.01

OUTPUT_WARNING_DELTA = 0.10
OUTPUT_CRITICAL_DELTA = 0.20

MIN_ROWS_FOR_DRIFT = 30


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def format_utc_timestamp(dt: datetime) -> str:
    """
    Machine-friendly UTC timestamp.
    Example: 2026-05-06T07:36:48+00:00
    """
    return dt.isoformat(timespec="seconds")


def format_local_timestamp(dt: datetime) -> str:
    """
    Human-friendly Beirut timestamp.
    Example: 2026-05-06 10:36:48 Asia/Beirut
    """
    return dt.astimezone(LOCAL_TIMEZONE).strftime(
        f"%Y-%m-%d %H:%M:%S {LOCAL_TIMEZONE_NAME}"
    )


def format_filename_timestamp(dt: datetime) -> str:
    """
    Filename-safe Beirut timestamp.
    Example: 2026-05-06_10-36-48
    """
    return dt.astimezone(LOCAL_TIMEZONE).strftime("%Y-%m-%d_%H-%M-%S")


def relative_path(path: Path) -> str:
    """
    Return a clean project-relative path for reports.

    Falls back to absolute path if relative conversion fails.
    """
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def make_report_base_metadata(
    *,
    window_size: int,
    recent_count: int,
    severity: DriftSeverity,
    status: str,
) -> dict[str, Any]:
    now = utc_now()
    local_filename_timestamp = format_filename_timestamp(now)

    report_id = (
        f"drift_{local_filename_timestamp}_"
        f"{severity}_{status}_n{recent_count}_w{window_size}"
    )

    return {
        "report_id": report_id,
        "created_at_local": format_local_timestamp(now),
        "created_at_utc": format_utc_timestamp(now),
        "timezone": LOCAL_TIMEZONE_NAME,
        "window_size": int(window_size),
        "recent_count": int(recent_count),
        "severity": severity,
        "status": status,
    }


def load_reference_stats() -> dict[str, Any]:
    """
    Load reference distributions created during training.
    """
    if not REFERENCE_STATS_PATH.exists():
        raise FileNotFoundError(
            f"Reference stats not found at {REFERENCE_STATS_PATH}. "
            "Run: uv run python scripts/train_register_model.py"
        )

    return json.loads(REFERENCE_STATS_PATH.read_text(encoding="utf-8"))


def calculate_psi(
    expected_proportions: list[float],
    actual_proportions: list[float],
    epsilon: float = 1e-6,
) -> float:
    """
    Population Stability Index.

    PSI = sum((actual - expected) * ln(actual / expected))
    """
    if len(expected_proportions) != len(actual_proportions):
        raise ValueError(
            "PSI inputs must have the same length. "
            f"expected={len(expected_proportions)}, actual={len(actual_proportions)}"
        )

    expected = np.asarray(expected_proportions, dtype=float)
    actual = np.asarray(actual_proportions, dtype=float)

    expected = np.clip(expected, epsilon, None)
    actual = np.clip(actual, epsilon, None)

    expected = expected / expected.sum()
    actual = actual / actual.sum()

    return float(np.sum((actual - expected) * np.log(actual / expected)))


def severity_from_psi(psi: float) -> DriftSeverity:
    if psi >= PSI_CRITICAL_THRESHOLD:
        return "critical"
    if psi >= PSI_WARNING_THRESHOLD:
        return "warning"
    return "none"


def severity_from_p_value(p_value: float) -> DriftSeverity:
    if p_value <= CATEGORICAL_CRITICAL_P_VALUE:
        return "critical"
    if p_value <= CATEGORICAL_WARNING_P_VALUE:
        return "warning"
    return "none"


def severity_from_output_delta(delta: float) -> DriftSeverity:
    if delta >= OUTPUT_CRITICAL_DELTA:
        return "critical"
    if delta >= OUTPUT_WARNING_DELTA:
        return "warning"
    return "none"


def combine_severities(severities: list[DriftSeverity]) -> DriftSeverity:
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    return "none"


def compute_numeric_drift(
    recent_df: pd.DataFrame,
    reference_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute PSI for each numeric feature.
    """
    numeric_results: dict[str, Any] = {}
    reference_numeric = reference_stats.get("numeric_features", {})

    for feature in NUMERIC_FEATURES:
        if feature not in recent_df.columns:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": "missing_recent_feature",
            }
            continue

        ref = reference_numeric.get(feature)

        if not ref:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": "missing_reference_feature",
            }
            continue

        bin_edges = ref.get("bin_edges", [])
        expected_proportions = ref.get("proportions", [])

        if len(bin_edges) < 2 or not expected_proportions:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": "missing_reference_bins",
            }
            continue

        series = pd.to_numeric(recent_df[feature], errors="coerce").dropna()

        if series.empty:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": "no_recent_values",
            }
            continue

        edges = np.asarray(bin_edges, dtype=float).copy()

        # Include values outside the training range in edge buckets.
        edges[0] = -np.inf
        edges[-1] = np.inf

        actual_counts, _ = np.histogram(series, bins=edges)
        actual_total = actual_counts.sum()

        if actual_total == 0:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": "no_recent_values_after_binning",
            }
            continue

        actual_proportions = (actual_counts / actual_total).tolist()

        try:
            psi = calculate_psi(
                expected_proportions=expected_proportions,
                actual_proportions=actual_proportions,
            )
        except ValueError as exc:
            numeric_results[feature] = {
                "psi": None,
                "severity": "none",
                "reason": str(exc),
            }
            continue

        severity = severity_from_psi(psi)

        numeric_results[feature] = {
            "psi": psi,
            "severity": severity,
            "recent_count": int(series.shape[0]),
        }

    return numeric_results


def compute_categorical_drift(
    recent_df: pd.DataFrame,
    reference_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Compute chi-square drift for each categorical feature.
    """
    categorical_results: dict[str, Any] = {}
    reference_categorical = reference_stats.get("categorical_features", {})

    for feature in CATEGORICAL_FEATURES:
        if feature not in recent_df.columns:
            categorical_results[feature] = {
                "chi_square": None,
                "p_value": None,
                "severity": "none",
                "reason": "missing_recent_feature",
            }
            continue

        ref = reference_categorical.get(feature)

        if not ref:
            categorical_results[feature] = {
                "chi_square": None,
                "p_value": None,
                "severity": "none",
                "reason": "missing_reference_feature",
            }
            continue

        ref_proportions: dict[str, float] = ref.get("proportions", {})

        if not ref_proportions:
            categorical_results[feature] = {
                "chi_square": None,
                "p_value": None,
                "severity": "none",
                "reason": "missing_reference_proportions",
            }
            continue

        recent_series = recent_df[feature].fillna("__MISSING__").astype(str)
        recent_counts = recent_series.value_counts()

        categories = sorted(
            set(ref_proportions.keys()) | set(recent_counts.index.astype(str))
        )

        observed = np.array(
            [recent_counts.get(category, 0) for category in categories],
            dtype=float,
        )

        observed_total = observed.sum()

        if observed_total == 0:
            categorical_results[feature] = {
                "chi_square": None,
                "p_value": None,
                "severity": "none",
                "reason": "no_recent_values",
            }
            continue

        # Tiny fallback probability for categories never seen in training.
        raw_expected_proportions = np.array(
            [ref_proportions.get(category, 1e-6) for category in categories],
            dtype=float,
        )

        expected_proportions = (
            raw_expected_proportions / raw_expected_proportions.sum()
        )
        expected = expected_proportions * observed_total

        # scipy chisquare is strict that expected and observed sums match.
        expected = expected * (observed_total / expected.sum())

        chi_stat, p_value = chisquare(f_obs=observed, f_exp=expected)
        severity = severity_from_p_value(float(p_value))

        categorical_results[feature] = {
            "chi_square": float(chi_stat),
            "p_value": float(p_value),
            "severity": severity,
            "recent_count": int(observed_total),
            "observed_categories": int(len(categories)),
        }

    return categorical_results


def compute_output_drift(
    recent_df: pd.DataFrame,
    reference_stats: dict[str, Any],
) -> dict[str, Any]:
    """
    Compare recent positive prediction rate against reference positive prediction rate.
    """
    if "prediction" not in recent_df.columns:
        return {
            "reference_positive_prediction_rate": None,
            "recent_positive_prediction_rate": None,
            "absolute_delta": None,
            "severity": "none",
            "recent_count": int(recent_df.shape[0]),
            "reason": "missing_prediction_column",
        }

    output_reference = reference_stats.get("output_distribution", {})

    if "positive_prediction_rate" not in output_reference:
        return {
            "reference_positive_prediction_rate": None,
            "recent_positive_prediction_rate": None,
            "absolute_delta": None,
            "severity": "none",
            "recent_count": int(recent_df.shape[0]),
            "reason": "missing_reference_output_distribution",
        }

    reference_positive_rate = float(output_reference["positive_prediction_rate"])
    recent_positive_rate = float(pd.to_numeric(recent_df["prediction"]).mean())

    delta = abs(recent_positive_rate - reference_positive_rate)
    severity = severity_from_output_delta(delta)

    return {
        "reference_positive_prediction_rate": reference_positive_rate,
        "recent_positive_prediction_rate": recent_positive_rate,
        "absolute_delta": delta,
        "severity": severity,
        "recent_count": int(recent_df.shape[0]),
    }


def write_drift_report_artifacts(report: dict[str, Any]) -> None:
    """
    Write drift report in two forms:

    1. latest_drift_report.json
       Always overwritten. Useful for dashboard/current state.

    2. drift_reports/<report_id>.json
       Archived report. Useful for demo, debugging, and audit trail.

    Also appends one compact row to drift_reports/index.jsonl.
    """
    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_id = str(report["report_id"])
    archive_filename = f"{report_id}.json"
    archive_path = DRIFT_REPORTS_DIR / archive_filename

    report_with_path = {
        **report,
        "archive_filename": archive_filename,
        "archive_path": relative_path(archive_path),
    }

    DRIFT_REPORT_PATH.write_text(
        json.dumps(report_with_path, indent=2),
        encoding="utf-8",
    )

    archive_path.write_text(
        json.dumps(report_with_path, indent=2),
        encoding="utf-8",
    )

    index_record = {
        "report_id": report_with_path["report_id"],
        "created_at_local": report_with_path["created_at_local"],
        "created_at_utc": report_with_path["created_at_utc"],
        "timezone": report_with_path["timezone"],
        "severity": report_with_path["severity"],
        "status": report_with_path["status"],
        "recent_count": report_with_path["recent_count"],
        "window_size": report_with_path["window_size"],
        "archive_filename": archive_filename,
        "archive_path": report_with_path["archive_path"],
    }

    with DRIFT_REPORT_INDEX_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(index_record) + "\n")

    logger.info(
        "Drift report written: report_id=%s latest=%s archive=%s",
        report_id,
        DRIFT_REPORT_PATH,
        archive_path,
    )


def build_not_enough_data_report(
    *,
    window_size: int,
    recent_count: int,
    message: str,
) -> dict[str, Any]:
    report = make_report_base_metadata(
        window_size=window_size,
        recent_count=recent_count,
        severity="none",
        status="not_enough_data",
    )

    report["message"] = message

    return report


def collect_severities(
    numeric_drift: dict[str, Any],
    categorical_drift: dict[str, Any],
    output_drift: dict[str, Any],
) -> list[DriftSeverity]:
    severities: list[DriftSeverity] = []

    severities.extend(
        result["severity"]
        for result in numeric_drift.values()
        if result.get("severity") is not None
    )

    severities.extend(
        result["severity"]
        for result in categorical_drift.values()
        if result.get("severity") is not None
    )

    if output_drift.get("severity") is not None:
        severities.append(output_drift["severity"])

    return severities


def build_drift_report(window_size: int = 200) -> dict[str, Any]:
    """
    Build a drift report over the latest prediction records.

    The report is always written to:
    - artifacts/reports/latest_drift_report.json
    - artifacts/reports/drift_reports/<report_id>.json
    - artifacts/reports/drift_reports/index.jsonl
    """
    reference_stats = load_reference_stats()
    recent_df = load_recent_predictions(window_size=window_size)

    if recent_df.empty:
        report = build_not_enough_data_report(
            window_size=window_size,
            recent_count=0,
            message="No prediction records found yet.",
        )
        write_drift_report_artifacts(report)
        return report

    if recent_df.shape[0] < MIN_ROWS_FOR_DRIFT:
        report = build_not_enough_data_report(
            window_size=window_size,
            recent_count=int(recent_df.shape[0]),
            message=f"Need at least {MIN_ROWS_FOR_DRIFT} predictions for drift check.",
        )
        write_drift_report_artifacts(report)
        return report

    numeric_drift = compute_numeric_drift(recent_df, reference_stats)
    categorical_drift = compute_categorical_drift(recent_df, reference_stats)
    output_drift = compute_output_drift(recent_df, reference_stats)

    overall_severity = combine_severities(
        collect_severities(
            numeric_drift=numeric_drift,
            categorical_drift=categorical_drift,
            output_drift=output_drift,
        )
    )

    report = make_report_base_metadata(
        window_size=window_size,
        recent_count=int(recent_df.shape[0]),
        severity=overall_severity,
        status="ok",
    )

    report.update(
        {
            "numeric_drift": numeric_drift,
            "categorical_drift": categorical_drift,
            "output_drift": output_drift,
        }
    )

    write_drift_report_artifacts(report)

    logger.info(
        "Drift report built: report_id=%s severity=%s recent_count=%s latest_path=%s",
        report["report_id"],
        overall_severity,
        recent_df.shape[0],
        DRIFT_REPORT_PATH,
    )

    return report