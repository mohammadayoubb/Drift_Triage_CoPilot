# src/common/paths.py

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REFERENCE_DATA_DIR = DATA_DIR / "reference"

RAW_BANK_DATA_PATH = RAW_DATA_DIR / "bank-additional-full.csv"

TRAIN_DATA_PATH = PROCESSED_DATA_DIR / "train.csv"
VAL_DATA_PATH = PROCESSED_DATA_DIR / "val.csv"
TEST_DATA_PATH = PROCESSED_DATA_DIR / "test.csv"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
MODEL_CARDS_DIR = ARTIFACTS_DIR / "model_cards"

MODEL_PATH = MODELS_DIR / "bank_marketing_pipeline.joblib"
METRICS_PATH = REPORTS_DIR / "metrics.json"
THRESHOLD_PATH = REPORTS_DIR / "threshold.json"
MODEL_CARD_PATH = MODEL_CARDS_DIR / "bank_marketing_model_card.md"
INPUT_SCHEMA_PATH = REPORTS_DIR / "input_schema.json"
REFERENCE_STATS_PATH = REFERENCE_DATA_DIR / "reference_stats.json"

PREDICTIONS_LOG_PATH = REPORTS_DIR / "predictions_log.csv"

DRIFT_REPORT_PATH = REPORTS_DIR / "latest_drift_report.json"

DRIFT_REPORTS_DIR = REPORTS_DIR / "drift_reports"
DRIFT_REPORT_INDEX_PATH = DRIFT_REPORTS_DIR / "index.jsonl"

LAST_DRIFT_SEVERITY_PATH = REPORTS_DIR / "last_drift_severity.txt"

AGENT_STATE_DIR = ARTIFACTS_DIR / "agent_state"
INVESTIGATIONS_PATH = AGENT_STATE_DIR / "investigations.json"

def ensure_project_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REFERENCE_DATA_DIR.mkdir(parents=True, exist_ok=True)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CARDS_DIR.mkdir(parents=True, exist_ok=True)

    DRIFT_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    AGENT_STATE_DIR.mkdir(parents=True, exist_ok=True)