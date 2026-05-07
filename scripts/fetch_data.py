# scripts/fetch_data.py

import shutil
from pathlib import Path

import kagglehub

from common.paths import RAW_BANK_DATA_PATH, ensure_project_dirs

KAGGLE_DATASET_HANDLE = "sahistapatel96/bankadditionalfullcsv"


def find_bank_csv(download_dir: Path) -> Path:
    csv_files = list(download_dir.rglob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in Kaggle download folder: {download_dir}")

    preferred_matches = [
        file for file in csv_files
        if "bank" in file.name.lower()
        and "additional" in file.name.lower()
        and "full" in file.name.lower()
    ]

    if preferred_matches:
        return preferred_matches[0]

    if len(csv_files) == 1:
        return csv_files[0]

    raise FileNotFoundError(
        "Multiple CSV files found, but could not identify bank-additional-full.csv. "
        f"Found: {[str(file) for file in csv_files]}"
    )


def main() -> None:
    ensure_project_dirs()

    print("--- DOWNLOADING DATASET FROM KAGGLE ---")
    download_path = Path(kagglehub.dataset_download(KAGGLE_DATASET_HANDLE))

    print(f"Kaggle cache location: {download_path}")
    print("Files in downloaded folder:")
    for file in download_path.rglob("*"):
        print(f" - {file}")

    source_csv = find_bank_csv(download_path)

    shutil.copy2(source_csv, RAW_BANK_DATA_PATH)

    print("--- DOWNLOAD COMPLETE ---")
    print(f"Source CSV: {source_csv}")
    print(f"Copied to project path: {RAW_BANK_DATA_PATH}")


if __name__ == "__main__":
    main()