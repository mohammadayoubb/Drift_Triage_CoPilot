from common.paths import TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH
from ml.clean import clean_bank_data
from ml.data_loader import read_bank_csv, save_splits
from ml.split import print_split_report, split_train_val_test
from ml.validate import validate_raw_bank_data


def main() -> None:
    print("--- LOADING RAW DATA ---")
    raw_df = read_bank_csv()

    validate_raw_bank_data(raw_df)

    print("\n--- CLEANING DATA ---")
    clean_df = clean_bank_data(raw_df)

    print(f"Clean shape: {clean_df.shape}")
    print(f"Clean columns: {list(clean_df.columns)}")

    if "duration" in clean_df.columns:
        raise AssertionError("duration is still present. It must be dropped.")

    print("\n--- SPLITTING DATA ---")
    train_df, val_df, test_df = split_train_val_test(clean_df)

    print_split_report(train_df, val_df, test_df)

    save_splits(train_df, val_df, test_df)

    print("\n--- SPLITS SAVED ---")
    print(f"Train: {TRAIN_DATA_PATH}")
    print(f"Val:   {VAL_DATA_PATH}")
    print(f"Test:  {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()