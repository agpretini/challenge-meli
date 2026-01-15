from pathlib import Path

from src.data_process.read_utils import build_dataset, flatten_items


def main():
    # Paths
    ROOT_DIR = Path(__file__).resolve().parents[2]
    RAW_DATA_PATH = ROOT_DIR / "data" / "raw" / "MLA_100k_checked_v3.jsonlines"
    OUTPUT_DIR = ROOT_DIR / "data" / "processed"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    X_train, y_train, X_test, y_test = build_dataset(RAW_DATA_PATH)

    # Flatten
    X_train_flat = flatten_items(X_train)
    X_test_flat = flatten_items(X_test)

    # Add target to train
    X_train_flat["condition"] = y_train
    X_test_flat["condition"] = y_test
    X_train_flat["target_bin"] = (X_train_flat["condition"] == "used").astype(int)
    X_test_flat["target_bin"] = (X_test_flat["condition"] == "used").astype(int)

    # Export
    X_train_flat.to_parquet(OUTPUT_DIR / "train_base.parquet", index=False)
    X_test_flat.to_parquet(OUTPUT_DIR / "test_base.parquet", index=False)

    print("Dataset exportado correctamente")
    print(f"Train shape: {X_train_flat.shape}")
    print(f"Test shape: {X_test_flat.shape}")


if __name__ == "__main__":
    main()
