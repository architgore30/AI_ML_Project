from pathlib import Path
import shutil

import kagglehub


BASE_DIR = Path(__file__).resolve().parent
TARGET_PATH = BASE_DIR / "dataset.csv"


def find_best_csv(download_root: Path) -> Path:
    csv_candidates = list(download_root.rglob("*.csv"))
    if not csv_candidates:
        raise FileNotFoundError(f"No CSV files found under {download_root}")
    return max(csv_candidates, key=lambda candidate: candidate.stat().st_size)


def main():
    download_path = Path(kagglehub.dataset_download("mczielinski/bitcoin-historical-data"))
    print("Downloaded dataset directory:", download_path)

    best_csv = find_best_csv(download_path)
    print("Using CSV:", best_csv)

    shutil.copy2(best_csv, TARGET_PATH)
    print("Copied dataset to:", TARGET_PATH)


if __name__ == "__main__":
    main()
