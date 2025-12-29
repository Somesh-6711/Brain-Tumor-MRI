"""
list_raw_csvs.py

Utility script to:
- Find all CSV files under data/raw/brats
- Print their paths
- Print the first few rows (head) of each

Run from repo root:

    python -m src.utils.list_raw_csvs
"""

from pathlib import Path
import pandas as pd


def main():
    repo_root = Path(__file__).resolve().parents[2]
    data_root = repo_root / "data" / "raw" / "brats"

    print(f"[INFO] Searching for CSVs under: {data_root}")
    csv_paths = list(data_root.rglob("*.csv"))

    if not csv_paths:
        print("[WARN] No CSV files found under data/raw/brats")
        return

    for path in csv_paths:
        print("\n" + "=" * 80)
        print(f"[CSV] {path}")
        try:
            df = pd.read_csv(path)
            print(f"[INFO] Shape: {df.shape}")
            print("[INFO] Columns:", list(df.columns))
            print("[HEAD]")
            print(df.head(3))
        except Exception as e:
            print(f"[ERROR] Failed to read {path}: {e}")


if __name__ == "__main__":
    main()
