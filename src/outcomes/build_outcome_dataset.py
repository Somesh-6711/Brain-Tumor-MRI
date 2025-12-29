"""
build_outcome_dataset.py

Merge:
- Radiomics features from: data/processed/brats_radiomics_gt.csv
- Clinical survival info from:
    data/raw/brats/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv

and build a tabular outcome dataset for ML.

Output:
    data/processed/brats_outcome_dataset.csv

Label:
    - high_risk_1yr (binary):
        1 if Survival_days < 365
        0 otherwise
"""

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_ROOT = REPO_ROOT / "data" / "raw" / "brats"
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    # Paths
    radiomics_csv = PROCESSED_ROOT / "brats_radiomics_gt.csv"
    survival_csv = RAW_ROOT / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData" / "survival_info.csv"

    if not radiomics_csv.exists():
        raise FileNotFoundError(f"Radiomics CSV not found: {radiomics_csv}")
    if not survival_csv.exists():
        raise FileNotFoundError(f"Survival info CSV not found: {survival_csv}")

    print(f"[INFO] Loading radiomics from: {radiomics_csv}")
    df_rad = pd.read_csv(radiomics_csv)

    print(f"[INFO] Loading survival info from: {survival_csv}")
    df_surv = pd.read_csv(survival_csv)

    # Standardize column names
    # df_rad has 'case_id' like 'BraTS20_Training_001'
    # df_surv has 'Brats20ID' like 'BraTS20_Training_001'
    df_surv = df_surv.rename(columns={"Brats20ID": "case_id"})

    # Ensure Survival_days is numeric
    df_surv["Survival_days"] = pd.to_numeric(df_surv["Survival_days"], errors="coerce")

    # Inner join on case_id: only cases that have both radiomics & survival
    df_merged = pd.merge(df_rad, df_surv, on="case_id", how="inner")
    print(f"[INFO] Merged dataset shape: {df_merged.shape}")

    # Drop rows with missing Survival_days
    before = df_merged.shape[0]
    df_merged = df_merged.dropna(subset=["Survival_days"])
    after = df_merged.shape[0]
    print(f"[INFO] Dropped {before - after} rows with missing Survival_days. Remaining: {after}")

    # Create binary outcome: high risk if survival < 365 days
    df_merged["high_risk_1yr"] = (df_merged["Survival_days"] < 365).astype(int)

    # Optional: basic sanity check stats
    print("[INFO] Outcome class distribution (high_risk_1yr):")
    print(df_merged["high_risk_1yr"].value_counts())

    # Save outcome dataset
    out_csv = PROCESSED_ROOT / "brats_outcome_dataset.csv"
    df_merged.to_csv(out_csv, index=False)
    print(f"[SAVE] Outcome dataset -> {out_csv}")
    print("[INFO] Columns:", list(df_merged.columns))


if __name__ == "__main__":
    main()
