"""
train_outcome_model.py

Train a simple outcome prediction model on BraTS radiomics + clinical features.

Input:
    data/processed/brats_outcome_dataset.csv

Target:
    high_risk_1yr (1 if Survival_days < 365, else 0)

IMPORTANT:
    We DO NOT use Survival_days as a feature, only to define the label.
    Features = radiomics + Age + Extent_of_Resection.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.ensemble import RandomForestClassifier

import joblib


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_ROOT = REPO_ROOT / "data" / "processed"
RUNS_ROOT = REPO_ROOT / "runs" / "outcomes"
RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    outcome_csv = PROCESSED_ROOT / "brats_outcome_dataset.csv"
    if not outcome_csv.exists():
        raise FileNotFoundError(f"Outcome dataset not found: {outcome_csv}")

    print(f"[INFO] Loading outcome dataset from: {outcome_csv}")
    df = pd.read_csv(outcome_csv)

    # Target
    if "high_risk_1yr" not in df.columns:
        raise KeyError("Column 'high_risk_1yr' not found in outcome dataset.")
    y = df["high_risk_1yr"].astype(int)

    # Radiomics features:
    radiomics_cols = [
        "voxel_count_class_1",
        "voxel_count_class_2",
        "voxel_count_class_3",
        "voxel_count_whole_tumor",
        "volume_class_1_ml",
        "volume_class_2_ml",
        "volume_class_3_ml",
        "volume_whole_tumor_ml",
        "ratio_core_to_whole",
        "ratio_edema_to_whole",
    ]

    # Clinical features:
    clinical_numeric = ["Age"]              # <-- Survival_days is intentionally EXCLUDED
    clinical_categorical = ["Extent_of_Resection"]

    # Sanity check columns
    for col in radiomics_cols + clinical_numeric + clinical_categorical:
        if col not in df.columns:
            raise KeyError(f"Expected column '{col}' not found in dataset.")

    X = df[radiomics_cols + clinical_numeric + clinical_categorical]

    # Train/val split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("[INFO] Train size:", X_train.shape, "Test size:", X_test.shape)
    print("[INFO] Class distribution in train:")
    print(y_train.value_counts())

    numeric_features = radiomics_cols + clinical_numeric
    categorical_features = clinical_categorical

    # Preprocessing: numerical passthrough, categorical one-hot
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("clf", clf),
        ]
    )

    print("[INFO] Training RandomForestClassifier for 1-year risk (no label leakage)...")
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    if len(np.unique(y_test)) == 2:
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
    else:
        auc = float("nan")

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    print("\n=== Evaluation on held-out test set (no Survival_days as feature) ===")
    print(f"Accuracy: {acc:.3f}")
    if not np.isnan(auc):
        print(f"ROC AUC:  {auc:.3f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, digits=3))

    # Save model
    model_path = RUNS_ROOT / "rf_high_risk_1yr.joblib"
    joblib.dump(model, model_path)
    print(f"[SAVE] Outcome model -> {model_path}")

    # Save feature list
    feature_list_path = RUNS_ROOT / "rf_high_risk_1yr_features.txt"
    with open(feature_list_path, "w") as f:
        for col in numeric_features + categorical_features:
            f.write(col + "\n")
    print(f"[SAVE] Feature list -> {feature_list_path}")


if __name__ == "__main__":
    main()
