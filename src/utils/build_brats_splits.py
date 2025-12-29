import os
from pathlib import Path
from typing import List, Dict

import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[2]  # go up from src/utils/
DATA_ROOT = REPO_ROOT / "data" / "raw" / "brats"

TRAIN_ROOT = DATA_ROOT / "BraTS2020_TrainingData" / "MICCAI_BraTS2020_TrainingData"
VAL_ROOT = DATA_ROOT / "BraTS2020_ValidationData" / "MICCAI_BraTS2020_ValidationData"

SPLITS_DIR = REPO_ROOT / "data" / "splits"
SPLITS_DIR.mkdir(parents=True, exist_ok=True)


def find_case_dirs(root: Path, prefix: str) -> List[Path]:
    """
    Return all case directories matching e.g. BraTS20_Training_XXX
    """
    if not root.exists():
        raise FileNotFoundError(f"Expected root folder does not exist: {root}")

    case_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith(prefix)])
    return case_dirs


def collect_cases(root: Path, prefix: str, has_seg: bool) -> pd.DataFrame:
    """
    Collect metadata for each case:
    - case_id
    - flair_path, t1_path, t1ce_path, t2_path
    - seg_path (optional, empty if has_seg=False)
    """
    case_dirs = find_case_dirs(root, prefix)
    rows: List[Dict] = []

    for case_dir in case_dirs:
        case_id = case_dir.name

        def _path_for_suffix(suffix: str) -> str:
            filename = f"{case_id}_{suffix}.nii"
            path = case_dir / filename
            if not path.exists():
                # some datasets use .nii.gz instead of .nii
                gz_path = case_dir / f"{filename}.gz"
                if gz_path.exists():
                    return str(gz_path)
                return ""
            return str(path)

        flair = _path_for_suffix("flair")
        t1 = _path_for_suffix("t1")
        t1ce = _path_for_suffix("t1ce")
        t2 = _path_for_suffix("t2")
        seg = _path_for_suffix("seg") if has_seg else ""

        # Basic sanity check: must have all 4 modalities
        if not (flair and t1 and t1ce and t2):
            print(f"[WARN] Skipping {case_id}: missing one or more modalities.")
            continue

        if has_seg and not seg:
            print(f"[WARN] Skipping {case_id}: expected segmentation mask but not found.")
            continue

        rows.append(
            {
                "case_id": case_id,
                "split_source": "train" if has_seg else "val_challenge",
                "flair_path": flair,
                "t1_path": t1,
                "t1ce_path": t1ce,
                "t2_path": t2,
                "seg_path": seg,
            }
        )

    df = pd.DataFrame(rows)
    return df


def main():
    print(f"Repo root: {REPO_ROOT}")
    print(f"Training root: {TRAIN_ROOT}")
    print(f"Validation root: {VAL_ROOT}")

    # Collect supervised segmentation cases (with seg labels)
    print("\n[INFO] Collecting TRAIN cases (with segmentation labels)...")
    df_train_all = collect_cases(TRAIN_ROOT, prefix="BraTS20_Training", has_seg=True)
    print(f"[INFO] Found {len(df_train_all)} training cases.")

    # Collect validation cases (no seg labels, optional for inference)
    print("\n[INFO] Collecting VALIDATION cases (no segmentation labels)...")
    df_val_challenge = collect_cases(VAL_ROOT, prefix="BraTS20_Validation", has_seg=False)
    print(f"[INFO] Found {len(df_val_challenge)} validation (challenge) cases.")

    # Save combined metadata
    df_all = pd.concat([df_train_all, df_val_challenge], ignore_index=True)
    all_csv = SPLITS_DIR / "brats_all_cases.csv"
    df_all.to_csv(all_csv, index=False)
    print(f"[SAVE] All cases metadata -> {all_csv}")

    # Create supervised train/val split ONLY from labeled training cases
    # For now: 80% train, 20% val
    if len(df_train_all) == 0:
        raise RuntimeError("No training cases found with segmentation labels. Check your paths.")

    df_train, df_val = train_test_split(
        df_train_all,
        test_size=0.2,
        random_state=42,
        shuffle=True,
        stratify=None,  # we don't have labels yet here; can add later
    )

    train_csv = SPLITS_DIR / "brats_train.csv"
    val_csv = SPLITS_DIR / "brats_val.csv"

    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)

    print(f"[SAVE] Train split -> {train_csv} ({len(df_train)} cases)")
    print(f"[SAVE] Val split   -> {val_csv} ({len(df_val)} cases)")

    print("\nDone. You can inspect the CSVs in data/splits/.")


if __name__ == "__main__":
    main()
