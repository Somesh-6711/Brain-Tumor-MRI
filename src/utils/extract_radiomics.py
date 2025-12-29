"""
extract_radiomics.py

Extract simple radiomic-style features from BraTS segmentation masks.

For now we use the *ground-truth* segmentation masks from brats_train.csv,
because they are clean and complete. Later we can add predicted masks too.

Outputs:
    data/processed/brats_radiomics_gt.csv

Columns (per case):
    - case_id
    - voxel_count_class_1, voxel_count_class_2, voxel_count_class_3
    - volume_class_1_mm3, volume_class_2_mm3, volume_class_3_mm3
    - volume_class_1_ml,  volume_class_2_ml,  volume_class_3_ml
    - voxel_count_whole_tumor
    - volume_whole_tumor_mm3
    - volume_whole_tumor_ml
    - ratio_core_to_whole
    - ratio_edema_to_whole
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import nibabel as nib

from src.training.seg_trainer import remap_labels_to_4_classes


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def compute_features_for_case(seg_path: str) -> Dict[str, float]:
    """
    Compute radiomic-style features from a single GT segmentation.

    seg_path: path to NIfTI with labels {0,1,2,4}. We remap 4->3 internally.
    """
    seg_img = nib.load(seg_path)
    seg = seg_img.get_fdata().astype(np.int16)  # (H, W, D)

    # Remap 4 -> 3 so classes are {0,1,2,3}
    seg_tensor = remap_labels_to_4_classes(
        # convert to torch then back to numpy to reuse function
        # (remap_labels_to_4_classes expects a tensor)
        # shape: (H, W, D) -> we'll add dummy batch+channel dims just for convenience
        # then squeeze back
        __import__("torch").from_numpy(seg)[None, None, ...]
    )
    seg = seg_tensor.squeeze().numpy()

    # Counts
    counts = {}
    for c in [1, 2, 3]:
        counts[c] = int((seg == c).sum())

    whole_tumor_voxels = int((seg > 0).sum())

    # Assume voxel spacing is 1mm^3 (true for BraTS preprocessed)
    # So volume in mm^3 = voxel_count; volume in mL = voxel_count / 1000
    feats: Dict[str, float] = {}

    for c in [1, 2, 3]:
        feats[f"voxel_count_class_{c}"] = counts[c]
        feats[f"volume_class_{c}_mm3"] = float(counts[c])
        feats[f"volume_class_{c}_ml"] = float(counts[c]) / 1000.0

    feats["voxel_count_whole_tumor"] = whole_tumor_voxels
    feats["volume_whole_tumor_mm3"] = float(whole_tumor_voxels)
    feats["volume_whole_tumor_ml"] = float(whole_tumor_voxels) / 1000.0

    core_voxels = counts[1] + counts[3]  # necrotic + enhancing
    edema_voxels = counts[2]

    if whole_tumor_voxels > 0:
        feats["ratio_core_to_whole"] = core_voxels / whole_tumor_voxels
        feats["ratio_edema_to_whole"] = edema_voxels / whole_tumor_voxels
    else:
        feats["ratio_core_to_whole"] = 0.0
        feats["ratio_edema_to_whole"] = 0.0

    return feats


def main():
    train_csv = SPLITS_DIR / "brats_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv}")

    df_train = pd.read_csv(train_csv)
    if df_train.empty:
        raise RuntimeError("Train CSV is empty.")

    records: List[Dict] = []

    print(f"[INFO] Computing features for {len(df_train)} training cases...")

    for _, row in df_train.iterrows():
        case_id = row["case_id"]
        seg_path = row["seg_path"]

        if not isinstance(seg_path, str) or seg_path.strip() == "":
            print(f"[WARN] Skipping {case_id}: empty seg_path")
            continue

        seg_file = Path(seg_path)
        if not seg_file.exists():
            print(f"[WARN] Skipping {case_id}: seg file not found at {seg_file}")
            continue

        try:
            feats = compute_features_for_case(str(seg_file))
            feats["case_id"] = case_id
            records.append(feats)
            print(f"[OK] {case_id}")
        except Exception as e:
            print(f"[ERROR] Failed on {case_id}: {e}")

    if not records:
        raise RuntimeError("No features computed; something went wrong.")

    df_feats = pd.DataFrame(records)
    out_csv = PROCESSED_DIR / "brats_radiomics_gt.csv"
    df_feats.to_csv(out_csv, index=False)
    print(f"[SAVE] Radiomics features (GT masks) -> {out_csv}")
    print("[INFO] Columns:", list(df_feats.columns))


if __name__ == "__main__":
    main()
