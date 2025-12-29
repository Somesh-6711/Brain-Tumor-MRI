"""
make_publication_figure.py

Create a publication-style figure for a single BraTS case with:
1) FLAIR MRI (raw)
2) FLAIR + Ground-truth segmentation
3) FLAIR + Predicted segmentation

Output:
    outputs/<case_id>_before_after.png

Usage (from repo root):

    python -m src.utils.make_publication_figure --case-id BraTS20_Training_166

Assumptions:
- Splits are defined in data/splits/brats_train.csv and brats_val.csv
- Ground truth seg path is in seg_path column.
- Predicted NIfTI exists at: outputs/<case_id>_pred.nii.gz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_case_record(case_id: str):
    """
    Load a row for the given case_id from either brats_train.csv or brats_val.csv.
    """
    train_csv = SPLITS_DIR / "brats_train.csv"
    val_csv = SPLITS_DIR / "brats_val.csv"

    dfs = []
    if train_csv.exists():
        dfs.append(pd.read_csv(train_csv))
    if val_csv.exists():
        dfs.append(pd.read_csv(val_csv))

    if not dfs:
        raise FileNotFoundError("No split CSVs found in data/splits/")

    df_all = pd.concat(dfs, ignore_index=True)
    row = df_all[df_all["case_id"] == case_id]

    if row.empty:
        raise ValueError(f"case_id {case_id} not found in brats_train.csv or brats_val.csv")

    return row.iloc[0]


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    """
    Percentile-based normalization of an MRI slice (for nicer display).
    """
    low, high = np.percentile(slice_2d, [1, 99])
    slice_2d = np.clip(slice_2d, low, high)
    if high > low:
        slice_2d = (slice_2d - low) / (high - low)
    else:
        slice_2d = np.zeros_like(slice_2d)
    return slice_2d


def create_publication_figure(case_id: str):
    record = load_case_record(case_id)

    flair_path = record["flair_path"]
    seg_gt_path = record["seg_path"]
    pred_path = OUTPUTS_DIR / f"{case_id}_pred.nii.gz"

    if not Path(flair_path).exists():
        raise FileNotFoundError(f"FLAIR not found: {flair_path}")
    if not Path(seg_gt_path).exists():
        raise FileNotFoundError(f"GT segmentation not found: {seg_gt_path}")
    if not pred_path.exists():
        raise FileNotFoundError(
            f"Predicted segmentation not found: {pred_path}\n"
            "Run inference first with:\n"
            f'  python -m src.inference.predict_single_case --case-id {case_id}'
        )

    # Load images
    flair_nii = nib.load(flair_path)
    seg_gt_nii = nib.load(seg_gt_path)
    seg_pred_nii = nib.load(str(pred_path))

    flair = flair_nii.get_fdata().astype(np.float32)
    seg_gt = seg_gt_nii.get_fdata().astype(np.int16)
    seg_pred = seg_pred_nii.get_fdata().astype(np.int16)

    # Use central axial slice
    z = flair.shape[2] // 2
    flair_slice = normalize_slice(flair[:, :, z])
    gt_slice = seg_gt[:, :, z]
    pred_slice = seg_pred[:, :, z]

    # Masked arrays for overlay (ignore background 0)
    gt_mask = np.ma.masked_where(gt_slice == 0, gt_slice)
    pred_mask = np.ma.masked_where(pred_slice == 0, pred_slice)

    # Simple colormap for labels 1,2,3 (we'll rely on their numeric value)
    # You can tweak these colors if you want.
    cmap = ListedColormap(["black", "red", "yellow", "cyan"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = [
        "FLAIR (raw MRI)",
        "FLAIR + Ground Truth Segmentation",
        "FLAIR + Predicted Segmentation",
    ]

    # 1) FLAIR only
    ax = axes[0]
    ax.imshow(flair_slice, cmap="gray")
    ax.set_title(titles[0])
    ax.axis("off")

    # 2) FLAIR + GT
    ax = axes[1]
    ax.imshow(flair_slice, cmap="gray")
    gt_im = ax.imshow(gt_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    ax.set_title(titles[1])
    ax.axis("off")

    # 3) FLAIR + Pred
    ax = axes[2]
    ax.imshow(flair_slice, cmap="gray")
    pred_im = ax.imshow(pred_mask, cmap=cmap, alpha=0.6, vmin=0, vmax=3)
    ax.set_title(titles[2])
    ax.axis("off")

    plt.suptitle(f"Brain Tumor Segmentation – Case: {case_id}", fontsize=14)
    plt.tight_layout()

    out_path = OUTPUTS_DIR / f"{case_id}_before_after.png"
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

    print(f"[SAVE] Publication-style figure -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        type=str,
        default="BraTS20_Training_166",
        help="Case ID to visualize (must exist in brats_train.csv or brats_val.csv).",
    )
    args = parser.parse_args()
    create_publication_figure(args.case_id)


if __name__ == "__main__":
    main()
