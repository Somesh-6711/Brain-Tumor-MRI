"""
visualize_case.py

Quick sanity-check of one BraTS case:
- Loads paths from data/splits/brats_train.csv
- Loads NIfTI volumes for FLAIR, T1, T1ce, T2, and seg (mask)
- Plots a central axial slice for each modality + segmentation overlay
- Saves the figure to outputs/

"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = REPO_ROOT / "data" / "splits"
OUTPUTS_DIR = REPO_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)


def load_nifti(path: str) -> np.ndarray:
    """
    Load a NIfTI image and return the data as a numpy array (float32).
    """
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    return data


def visualize_single_case(row: pd.Series):
    """
    Given a row from brats_train.csv, load its volumes and create a simple visualization.
    """
    case_id = row["case_id"]
    print(f"[INFO] Visualizing case: {case_id}")

    flair = load_nifti(row["flair_path"])
    t1 = load_nifti(row["t1_path"])
    t1ce = load_nifti(row["t1ce_path"])
    t2 = load_nifti(row["t2_path"])
    seg = load_nifti(row["seg_path"])

    print(f"[SHAPE] flair: {flair.shape}, t1: {t1.shape}, t1ce: {t1ce.shape}, t2: {t2.shape}, seg: {seg.shape}")
    print(f"[RANGE] flair: {flair.min()} -> {flair.max()}")
    print(f"[RANGE] seg unique labels: {np.unique(seg)}")

    # We assume shape is (H, W, D) and take the middle slice along the last axis (axial view).
    z_mid = flair.shape[2] // 2

    flair_slice = flair[:, :, z_mid]
    t1_slice = t1[:, :, z_mid]
    t1ce_slice = t1ce[:, :, z_mid]
    t2_slice = t2[:, :, z_mid]
    seg_slice = seg[:, :, z_mid]

    # Simple intensity normalization for display: clip to 1st-99th percentile, then scale to [0,1]
    def normalize_for_display(slice_2d: np.ndarray) -> np.ndarray:
        low, high = np.percentile(slice_2d, [1, 99])
        slice_2d = np.clip(slice_2d, low, high)
        if high - low > 0:
            slice_2d = (slice_2d - low) / (high - low)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return slice_2d

    flair_disp = normalize_for_display(flair_slice)
    t1_disp = normalize_for_display(t1_slice)
    t1ce_disp = normalize_for_display(t1ce_slice)
    t2_disp = normalize_for_display(t2_slice)

    # Plot
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    ax_titles = ["FLAIR", "T1", "T1ce", "T2", "Segmentation"]

    images = [flair_disp, t1_disp, t1ce_disp, t2_disp, seg_slice]

    for ax, img, title in zip(axes, images, ax_titles):
        if title == "Segmentation":
            # Show segmentation with a discrete colormap
            im = ax.imshow(img, cmap="nipy_spectral")
        else:
            im = ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"BraTS Case: {case_id} (central axial slice)", fontsize=14)

    out_path = OUTPUTS_DIR / f"{case_id}_central_slice.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[SAVE] Visualization saved to: {out_path}")


def main():
    train_csv = SPLITS_DIR / "brats_train.csv"
    if not train_csv.exists():
        raise FileNotFoundError(f"Could not find train CSV at: {train_csv}")

    df_train = pd.read_csv(train_csv)
    if df_train.empty:
        raise RuntimeError("Train CSV is empty. Something went wrong with split generation.")

    # For now take the first case
    row = df_train.iloc[0]
    visualize_single_case(row)


if __name__ == "__main__":
    main()
