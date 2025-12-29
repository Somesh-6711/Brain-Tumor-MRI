"""
predict_single_case.py

Run inference with the trained 3D U-Net on a single BraTS case.

- Loads best checkpoint from runs/segmentation/unet3d_brats_best.pth
- Loads one case from data/splits/brats_val.csv (or by case_id)
- Runs sliding-window inference on the full 3D volume
- Saves:
    - NIfTI of predicted segmentation -> outputs/<case_id>_pred.nii.gz
    - PNG comparison (FLAIR, GT seg, Pred seg) -> outputs/<case_id>_pred_vis.png

Usage (from repo root):

    python -m src.inference.predict_single_case
    python -m src.inference.predict_single_case --case-id BraTS20_Validation_001

If --case-id is omitted, it will take the first row from brats_val.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
import torch
import matplotlib.pyplot as plt

from monai.inferers import sliding_window_inference
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, NormalizeIntensityd, ToTensord

from src.datasets.brats_dataset import BratsDataConfig
from src.models.unet_3d import UNet3DConfig, build_unet_3d
from src.training.seg_trainer import remap_labels_to_4_classes


def build_inference_transforms():
    """
    For inference, we mimic the validation transforms:
    - Load NIfTI
    - Ensure label has channel dim
    - Orientation to RAS
    - Intensity normalization
    - ToTensord so we can feed into PyTorch model
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def load_case_row(val_csv_path: Path, case_id: str | None = None) -> dict:
    """
    Load metadata row for a given case_id from brats_val.csv.
    If case_id is None, return the first row.
    """
    df = pd.read_csv(val_csv_path)
    if df.empty:
        raise RuntimeError(f"No rows in {val_csv_path}")

    if case_id is not None:
        row = df[df["case_id"] == case_id]
        if row.empty:
            raise ValueError(f"case_id {case_id} not found in {val_csv_path}")
        row = row.iloc[0]
    else:
        row = df.iloc[0]

    record = {
        "case_id": row["case_id"],
        "image": [
            row["flair_path"],
            row["t1_path"],
            row["t1ce_path"],
            row["t2_path"],
        ],
        "label": row["seg_path"],
    }
    return record


def save_nifti_prediction(pred_labels: np.ndarray, reference_image_path: str, out_path: Path):
    """
    Save predicted labels as a NIfTI file, using the affine from a reference image.

    NOTE:
    - We use the original FLAIR image's affine.
    - Orientation may not perfectly match the MONAI-processed orientation,
      but for visualization & portfolio purposes this is fine.
    """
    ref_img = nib.load(reference_image_path)
    affine = ref_img.affine

    nii = nib.Nifti1Image(pred_labels.astype(np.uint8), affine)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nii, str(out_path))
    print(f"[SAVE] Predicted segmentation NIfTI -> {out_path}")


def save_comparison_png(
    flair_vol: np.ndarray,
    gt_seg: np.ndarray,
    pred_seg: np.ndarray,
    case_id: str,
    out_path: Path,
):
    """
    Save a PNG with three panels:
    - FLAIR
    - Ground-truth segmentation
    - Predicted segmentation

    We take the central axial slice (z mid-plane).
    """
    z_mid = flair_vol.shape[2] // 2

    flair_slice = flair_vol[:, :, z_mid]
    gt_slice = gt_seg[:, :, z_mid]
    pred_slice = pred_seg[:, :, z_mid]

    # Normalize flair slice for nicer display
    def norm(slice_2d):
        low, high = np.percentile(slice_2d, [1, 99])
        slice_2d = np.clip(slice_2d, low, high)
        if high > low:
            slice_2d = (slice_2d - low) / (high - low)
        else:
            slice_2d = np.zeros_like(slice_2d)
        return slice_2d

    flair_disp = norm(flair_slice)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["FLAIR", "Ground Truth Seg", "Predicted Seg"]

    imgs = [flair_disp, gt_slice, pred_slice]

    for ax, img, title in zip(axes, imgs, titles):
        if "Seg" in title:
            im = ax.imshow(img, cmap="nipy_spectral")
        else:
            im = ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.suptitle(f"Case: {case_id} (central axial slice)", fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[SAVE] Visualization PNG -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case-id",
        type=str,
        default=None,
        help="BraTS case_id to run inference on (e.g., BraTS20_Validation_001). "
             "If omitted, the first row of brats_val.csv is used.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # Configs
    seg_ckpt = repo_root / "runs" / "segmentation" / "unet3d_brats_best.pth"
    if not seg_ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {seg_ckpt}")

    data_cfg = BratsDataConfig()
    model_cfg = UNet3DConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load model
    model = build_unet_3d(model_cfg).to(device)
    state_dict = torch.load(seg_ckpt, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"[INFO] Loaded model weights from {seg_ckpt}")

    # Load case metadata
    val_csv = repo_root / data_cfg.val_csv
    record = load_case_row(val_csv, case_id=args.case_id)
    case_id = record["case_id"]
    print(f"[INFO] Running inference for case: {case_id}")

    # Build transforms and apply to single sample
    transforms = build_inference_transforms()
    sample = transforms(record)

    # sample["image"]: Tensor (4, H, W, D)
    # sample["label"]: Tensor (1, H, W, D)
    image = sample["image"].unsqueeze(0).to(device)   # (1, 4, H, W, D)
    label = sample["label"].squeeze(0).cpu().numpy()  # (H, W, D), for visualization

    # We'll also load the original FLAIR for NIfTI affine
    flair_path = record["image"][0]

    # Model inference (sliding window)
    with torch.no_grad():
        logits = sliding_window_inference(
            image,
            roi_size=data_cfg.patch_size,
            sw_batch_size=1,
            predictor=model,
        )
        probs = torch.softmax(logits, dim=1)      # (1, 4, H, W, D)
        preds = torch.argmax(probs, dim=1)        # (1, H, W, D)

    preds_np = preds.squeeze(0).cpu().numpy().astype(np.uint8)  # (H, W, D)

    # For fairness, we remap GT labels 4->3 as well for comparison (though images still use original indices)
    label_remap = remap_labels_to_4_classes(torch.from_numpy(label)).numpy()

    # Save NIfTI prediction
    nifti_out = outputs_dir / f"{case_id}_pred.nii.gz"
    save_nifti_prediction(preds_np, flair_path, nifti_out)

    # For visualization we also want a FLAIR volume
    flair_nii = nib.load(flair_path)
    flair_vol = flair_nii.get_fdata().astype(np.float32)

    png_out = outputs_dir / f"{case_id}_pred_vis.png"
    save_comparison_png(flair_vol, label_remap, preds_np, case_id, png_out)


if __name__ == "__main__":
    main()
