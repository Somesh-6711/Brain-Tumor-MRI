"""
check_dataloader.py

Quick check to make sure our MONAI Dataset + DataLoader works.

Run from repo root:

    python -m src.utils.check_dataloader
"""

import torch

from src.datasets.brats_dataset import BratsDataConfig, get_brats_dataloaders


def main():
    cfg = BratsDataConfig(
        patch_size=(128, 128, 128),
        batch_size=1,
        num_workers=0,  # keep 0 on Windows initially
    )

    train_loader, val_loader = get_brats_dataloaders(cfg)

    print("[INFO] Iterating over one batch from train_loader...")
    batch = next(iter(train_loader))

    images = batch["image"]  # shape: (B, C, H, W, D)
    labels = batch["label"]  # shape: (B, 1, H, W, D)

    print(f"Image batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")

    # Check device & dtype
    print(f"Image dtype: {images.dtype}, label dtype: {labels.dtype}")
    print(f"Label unique values in this patch: {torch.unique(labels)}")


if __name__ == "__main__":
    main()
