"""
seg_trainer.py

End-to-end training loop for 3D BraTS segmentation using MONAI + PyTorch.

Run from repo root:

    python -m src.training.seg_trainer

This will:
- Load train/val splits from data/splits/
- Build 3D U-Net
- Train for a few epochs
- Compute validation Dice
- Save best checkpoint to runs/segmentation/
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import os

import torch
from torch.optim import Adam

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference

from src.datasets.brats_dataset import BratsDataConfig, get_brats_dataloaders
from src.models.unet_3d import UNet3DConfig, build_unet_3d


# -------------------------
# Configs
# -------------------------

@dataclass
class SegmentationTrainConfig:
    """
    Hyperparameters and paths for segmentation training.
    """
    max_epochs: int = 2          # start small for CPU sanity check
    learning_rate: float = 1e-4
    val_interval: int = 1        # run validation every N epochs
    amp: bool = False            # no AMP on CPU

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir: str = "runs/segmentation"
    ckpt_name: str = "unet3d_brats_best.pth"


# -------------------------
# Utilities
# -------------------------

def remap_labels_to_4_classes(label: torch.Tensor) -> torch.Tensor:
    """
    BraTS labels: {0, 1, 2, 4}
    We want 4 consecutive classes: {0, 1, 2, 3}

    This function maps:
        4 -> 3

    Expects label shape (B, 1, H, W, D) or (B, H, W, D).
    Returns same shape as input.
    """
    label = label.clone()
    label[label == 4] = 3
    return label


def compute_batch_dice(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int = 4,
    ignore_background: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Simple multi-class Dice metric computed on a batch.

    preds, targets: (B, H, W, D) with integer class indices in [0, num_classes-1]
    Returns mean Dice across classes (and batch), optionally ignoring class 0.
    """
    assert preds.shape == targets.shape, f"preds {preds.shape}, targets {targets.shape} mismatch"
    b = preds.shape[0]
    class_range = range(1 if ignore_background else 0, num_classes)
    dices: List[torch.Tensor] = []

    for c in class_range:
        pred_c = (preds == c)
        targ_c = (targets == c)

        intersection = (pred_c & targ_c).sum(dim=(1, 2, 3))  # per-sample
        pred_sum = pred_c.sum(dim=(1, 2, 3))
        targ_sum = targ_c.sum(dim=(1, 2, 3))

        dice_c = (2.0 * intersection + eps) / (pred_sum + targ_sum + eps)
        dices.append(dice_c)

    if not dices:
        return torch.tensor(0.0, device=preds.device)

    dices_stack = torch.stack(dices, dim=0)  # (C, B)
    return dices_stack.mean()  # scalar


# -------------------------
# Training
# -------------------------

def train_segmentation():
    # Paths & configs
    repo_root = Path(__file__).resolve().parents[2]
    seg_cfg = SegmentationTrainConfig()
    out_dir = repo_root / seg_cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)

    data_cfg = BratsDataConfig(
        patch_size=(96, 96, 96),
        batch_size=1,      # effective: 2 patches per batch due to RandCropByPosNegLabel
        num_workers=0,     # keep 0 on Windows
    )
    model_cfg = UNet3DConfig()

    device = torch.device(seg_cfg.device)
    print(f"[INFO] Using device: {device}")

    # Dataloaders
    train_loader, val_loader = get_brats_dataloaders(data_cfg)

    # Model
    model = build_unet_3d(model_cfg).to(device)

    # Loss: Dice + CrossEntropy combined, multi-class
    loss_fn = DiceCELoss(
        include_background=True,
        to_onehot_y=True,   # we pass integer labels with channel dim 1
        softmax=True,
    )

    optimizer = Adam(model.parameters(), lr=seg_cfg.learning_rate)

    best_val_dice = 0.0
    ckpt_path = out_dir / seg_cfg.ckpt_name

    # -------------------------
    # Epoch loop
    # -------------------------

    for epoch in range(1, seg_cfg.max_epochs + 1):
        print(f"\n===== Epoch {epoch}/{seg_cfg.max_epochs} =====")

        # ---- Train ----
        model.train()
        epoch_loss = 0.0
        step = 0

        for batch in train_loader:
            images = batch["image"].to(device)   # (B, 4, H, W, D)
            labels = batch["label"].to(device)   # (B, 1, H, W, D)

            # Remap labels {0,1,2,4} -> {0,1,2,3}
            labels = remap_labels_to_4_classes(labels).long()

            optimizer.zero_grad()

            logits = model(images)               # (B, 4, H, W, D)

            # Sanity check: spatial dims match (B, C, H, W, D) vs (B, 1, H, W, D)
            if logits.shape[2:] != labels.shape[2:]:
                raise RuntimeError(
                    f"Shape mismatch before loss: logits={logits.shape}, labels={labels.shape}"
                )

            # DiceCELoss will one-hot encode labels from shape (B,1,H,W,D)
            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            step += 1

            if step % 10 == 0:
                print(f"[Train] Step {step}, Loss = {loss.item():.4f}")

        epoch_loss /= max(step, 1)
        print(f"[Train] Epoch {epoch} average loss: {epoch_loss:.4f}")

        # ---- Validate ----
        if epoch % seg_cfg.val_interval == 0:
            model.eval()
            val_dices: List[torch.Tensor] = []

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)   # (B, 4, H, W, D)
                    labels = batch["label"].to(device)   # (B, 1, H, W, D)
                    labels = remap_labels_to_4_classes(labels).long()

                    # Sliding window inference over full (cropped) volume
                    logits = sliding_window_inference(
                        images,
                        roi_size=data_cfg.patch_size,
                        sw_batch_size=1,
                        predictor=model,
                    )

                    probs = torch.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)      # (B, H, W, D)
                    labels_idx = labels.squeeze(1)          # (B, H, W, D)

                    if preds.shape != labels_idx.shape:
                        raise RuntimeError(
                            f"Val shape mismatch: preds={preds.shape}, labels={labels_idx.shape}"
                        )

                    dice_val = compute_batch_dice(
                        preds,
                        labels_idx,
                        num_classes=4,
                        ignore_background=True,
                    )
                    val_dices.append(dice_val)

            if val_dices:
                mean_dice = torch.stack(val_dices).mean().item()
            else:
                mean_dice = 0.0

            print(f"[Val] Epoch {epoch} mean Dice (tumor classes only): {mean_dice:.4f}")

            # Save best model
            if mean_dice > best_val_dice:
                best_val_dice = mean_dice
                torch.save(model.state_dict(), ckpt_path)
                print(f"[SAVE] New best model saved to: {ckpt_path} (Dice={best_val_dice:.4f})")

    print(f"\nTraining complete. Best validation Dice: {best_val_dice:.4f}")
    print(f"Best model checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train_segmentation()
