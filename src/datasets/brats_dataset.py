"""
brats_dataset.py

MONAI Dataset + transforms for BraTS2020 segmentation.

We read from:
    data/splits/brats_train.csv
    data/splits/brats_val.csv

Each row has:
    case_id, split_source, flair_path, t1_path, t1ce_path, t2_path, seg_path

We convert each row into a MONAI-style dict:
    {
        "case_id": ...,
        "image": [flair, t1, t1ce, t2],   # multi-modal input
        "label": seg_path                 # segmentation mask
    }
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    ToTensord,
)


@dataclass
class BratsDataConfig:
    """
    Config for BraTS dataloaders.

    patch_size: spatial size of 3D crops for training.
    batch_size: number of patches per iteration.
    num_workers: DataLoader workers (0 is safest on Windows).
    """
    train_csv: str = "data/splits/brats_train.csv"
    val_csv: str = "data/splits/brats_val.csv"
    patch_size: Tuple[int, int, int] = (96, 96, 96)
    batch_size: int = 1
    num_workers: int = 0  # start with 0 on Windows; you can increase later


def _load_split(csv_path: Path) -> List[Dict]:
    """
    Read the CSV and convert rows into a list of MONAI dicts.

    image key: list of 4 modality paths -> MONAI's LoadImaged will stack them as channels.
    label key: single segmentation path.
    """
    df = pd.read_csv(csv_path)
    records: List[Dict] = []

    for _, row in df.iterrows():
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
        records.append(record)

    return records


def get_brats_transforms_train(cfg: BratsDataConfig) -> Compose:
    """
    Training transforms:
    - Load multi-modal image + label
    - Ensure label has channel dimension
    - Reorient to RAS
    - Intensity normalize each channel separately
    - Random 3D patches around tumor & background

    Note: we do NOT CropForeground here to avoid cases where the image becomes
    smaller than the desired patch_size.
    """
    return Compose(
        [
            LoadImaged(keys=["image", "label"]),
            # image becomes (4, H, W, D) automatically from list of 4 paths
            EnsureChannelFirstd(keys=["label"]),  # seg: (1, H, W, D)
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(
                keys="image",
                nonzero=True,
                channel_wise=True,
            ),
            # Random crops: pick patches with and without tumor
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=cfg.patch_size,
                pos=1,
                neg=1,
                num_samples=2,  # 2 patches per case per iteration
                image_key="image",
                image_threshold=0,
            ),
            ToTensord(keys=["image", "label"]),
        ]
    )


def get_brats_transforms_val(cfg: BratsDataConfig) -> Compose:
    """
    Validation transforms:
    - Load full volume
    - Reorient to RAS
    - Intensity normalize

    We will use sliding-window inference, so no random cropping here.
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


def get_brats_dataloaders(cfg: BratsDataConfig):
    """
    Build MONAI Datasets and DataLoaders for train & val.
    """
    repo_root = Path(__file__).resolve().parents[2]

    train_csv_path = repo_root / cfg.train_csv
    val_csv_path = repo_root / cfg.val_csv

    if not train_csv_path.exists():
        raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
    if not val_csv_path.exists():
        raise FileNotFoundError(f"Val CSV not found: {val_csv_path}")

    train_records = _load_split(train_csv_path)
    val_records = _load_split(val_csv_path)

    train_transforms = get_brats_transforms_train(cfg)
    val_transforms = get_brats_transforms_val(cfg)

    train_ds = Dataset(data=train_records, transform=train_transforms)
    val_ds = Dataset(data=val_records, transform=val_transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.num_workers,
    )

    return train_loader, val_loader
