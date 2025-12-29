"""
unet_3d.py

Defines a 3D U-Net model for BraTS segmentation using MONAI.
"""

from dataclasses import dataclass
from monai.networks.nets import UNet
from torch import nn


@dataclass
class UNet3DConfig:
    in_channels: int = 4        # 4 MRI modalities: flair, t1, t1ce, t2
    out_channels: int = 4       # background + 3 tumor labels (0,1,2,4 -> 4 classes)
    spatial_dims: int = 3
    base_channels: int = 16     # can increase later if you have more GPU
    num_res_units: int = 2


def build_unet_3d(cfg: UNet3DConfig) -> nn.Module:
    """
    Build a MONAI 3D U-Net based on the given config.
    """
    model = UNet(
        spatial_dims=cfg.spatial_dims,
        in_channels=cfg.in_channels,
        out_channels=cfg.out_channels,
        channels=(
            cfg.base_channels,
            cfg.base_channels * 2,
            cfg.base_channels * 4,
            cfg.base_channels * 8,
            cfg.base_channels * 16,
        ),
        strides=(2, 2, 2, 2),
        num_res_units=cfg.num_res_units,
    )
    return model
