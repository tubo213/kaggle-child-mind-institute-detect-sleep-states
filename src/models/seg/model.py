import torch.nn as nn
from omegaconf import DictConfig

from src.models.seg.UNet1D import UNet1D


def get_model(cfg: DictConfig, feature_dim: int, num_classes: int, duration: int) -> nn.Module:
    if cfg.model.name == "UNet1D":
        return (
            UNet1D(
                n_channels=feature_dim,
                n_classes=num_classes,
                duration=duration,
                **cfg.model.params,
            )
        )
    else:
        raise ValueError(f"Invalid model name: {cfg.model.name}")
