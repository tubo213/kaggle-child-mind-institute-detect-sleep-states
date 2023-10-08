import torch.nn as nn
from omegaconf import DictConfig

from src.models.decoder.unet1d import UNet1D
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.spec2Dcnn import Spec2DCNN


def get_feature_extractor(cfg: DictConfig, feature_dim: int, num_timesteps: int) -> nn.Module:
    if cfg.feature_extractor.name == "CNNSpectrogram":
        feature_extractor = CNNSpectrogram(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.feature_extractor.name}")

    return feature_extractor


def get_decoder(cfg: DictConfig, n_channels: int, n_classes: int, num_timesteps: int) -> nn.Module:
    if cfg.decoder.name == "UNet1D":
        decoder = UNet1D(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            bilinear=cfg.decoder.bilinear,
            se=cfg.decoder.se,
            res=cfg.decoder.res,
            scale_factor=cfg.decoder.scale_factor,
            dropout=cfg.decoder.dropout,
        )
    else:
        raise ValueError(f"Invalid decoder name: {cfg.decoder.name}")

    return decoder


def get_model(cfg: DictConfig, feature_dim: int, n_classes: int, num_timesteps: int) -> nn.Module:
    if cfg.model.name == "Spec2DCNN":
        feature_extractor = get_feature_extractor(cfg, feature_dim, num_timesteps)
        decoder = get_decoder(
            cfg, feature_extractor.height, n_classes, num_timesteps  # type: ignore
        )
        model = Spec2DCNN(
            feature_extractor=feature_extractor,
            decoder=decoder,
            encoder_name=cfg.model.encoder_name,
            in_channels=feature_extractor.out_chans,  # type: ignore
            encoder_weights=cfg.model.encoder_weights,
        )
    else:
        raise NotImplementedError

    return model
