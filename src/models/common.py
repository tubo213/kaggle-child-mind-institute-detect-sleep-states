import torch.nn as nn
from omegaconf import DictConfig

from src.models.decoder.lstmdecoder import LSTMDecoder
from src.models.decoder.mlpdecoder import MLPDecoder
from src.models.decoder.transformerdecoder import TransformerDecoder
from src.models.decoder.unet1ddecoder import UNet1DDecoder
from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.panns import PANNsFeatureExtractor
from src.models.spec2Dcnn import Spec2DCNN


def get_feature_extractor(cfg: DictConfig, feature_dim: int, num_timesteps: int) -> nn.Module:
    feature_extractor: CNNSpectrogram | PANNsFeatureExtractor
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
    elif cfg.feature_extractor.name == "PANNsFeatureExtractor":
        feature_extractor = PANNsFeatureExtractor(
            in_channels=feature_dim,
            base_filters=cfg.feature_extractor.base_filters,
            kernel_sizes=cfg.feature_extractor.kernel_sizes,
            stride=cfg.feature_extractor.stride,
            sigmoid=cfg.feature_extractor.sigmoid,
            output_size=num_timesteps,
            conv=nn.Conv1d,
            reinit=cfg.feature_extractor.reinit,
            win_length=cfg.feature_extractor.win_length,
        )
    else:
        raise ValueError(f"Invalid feature extractor name: {cfg.feature_extractor.name}")

    return feature_extractor


def get_decoder(cfg: DictConfig, n_channels: int, n_classes: int, num_timesteps: int) -> nn.Module:
    decoder: UNet1DDecoder | LSTMDecoder | TransformerDecoder | MLPDecoder
    if cfg.decoder.name == "UNet1DDecoder":
        decoder = UNet1DDecoder(
            n_channels=n_channels,
            n_classes=n_classes,
            duration=num_timesteps,
            bilinear=cfg.decoder.bilinear,
            se=cfg.decoder.se,
            res=cfg.decoder.res,
            scale_factor=cfg.decoder.scale_factor,
            dropout=cfg.decoder.dropout,
        )
    elif cfg.decoder.name == "LSTMDecoder":
        decoder = LSTMDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            bidirectional=cfg.decoder.bidirectional,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "TransformerDecoder":
        decoder = TransformerDecoder(
            input_size=n_channels,
            hidden_size=cfg.decoder.hidden_size,
            num_layers=cfg.decoder.num_layers,
            dropout=cfg.decoder.dropout,
            nhead=cfg.decoder.nhead,
            n_classes=n_classes,
        )
    elif cfg.decoder.name == "MLPDecoder":
        decoder = MLPDecoder(n_channels=n_channels, n_classes=n_classes)
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
            mixup_alpha=cfg.augmentation.mixup_alpha,
            cutmix_alpha=cfg.augmentation.cutmix_alpha,
        )
    else:
        raise NotImplementedError

    return model
    return model
