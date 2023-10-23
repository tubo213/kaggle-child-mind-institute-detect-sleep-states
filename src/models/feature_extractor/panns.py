from typing import Callable, Optional

import torch
import torch.nn as nn

from src.models.feature_extractor.cnn import CNNSpectrogram
from src.models.feature_extractor.spectrogram import SpecFeatureExtractor


class PANNsFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_filters: int | tuple = 128,
        kernel_sizes: tuple = (32, 16, 4, 2),
        stride: int = 4,
        sigmoid: bool = False,
        output_size: Optional[int] = None,
        conv: Callable = nn.Conv1d,
        reinit: bool = True,
        win_length: Optional[int] = None,
    ):
        super().__init__()
        self.cnn_feature_extractor = CNNSpectrogram(
            in_channels=in_channels,
            base_filters=base_filters,
            kernel_sizes=kernel_sizes,
            stride=stride,
            sigmoid=sigmoid,
            output_size=output_size,
            conv=conv,
            reinit=reinit,
        )
        self.spec_feature_extractor = SpecFeatureExtractor(
            in_channels=in_channels,
            height=self.cnn_feature_extractor.height,
            hop_length=stride,
            win_length=win_length,
            out_size=output_size,
        )
        self.height = self.cnn_feature_extractor.height
        self.out_chans = self.cnn_feature_extractor.out_chans + in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, in_channels, time_steps)

        Returns:
            torch.Tensor : (batch_size, out_chans, height, time_steps)
        """

        cnn_img = self.cnn_feature_extractor(x)  # (batch_size, cnn_chans, height, time_steps)
        spec_img = self.spec_feature_extractor(x)  # (batch_size, in_channels, height, time_steps)

        img = torch.cat([cnn_img, spec_img], dim=1)  # (batch_size, out_chans, height, time_steps)

        return img
