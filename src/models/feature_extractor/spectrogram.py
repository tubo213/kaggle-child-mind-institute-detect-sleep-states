from typing import Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T


class SpecNormalize(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        # batch, channel毎に正規化
        # x: (batch, channel, freq, time)
        min_ = x.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_ = x.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]

        return (x - min_) / (max_ - min_ + self.eps)


class SpecFeatureExtractor(nn.Module):
    def __init__(
        self,
        in_channels: int,
        height: int,
        hop_length: int,
        win_length: Optional[int] = None,
        out_size: Optional[int] = None,
    ):
        super().__init__()
        self.height = height
        self.out_chans = in_channels
        n_fft = height * 2 - 1
        self.feature_extractor = nn.Sequential(
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length, win_length=win_length),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )
        self.out_size = out_size

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        img = self.feature_extractor(x)
        if self.out_size is not None:
            img = self.pool(img)

        return img
