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
    def __init__(self, n_fft: int, hop_length: int):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            T.Spectrogram(n_fft=n_fft, hop_length=hop_length),
            T.AmplitudeToDB(top_db=80),
            SpecNormalize(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.feature_extractor(x)
