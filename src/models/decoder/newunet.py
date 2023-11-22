from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SEModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        (b, c, _) = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
def create_layer_norm(channel, length):
    return nn.LayerNorm([channel, length])

class ResidualBlock(nn.Module):
    def __init__(self, inputChannel, outputChannel, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(inputChannel, outputChannel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(outputChannel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(outputChannel, outputChannel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(outputChannel)
        self.downsample = downsample
        self.ca = SEModule(outputChannel)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        caOutput = self.ca(out)
        out = caOutput * out
        return out

class Down(nn.Module):
    """Downscaling with maxpool then residual block"""

    def __init__(self, in_channels, out_channels, scale_factor, norm=nn.BatchNorm1d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(scale_factor),
            ResidualBlock(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then residual block"""

    def __init__(self, in_channels, out_channels, bilinear=True, scale_factor=2, norm=nn.BatchNorm1d):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)
            self.conv = ResidualBlock(in_channels, out_channels)
        else:
            self.up = nn.ConvTranspose1d(
                in_channels, in_channels // 2, kernel_size=scale_factor, stride=scale_factor
            )
            self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class ResAttnUNet1DDecoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        duration: int,
        bilinear: bool = True,
        se: bool = False,
        res: bool = False,
        scale_factor: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.duration = duration
        self.bilinear = bilinear
        self.se = se
        self.res = res
        self.scale_factor = scale_factor

        factor = 2 if bilinear else 1
        self.inc = ResidualBlock(
            self.n_channels, 64, norm=partial(create_layer_norm, length=self.duration)
        )
        self.down1 = Down(
            64, 128, scale_factor, norm=partial(create_layer_norm, length=self.duration // 2)
        )
        self.down2 = Down(
            128, 256, scale_factor, norm=partial(create_layer_norm, length=self.duration // 4)
        )
        self.down3 = Down(
            256, 512, scale_factor, norm=partial(create_layer_norm, length=self.duration // 8)
        )
        self.down4 = Down(
            512,
            1024 // factor,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 16),
        )
        self.up1 = Up(
            1024,
            512 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 8),
        )
        self.up2 = Up(
            512,
            256 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 4),
        )
        self.up3 = Up(
            256,
            128 // factor,
            bilinear,
            scale_factor,
            norm=partial(create_layer_norm, length=self.duration // 2),
        )
        self.up4 = Up(
            128, 64, bilinear, scale_factor, norm=partial(create_layer_norm, length=self.duration)
        )

        self.cls = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, self.n_classes, kernel_size=1, padding=0),
            nn.Dropout(dropout),
        )
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> dict[str, Optional[torch.Tensor]]:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """

        # 1D ResAttnUNet
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)
        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        # classifier
        logits = self.cls(x)  # (batch_size,
