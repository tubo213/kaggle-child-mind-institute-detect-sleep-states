from typing import Callable, Optional

import torch
import torch.nn as nn


# ref: https://github.com/analokmaus/kaggle-g2net-public/tree/main/models1d_pytorch
class CNNSpectrogram(nn.Module):
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
    ):
        super().__init__()
        self.out_chans = len(kernel_sizes)
        self.out_size = output_size
        self.sigmoid = sigmoid
        if isinstance(base_filters, int):
            base_filters = tuple([base_filters])
        self.height = base_filters[-1]
        self.spec_conv = nn.ModuleList()
        for i in range(self.out_chans):
            tmp_block = [
                conv(
                    in_channels,
                    base_filters[0],
                    kernel_size=kernel_sizes[i],
                    stride=stride,
                    padding=(kernel_sizes[i] - 1) // 2,
                )
            ]
            if len(base_filters) > 1:
                for j in range(len(base_filters) - 1):
                    tmp_block = tmp_block + [
                        nn.BatchNorm1d(base_filters[j]),
                        nn.ReLU(inplace=True),
                        conv(
                            base_filters[j],
                            base_filters[j + 1],
                            kernel_size=kernel_sizes[i],
                            stride=stride,
                            padding=(kernel_sizes[i] - 1) // 2,
                        ),
                    ]
                self.spec_conv.append(nn.Sequential(*tmp_block))
            else:
                self.spec_conv.append(tmp_block[0])

        if self.out_size is not None:
            self.pool = nn.AdaptiveAvgPool2d((None, self.out_size))

        if reinit:
            for m in self.modules():
                if isinstance(m, nn.Conv1d):
                    nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm1d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (_type_): (batch_size, in_channels, time_steps)

        Returns:
            _type_: (batch_size, out_chans, height, time_steps)
        """
        # x: (batch_size, in_channels, time_steps)
        out: list[torch.Tensor] = []
        for i in range(self.out_chans):
            out.append(self.spec_conv[i](x))
        img = torch.stack(out, dim=1)  # (batch_size, out_chans, height, time_steps)
        if self.out_size is not None:
            img = self.pool(img)  # (batch_size, out_chans, height, out_size)
        if self.sigmoid:
            img = img.sigmoid()
        return img
