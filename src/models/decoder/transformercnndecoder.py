import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, kernel_sizes: list[int], n_classes: int):
        super().__init__()

        self.conv = nn.Conv1d(
            input_size,
            hidden_size,
            kernel_size=kernel_sizes[0],
            stride=1,
            padding=(kernel_sizes[0] - 1) // 2,
        )
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for i in range(1, len(kernel_sizes)):
            self.convs.append(
                nn.Conv1d(
                    hidden_size, hidden_size, kernel_size=(kernel_sizes[i] - 1) // 2, stride=1
                )
            )
            self.bns.append(nn.BatchNorm1d(hidden_size))
        self.linear = nn.Linear(hidden_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)

        # residual connection
        for conv, bn in zip(self.convs, self.bns):
            x = F.relu(x)
            x = bn(conv(x)) + x

        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.linear(x)  # (batch_size, n_timesteps, n_classes)

        return x


class TransformerCNNDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        nhead: int,
        n_classes: int,
    ):
        super().__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, 3, padding=1)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=nhead, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.head = CNNHead(hidden_size, hidden_size, [3, 3, 3], n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        x = self.transformer_encoder(x)  # (batch_size, n_timesteps, n_channels)
        x = x.transpose(1, 2)  # (batch_size, n_channels, n_timesteps)
        x = self.head(x)

        return x
