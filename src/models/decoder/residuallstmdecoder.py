import torch
import torch.nn as nn
from torch.nn import functional as F

class ResidualLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int, 
        hidden_size: int,
        dropout: float,
        bidirectional: bool
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size,
            num_layers=1,
            dropout=dropout,
            bidirectional=bidirectional,
            batch_first=True
        )
        d = 2 if bidirectional else 1
        self.linear1 = nn.Linear(hidden_size*d, hidden_size*d*2)
        self.linear2 = nn.Linear(hidden_size*d*2, hidden_size)

    def forward(self, x):
        res = x
        x, _ = self.lstm(x)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        x = res + x
        return x
        
class ResidualLSTMDecoder(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        n_classes: int
    ):
        super().__init__()
        self.res_lstm = nn.ModuleList([
            ResidualLSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                dropout=dropout,
                bidirectional=bidirectional
            ) for i in range(num_layers)
        ])
        self.linear = nn.Linear(hidden_size, n_classes)
         
    def forward(self, x):
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)
        for layer in self.res_lstm:
            x = layer(x)
        x = self.linear(x)
        return x