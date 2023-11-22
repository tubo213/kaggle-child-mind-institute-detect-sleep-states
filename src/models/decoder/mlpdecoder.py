import torch
import torch.nn as nn


class MLPDecoder(nn.Module):
    def __init__(self, n_channels: int, hidden_size: int, num_layers: int, n_classes: int):
        super(MLPDecoder, self).__init__()

        self.num_hidden_layers = num_layers - 1

        assert num_layers >= 3
        modules: list[nn.Module] = []
        for i in range(num_layers):
            if i == 0:
                modules.append(nn.Linear(n_channels, hidden_size))
                modules.append(nn.GELU())
            elif i == num_layers - 1:
                modules.append(nn.Linear(hidden_size, n_classes))
            else:
                modules.append(nn.Linear(hidden_size, hidden_size))
                modules.append(nn.GELU())
        self.mlp = nn.ModuleList(modules)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)

        Returns:
            torch.Tensor: (batch_size, n_timesteps, n_classes)
        """
        x = x.transpose(1, 2)
        for layer in self.mlp:
            x = layer(x)
        return x
