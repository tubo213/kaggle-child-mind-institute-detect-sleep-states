from typing import Optional

import torch
import torch.nn as nn


class SegTransformer(nn.Module):
    def __init__(self, n_channels, n_classes, nhead=8, num_layers=6):
        super().__init__()
        self.emb = nn.Linear(n_channels, 128)
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=nhead, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            transformer_encoder_layer, num_layers=num_layers
        )
        self.fc = nn.Linear(128, n_classes)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor, labels: Optional[torch.Tensor] = None):
        x = self.emb(x)
        x = self.transformer_encoder(x)
        x = self.fc(x)

        loss = None
        if labels is not None:
            loss = self.loss_fn(x, labels)

        return {"logits": x, "loss": loss}
