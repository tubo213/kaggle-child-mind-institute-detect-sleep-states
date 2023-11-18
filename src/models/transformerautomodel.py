from typing import Optional

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize
from transformers import AutoConfig, AutoModel

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.base import BaseModel


class TransformerAutoModel(BaseModel):
    def __init__(
        self,
        model_name: str,
        n_channels: int,
        n_classes: int,
        hidden_size: int = 128,
        stride: int = 2,
        out_size: Optional[int] = None,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=n_channels,
            out_channels=hidden_size,
            kernel_size=stride,
            stride=stride,
            padding=(stride - 1) // 2,
        )
        self.pool = nn.AdaptiveAvgPool1d(out_size)
        # 推論時はinternet offなので、datasetから取得したconfigを使う
        self.config = AutoConfig.from_pretrained(model_name, hidden_size=hidden_size)
        self.backbone = AutoModel.from_config(self.config)
        self.head = nn.Linear(hidden_size, n_classes)

        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def _forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x)  # (batch_size, n_channels, n_timesteps)
        x = self.pool(x)  # (batch_size, n_channels, n_timesteps)
        x = x.transpose(1, 2)  # (batch_size, n_timesteps, n_channels)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.backbone(
            inputs_embeds=x
        ).last_hidden_state  # (batch_size, n_timesteps, hidden_size)
        logits = self.head(x)  # (batch_size, n_timesteps, n_classes)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)[:, :, [1, 2]]

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [1, 2]]
