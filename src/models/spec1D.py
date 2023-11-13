from typing import Optional

import torch
import torch.nn as nn
from torchvision.transforms.functional import resize

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.base import BaseModel


class Spec1D(BaseModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
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
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = x.transpose(1, 3)  # (batch_size, n_timesteps, height, n_channels)
        x = x.flatten(2, 3)  # (batch_size, n_timesteps, height * n_channels)
        x = x.transpose(1, 2)  # (batch_size, height * n_channels, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        preds = logits.sigmoid()
        return resize(preds, size=[org_duration, preds.shape[-1]], antialias=False)[:, :, [1, 2]]

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [1, 2]]
