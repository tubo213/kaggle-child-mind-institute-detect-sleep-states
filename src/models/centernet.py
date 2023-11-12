from typing import Optional

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from torchvision.transforms.functional import resize

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.base import BaseModel


class CenterNetLoss(nn.Module):
    def __init__(
        self,
        keypoint_weight: float = 1.0,
        offset_weight: float = 1.0,
        bbox_size_weight: float = 1.0,
    ):
        super().__init__()
        self.keypoint_weight = keypoint_weight
        self.offset_weight = offset_weight
        self.bbox_size_weight = bbox_size_weight
        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.l1 = nn.L1Loss(reduction="sum")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Forward function.

        keypoint loss: BCEWithLogitsLoss
        offset loss: L1Loss
        bbox size loss: L1Loss

        Args:
            logits (torch.Tensor):
                (batch_size, n_time_steps, 6),
                6: (onset, wakeup, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size)
            labels (torch.Tensor):
                (batch_size, n_time_steps, 6),
                6: (onset, wakeup, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size)

        Returns:
            torch.Tensor: loss
        """
        labels = labels.flatten(0, 1)  # (batch_size * n_time_steps, 6)
        logits = logits.flatten(0, 1)  # (batch_size * n_time_steps, 6)

        # count number of objects
        nonzero_idx_onset = labels[:, 4].nonzero().view(-1)
        nonzero_idx_wakeup = labels[:, 5].nonzero().view(-1)
        num_obj = nonzero_idx_onset.numel() + nonzero_idx_wakeup.numel()
        if num_obj == 0:
            return self.bce(logits[:, :2], labels[:, :2]) * 0.0
        else:
            # keypoint loss
            keypoint_loss = self.bce(logits[:, :2], labels[:, :2]) / num_obj

            # other losses
            nonzero_idx = torch.cat([nonzero_idx_onset, nonzero_idx_wakeup], dim=0)
            logits = logits[nonzero_idx]  # (num_obj, 6)
            labels = labels[nonzero_idx]  # (num_obj, 6)
            # offset loss
            offset_loss = self.l1(logits[:, 2:4], labels[:, 2:4]) / num_obj
            # bbox size loss
            bbox_size_loss = self.l1(logits[:, 4:], labels[:, 4:]) / num_obj
            total_loss = (
                self.keypoint_weight * keypoint_loss
                + self.offset_weight * offset_loss
                + self.bbox_size_weight * bbox_size_loss
            )
            return total_loss


class CenterNet(BaseModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        keypoint_weight: float = 1.0,
        offset_weight: float = 1.0,
        bbox_size_weight: float = 1.0,
        mixup_alpha: float = 0.5,
        cutmix_alpha: float = 0.5,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.encoder = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.decoder = decoder
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = CenterNetLoss(
            keypoint_weight=keypoint_weight,
            offset_weight=offset_weight,
            bbox_size_weight=bbox_size_weight,
        )

    def _forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward function.

        Args:
            x (torch.Tensor): (batch_size, n_time_steps, n_channels)
            labels (Optional[torch.Tensor], optional): (batch_size, n_time_steps, 6)
            do_mixup (bool, optional): [description]. Defaults to False.
            do_cutmix (bool, optional): [description]. Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: logits or (logits, labels)
            logits: (batch_size, n_time_steps, 6)
            6: (onset, wakeup, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size)
        """
        x = self.feature_extractor(x)  # (batch_size, n_channels, height, n_timesteps)

        if do_mixup and labels is not None:
            x, labels = self.mixup(x, labels)
        if do_cutmix and labels is not None:
            x, labels = self.cutmix(x, labels)

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        logits = self.decoder(x)  # (batch_size, n_timesteps, n_classes)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        # logits: (bs, duration, 6)
        # 6: (onset, wakeup, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size)
        # return: (bs, org_duration, 2), 2: (onset, wakeup)
        bs, duration = logits.shape[:2]
        device = logits.device
        logits_idx = torch.linspace(0, org_duration - 1, duration, device=device)
        pred_onset = logits[:, :, 0].sigmoid().cpu().numpy()
        pred_wakeup = logits[:, :, 1].sigmoid().cpu().numpy()
        pred_onset_pos = (logits_idx + logits[:, :, 2]).long().cpu().numpy()  # pos = idx + offset
        pred_wakeup_pos = (logits_idx + logits[:, :, 3]).long().cpu().numpy()  # pos = idx + offset

        x = np.arange(org_duration)
        proba_per_step = np.zeros((bs, org_duration, 2))
        for i in range(bs):
            # pred_posが重複する場合は、確率が高い方を採用する
            pred_onset_pos_i, pred_onset_i = np_groupby_max(pred_onset_pos[i], pred_onset[i])
            pred_wakeup_pos_i, pred_wakeup_i = np_groupby_max(pred_wakeup_pos[i], pred_wakeup[i])

            # もとの長さに戻す. 予測値がない部分は線形補間
            f_onset = interp1d(
                pred_onset_pos_i, pred_onset_i, kind="linear", fill_value=0, bounds_error=False
            )
            f_wakeup = interp1d(
                pred_wakeup_pos_i, pred_wakeup_i, kind="linear", fill_value=0, bounds_error=False
            )

            proba_per_step_onset_i = f_onset(x)
            proba_per_step_wakeup_i = f_wakeup(x)
            proba_per_step[i, :, 0] = proba_per_step_onset_i
            proba_per_step[i, :, 1] = proba_per_step_wakeup_i

        return torch.from_numpy(proba_per_step).to(device)

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        # labels: (bs, duration, 6)
        # 6: (onset, wakeup, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size)
        return resize(labels, size=[org_duration, labels.shape[-1]], antialias=False)[:, :, [0, 1]]


def np_groupby_max(groups: np.ndarray, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.lexsort((data, groups))
    groups = groups[order]  # this is only needed if groups is unsorted
    data = data[order]
    index = np.empty(len(groups), "bool")
    index[-1] = True
    index[:-1] = groups[1:] != groups[:-1]
    return groups[index], data[index]
