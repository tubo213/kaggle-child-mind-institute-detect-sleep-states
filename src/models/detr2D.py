import math
from typing import Optional

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from src.augmentation.cutmix import Cutmix
from src.augmentation.mixup import Mixup
from src.models.base import BaseModel


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Calculate generalized box IOU for 1D boxes. Each box is defined by a start and an end point.

    Args:
        boxes1 (torch.Tensor): (batch_size * num_queries, [start, end])
        boxes2 (torch.Tensor): (total_num_obj, [start, end])

    Returns:
        torch.Tensor: (batch_size * num_queries, total_num_obj) Generalized IOU for each pair of boxes
    """
    # calculate the intersection of the boxes
    start_max = torch.max(boxes1[:, None, 0], boxes2[None, :, 0])
    end_min = torch.min(boxes1[:, None, 1], boxes2[None, :, 1])
    intersection = torch.clamp(end_min - start_max, min=0)

    # calculate the union of the boxes
    start_min = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    end_max = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    union = end_max - start_min

    # calculate the iou
    iou = intersection / union

    # calculate the smallest enclosing box
    start_min = torch.min(boxes1[:, None, 0], boxes2[None, :, 0])
    end_max = torch.max(boxes1[:, None, 1], boxes2[None, :, 1])
    enclosing = end_max - start_min

    # calculate the giou
    giou = iou - (enclosing - union) / enclosing

    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        ref: https://github.com/facebookresearch/detr/blob/3af9fa878e73b6894ce3596450a8d9b89d918ca9/models/matcher.py#L12
        Params:
            cost_class: cost weight for class loss
            cost_bbox: cost weight for box loss
            cost_giou: cost weight for giou loss
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        """

        Args:
            logits (torch.Tensor): [batch_size, max_det, (objectness, onset, wakeup)]
            labels (torch.Tensor): [batch_size, max_det, (objectness, onset, wakeup)]

        Returns:
            _type_: _description_
        """
        bs, num_queries, _ = logits.shape
        assert logits.shape[0] == labels.shape[0], "batch_size of logits and labels must be same"
        prob = logits[:, :, [0]].flatten(0, 1).sigmoid()  # (batch_size * max_det, 1)
        bbox = logits[:, :, 1:].flatten(0, 1)  # (batch_size * max_det, 2)

        tgt_class = labels[:, :, [0]].flatten(0, 1)  # (batch_size * max_det, 1)
        tgt_obj_idx = tgt_class.squeeze().nonzero(as_tuple=True)  # (num_obj)
        tgt_class = tgt_class[tgt_obj_idx]  # (num_obj, 1)
        tgt_bbox = labels[:, :, 1:].flatten(0, 1)[tgt_obj_idx]  # (num_obj, 2)

        # objectnessのコスト
        cost_class = -prob  # (batch_size * max_det, 1)

        # L1 loss
        cost_bbox = torch.cdist(bbox, tgt_bbox, p=1)  # (batch_size * max_det, num_obj)

        # giou loss
        # cost_giou = -generalized_box_iou(bbox, tgt_bbox)  # (batch_size, max_det, num_obj)

        # Total cost
        cost = (
            self.cost_class * cost_class + self.cost_bbox * cost_bbox # + self.cost_giou * cost_giou
        )  # (batch_size * max_det, num_obj)
        cost = cost.view(bs, num_queries, -1).cpu()  # (batch_size, max_det, num_obj)

        sizes = labels[:, :, 0].sum(dim=1).long().cpu().tolist()  # (batch_size)
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(cost.split(sizes, -1))]
        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


class DETRLoss(nn.Module):
    def __init__(self, matcher: HungarianMatcher):
        super().__init__()
        self.matcher = matcher
        self.bce = nn.BCEWithLogitsLoss()
        self.reg_loss = nn.GaussianNLLLoss(reduction="none")

    def forward(self, logits: torch.Tensor, labels: torch.Tensor):
        indices = self.matcher(logits[:, :, [0, 1, 3]], labels)

        loss_boxes = self.loss_boxes(logits, labels, indices)
        loss_labels = self.loss_labels(logits, labels, indices)

        return loss_boxes + loss_labels

    def loss_labels(self, logits, labels, indices):
        src_logits = logits[:, :, 0]  # (batch_size, max_det)

        idx = self._get_src_permutation_idx(indices)
        target_classes = torch.zeros_like(src_logits, device=src_logits.device)
        target_classes[idx] = 1

        loss_ce = self.bce(src_logits, target_classes)

        return loss_ce

    def loss_boxes(self, logits, labels, indices):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = logits[idx]
        target_boxes = []
        for t, (_, i) in zip(labels, indices):
            non_zero_idx = t[:, 0].nonzero(as_tuple=True)
            t = t[non_zero_idx]
            t = t[i]
            target_boxes.append(t)
        target_boxes_tensor = torch.cat(target_boxes, dim=0)

        # GaussianNLLLossの入力はmuとvar
        mu = src_boxes[:, [1, 3]]
        var = src_boxes[:, [2, 4]]
        loss_bbox = self.reg_loss(mu, target_boxes_tensor[:, 1:], var)

        # 損失をnum_objで割る
        loss_bbox = loss_bbox.sum() / (target_boxes_tensor[:, 0].sum())

        return loss_bbox

    def _get_src_permutation_idx(self, indices: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx


class DETRHead(nn.Module):
    def __init__(
        self,
        max_det: int,
        hidden_dim: int,
        nheads: int,
        num_encoder_layers: int,
        num_decoder_layers: int,
    ):
        super().__init__()
        self.max_det = max_det
        self.transformer = nn.Transformer(
            hidden_dim,
            nhead=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )
        self.linear_class = nn.Linear(hidden_dim, 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)  # [onset, onset_var, wakeup, wakeup_var]

        self.query = nn.Parameter(torch.rand(1, max_det, hidden_dim))  # [max_det, hidden_dim]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward

        Args:
            x (torch.Tensor): (batch_size, num_frames, hidden_dim)

        Returns:
            torch.Tensor: (batch_size, max_det, 5)
        """
        query = self.query.repeat(x.shape[0], 1, 1)  # (batch_size, max_det, hidden_dim)
        x = self.transformer(x, query)  # (batch_size, max_det, hidden_dim)

        logits: torch.Tensor = self.linear_class(x)
        bbox: torch.Tensor = self.linear_bbox(x)
        bbox[:, :, [0, 2]] = bbox[:, :, [0, 2]].sigmoid()  # 0 <= onset, wakeup <= 1

        # varを正の値にする
        bbox[:, :, [1, 3]] = bbox[:, :, [1, 3]].sigmoid()

        return torch.concat([logits, bbox], dim=-1)  # (batch_size, max_det, 5)


class DETR2DCNN(BaseModel):
    def __init__(
        self,
        feature_extractor: nn.Module,
        decoder: nn.Module,
        encoder_name: str,
        in_channels: int,
        encoder_weights: Optional[str] = None,
        max_det: int = 20,
        hidden_dim: int = 256,
        nheads: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
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
        self.head = DETRHead(
            max_det=max_det,
            hidden_dim=hidden_dim,
            nheads=nheads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
        )
        self.mixup = Mixup(mixup_alpha)
        self.cutmix = Cutmix(cutmix_alpha)
        self.loss_fn = DETRLoss(HungarianMatcher())

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

        x = self.encoder(x).squeeze(1)  # (batch_size, height, n_timesteps)
        x = self.decoder(x)  # (batch_size, n_timesteps, n_classes)
        logits = self.head(x)  # (batch_size, max_det, 5)

        if labels is not None:
            return logits, labels
        else:
            return logits

    def _logits_to_proba_per_step(self, logits: torch.Tensor, org_duration: int) -> torch.Tensor:
        device = logits.device
        x = torch.linspace(0, 1, steps=org_duration, device=device)  # (org_duration)
        objectness = logits[:, :, 0].sigmoid()  # (batch_size, max_det)
        onset_mu = logits[:, :, 1]  # (batch_size, max_det)
        onset_var = logits[:, :, 2]  # (batch_size, max_det)
        wakeup_mu = logits[:, :, 3]  # (batch_size, max_det)
        wakeup_var = logits[:, :, 4]  # (batch_size, max_det)
        gaussian_onset = gaussian(x, objectness, onset_mu, onset_var)  # (batch_size, org_duration)
        gaussian_wakeup = gaussian(
            x, objectness, wakeup_mu, wakeup_var
        )  # (batch_size, org_duration)

        return torch.stack(
            [gaussian_onset, gaussian_wakeup], dim=-1
        )  # (batch_size, org_duration, 2)

    def _correct_labels(self, labels: torch.Tensor, org_duration: int) -> torch.Tensor:
        bs, num_queries, _ = labels.shape
        device = labels.device
        torch.arange(org_duration, device=device).float()
        objectness = labels[:, :, 0].unsqueeze(-1)  # (batch_size, max_det, 1)
        onset = (
            (labels[:, :, 1] * org_duration).round().clip(0, org_duration)
        )  # (batch_size, max_det)
        wakeup = (
            (labels[:, :, 2] * org_duration).round().clip(0, org_duration)
        )  # (batch_size, max_det)

        onset_label = (
            F.one_hot(onset.long(), org_duration + 1).float() * objectness
        )  # (batch_size, max_det, org_duration)
        wakeup_label = (
            F.one_hot(wakeup.long(), org_duration + 1).float() * objectness
        )  # (batch_size, max_det, org_duration)
        onset_label = onset_label.max(dim=1)[0]  # (batch_size, org_duration)
        wakeup_label = wakeup_label.max(dim=1)[0]  # (batch_size, org_duration)

        # 0番目と最後の要素は0にする
        # これは0以上org_duration以下にクリップしているので、0とorg_durationを除外するため
        onset_label[:, 0] = 0
        onset_label[:, -1] = 0
        wakeup_label[:, 0] = 0
        wakeup_label[:, -1] = 0

        return torch.stack([onset_label, wakeup_label], dim=-1)  # (batch_size, org_duration, 2)


def gaussian(
    x: torch.Tensor, objectness: torch.Tensor, mu: torch.Tensor, var: torch.Tensor
) -> torch.Tensor:
    """Gaussian distribution

    Args:
        x (torch.Tensor): (org_duration)
        objectness (torch.Tensor): (batch_size, num_queries)
        mu (torch.Tensor): (batch_size, num_queries)
        var (torch.Tensor): (batch_size, num_queries)

    Returns:
        torch.Tensor: (batch_size, org_duration)
    """
    device = x.device
    x = x.view(1, -1, 1)  # (1, org_duration, 1)
    objectness = objectness.unsqueeze(1)  # (batch_size, 1, num_queries)
    mu = mu.unsqueeze(1)  # (batch_size, 1, num_queries)
    var = var.unsqueeze(1)  # (batch_size, 1, num_queries)
    numerator = torch.exp(-0.5 * ((x - mu) ** 2 / var))  # (batch_size, org_duration, num_queries)
    denominator = torch.sqrt(
        2 * var * torch.tensor(math.pi, device=device)
    )  # (batch_size, 1, num_queries)

    prob = objectness * numerator / denominator  # (batch_size, org_duration, num_queries)

    return prob.max(dim=-1)[0]  # (batch_size, org_duration)
