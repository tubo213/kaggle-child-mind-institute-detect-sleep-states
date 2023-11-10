from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class ModelOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    preds: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> ModelOutput:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
            do_mixup (bool, optional): Defaults to False.
            do_cutmix (bool, optional):  Defaults to False.

        Returns:
            ModelOutput: model output
        """
        if labels is not None:
            logits, labels = self._forward(x, labels, do_mixup, do_cutmix)
            loss = self.loss_fn(logits, labels)
            return ModelOutput(logits=logits, loss=loss, labels=labels)
        else:
            logits = self._forward(x, labels=None, do_mixup=False, do_cutmix=False)
            if isinstance(logits, torch.Tensor):
                return ModelOutput(logits=logits)
            else:
                raise ValueError("logits must be a torch.Tensor")

    @torch.no_grad()
    def predict(
        self,
        x: torch.Tensor,
        org_duration: int,
        labels: Optional[torch.Tensor] = None,
    ) -> ModelOutput:
        output = self.forward(x, labels, False, False)
        preds = self._logits_to_proba_per_step(output.logits, org_duration)
        output.preds = preds

        if labels is not None:
            labels = self._correct_labels(labels, org_duration)
            output.labels = labels

        return output

    @abstractmethod
    def _forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        do_mixup: bool = False,
        do_cutmix: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of the model.

        Args:
            x (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (Optional[torch.Tensor], optional): (batch_size, n_timesteps, n_classes)
            do_mixup (bool, optional): Defaults to False.
            do_cutmix (bool, optional):  Defaults to False.

        Returns:
            torch.Tensor | tuple[torch.Tensor, torch.Tensor]: logits or (logits, labels)
        """
        raise NotImplementedError

    @abstractmethod
    def _logits_to_proba_per_step(
        self,
        logits: torch.Tensor,
        org_duration: int,
    ) -> torch.Tensor:
        """Convert logits to probabilities per step.

        Args:
            logits (torch.Tensor): (batch_size, n_timesteps, n_classes)
            org_duration (int): original duration in seconds

        Returns:
            torch.Tensor: (batch_size, org_duration, n_classes)
        """
        raise NotImplementedError

    @abstractmethod
    def _correct_labels(
        self,
        labels: torch.Tensor,
        org_duration: int,
    ) -> torch.Tensor:
        """Correct labels to match the output of the model.

        Args:
            labels (torch.Tensor): (batch_size, org_duration, n_classes)
            org_duration (int): original duration in seconds

        Returns:
            torch.Tensor: (batch_size, org_duration, n_classes)
        """
        raise NotImplementedError
