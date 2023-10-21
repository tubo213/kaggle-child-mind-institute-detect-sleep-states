import numpy as np
import torch


class Mixup:
    def __init__(self, alpha: float = 0.4, apply_prob: float = 0.5):
        self.alpha = alpha
        self.apply_prob = apply_prob

    def __call__(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Mixup augmentation.

        Args:
            imgs (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (torch.Tensor): (batch_size, n_timesteps, n_classes)

        Returns:
            tuple[torch.Tensor]: mixed_imgs (batch_size, n_channels, n_timesteps)
                                 mixed_labels (batch_size, n_timesteps, n_classes)
        """
        if np.random.rand() > self.apply_prob:
            return imgs, labels
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)
        lam = np.random.beta(self.alpha, self.alpha)

        mixed_imgs: torch.Tensor = lam * imgs + (1 - lam) * imgs[idx]
        mixed_labels: torch.Tensor = lam * labels + (1 - lam) * labels[idx]

        return mixed_imgs, mixed_labels
