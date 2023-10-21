import numpy as np
import torch


def get_rand_1dbbox(n_timesteps: int, lam: float) -> tuple[int, int]:
    """Get random 1D bounding box.

    Args:
        n_timesteps (int): Number of timesteps.
        lam (float): Lambda value.

    Returns:
        tuple[int, int]: (start, end) of the bounding box.
    """
    cut_rat = np.sqrt(1.0 - lam)
    cut_len = int(n_timesteps * cut_rat)

    start = np.random.randint(0, n_timesteps - cut_len)
    end = start + cut_len

    return start, end


class Cutmix:
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha

    def __call__(
        self, imgs: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cutmix augmentation.

        Args:
            imgs (torch.Tensor): (batch_size, n_channels, n_timesteps)
            labels (torch.Tensor): (batch_size, n_timesteps, n_classes)

        Returns:
            tuple[torch.Tensor]: mixed_imgs (batch_size, n_channels, n_timesteps)
                                 mixed_labels (batch_size, n_timesteps, n_classes)
        """
        batch_size = imgs.size(0)
        idx = torch.randperm(batch_size)

        shuffled_imgs = imgs[idx]
        shuffled_labels = labels[idx]

        lam = np.random.beta(self.alpha, self.alpha)
        start, end = get_rand_1dbbox(imgs.size(2), lam)

        mixed_imgs = torch.concatenate(
            [imgs[:, :, :start], shuffled_imgs[:, :, start:end], imgs[:, :, end:]], dim=2
        )
        mixed_labels = torch.concatenate(
            [labels[:, :start, :], shuffled_labels[:, start:end, :], labels[:, end:, :]], dim=1
        )

        return mixed_imgs, mixed_labels


if __name__ == "__main__":
    imgs = torch.randn(2, 3, 100)
    labels = torch.randn(2, 100, 5)
    cutmix = Cutmix()

    mixed_imgs, mixed_labels = cutmix(imgs, labels)

    print(mixed_imgs.shape)
    print(mixed_labels.shape)
