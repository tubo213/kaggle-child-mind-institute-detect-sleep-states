from typing import Optional

import numpy as np
import polars as pl
import torch
import torch.optim as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from sklearn.metrics import average_precision_score
from transformers import get_cosine_schedule_with_warmup

from src.models.seg.model import get_model
from src.utils.metrics import event_detection_ap
from src.utils.post_process import post_process_for_seg


class SegModel(LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        val_event_df: pl.DataFrame,
        feature_dim: int,
        num_classes: int,
        duration: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.val_event_df = val_event_df
        self.model = get_model(cfg, feature_dim, num_classes, duration)
        self.validation_step_outputs: list = []

    def forward(
        self, x: torch.Tensor, labels: Optional[torch.Tensor]
    ) -> dict[str, Optional[torch.Tensor]]:
        return self.model(x, labels)

    def training_step(self, batch, batch_idx):
        return self.__share_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.__share_step(batch, "val")

    def __share_step(self, batch, mode: str) -> torch.Tensor:
        output = self.model(batch["feature"], batch["label"])
        loss = output["loss"]
        logits = output["logits"]

        if mode == "train":
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )
        elif mode == "val":
            self.validation_step_outputs.append(
                (batch["key"], batch["label"].detach().cpu(), logits.detach().cpu())
            )
            self.log(
                f"{mode}_loss",
                loss.detach().item(),
                on_step=False,
                on_epoch=True,
                logger=True,
                prog_bar=True,
            )

        return loss

    def on_validation_epoch_end(self):
        keys = []
        for x in self.validation_step_outputs:
            keys.extend(x[0])
        labels = torch.concat([x[1] for x in self.validation_step_outputs]).detach().cpu().numpy()
        preds = (
            torch.concat([x[2] for x in self.validation_step_outputs])
            .sigmoid()
            .detach()
            .cpu()
            .numpy()
        )

        np.save("keys.npy", np.array(keys))
        np.save("labels.npy", labels)
        np.save("preds.npy", preds)

        val_pred_df = post_process_for_seg(
            keys=keys,
            preds=preds,
            score_th=self.cfg.post_process.score_th,
            distance=self.cfg.post_process.distance,
        )
        val_pred_df.write_csv("val_pred_df.csv")
        score = event_detection_ap(self.val_event_df.to_pandas(), val_pred_df.to_pandas())
        # score = average_precision_score((labels == 1).reshape(-1, 2), preds.reshape(-1, 2))
        self.log("val_score", score, on_step=False, on_epoch=True, logger=True, prog_bar=True)

        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_training_steps=self.trainer.max_steps, **self.cfg.scheduler
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
