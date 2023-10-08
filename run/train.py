import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from src.datamodule.seg import SegDataModule
from src.modelmodule.seg import SegModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s:%(name)s - %(message)s"
)
LOGGER = logging.getLogger(Path(__file__).name)


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):  # type: ignore
    seed_everything(cfg.seed)

    # init lightning model
    datamodule = SegDataModule(cfg)
    LOGGER.info("Set Up DataModule")
    model = SegModel(
        cfg, datamodule.valid_event_df, len(cfg.features), len(cfg.labels), cfg.duration
    )

    # set callbacks
    checkpoint_cb = ModelCheckpoint(
        verbose=True,
        monitor=cfg.monitor,
        mode=cfg.monitor_mode,
        save_top_k=1,
        save_last=False,
    )
    lr_monitor = LearningRateMonitor("epoch")
    progress_bar = RichProgressBar()
    model_summary = RichModelSummary(max_depth=2)

    # init experiment logger
    pl_logger = WandbLogger(
        name=cfg.exp_name,
        project="child-mind-institute-detect-sleep-states",
    )

    trainer = Trainer(
        # env
        default_root_dir=Path.cwd(),
        # num_nodes=cfg.training.num_gpus,
        accelerator=cfg.accelerator,
        precision=16 if cfg.use_amp else 32,
        # training
        fast_dev_run=cfg.debug,  # run only 1 train batch and 1 val batch
        max_epochs=cfg.epoch,
        max_steps=cfg.epoch * len(datamodule.train_dataloader()),
        gradient_clip_val=cfg.gradient_clip_val,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        callbacks=[checkpoint_cb, lr_monitor, progress_bar, model_summary],
        logger=pl_logger,
        # resume_from_checkpoint=resume_from,
        num_sanity_val_steps=0,
        log_every_n_steps=int(len(datamodule.train_dataloader()) * 0.1),
        sync_batchnorm=True,
        check_val_every_n_epoch=cfg.check_val_every_n_epoch,
    )

    trainer.fit(model, datamodule=datamodule)

    # load best weights
    model = model.load_from_checkpoint(
        checkpoint_cb.best_model_path,
        cfg=cfg,
        val_event_df=datamodule.valid_event_df,
        feature_dim=len(cfg.features),
        num_classes=len(cfg.labels),
        duration=cfg.duration,
    )
    weights_path = str("model_weights.pth")  # type: ignore
    LOGGER.info(f"Extracting and saving best weights: {weights_path}")
    torch.save(model.model.state_dict(), weights_path)

    return


if __name__ == "__main__":
    main()
