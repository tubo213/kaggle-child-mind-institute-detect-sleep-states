import numpy as np
import polars as pl

from src.conf import InferenceConfig, TrainConfig
from src.dataset.centernet import (
    CenterNetTestDataset,
    CenterNetTrainDataset,
    CenterNetValidDataset,
)
from src.dataset.detr import DETRTestDataset, DETRTrainDataset, DETRValidDataset
from src.dataset.seg import SegTestDataset, SegTrainDataset, SegValidDataset

TRAIN_DATASET_TYPE = SegTrainDataset | DETRTrainDataset | CenterNetTrainDataset
VALID_DATASET_TYPE = SegValidDataset | DETRValidDataset | CenterNetValidDataset
TEST_DATASET_TYPE = SegTestDataset | DETRTestDataset | CenterNetTestDataset


def get_train_ds(
    cfg: TrainConfig,
    event_df: pl.DataFrame,
    features: dict[str, np.ndarray],
) -> TRAIN_DATASET_TYPE:
    if cfg.dataset.name == "seg":
        return SegTrainDataset(cfg=cfg, features=features, event_df=event_df)
    elif cfg.dataset.name == "detr":
        return DETRTrainDataset(cfg=cfg, features=features, event_df=event_df)
    elif cfg.dataset.name == "centernet":
        return CenterNetTrainDataset(cfg=cfg, features=features, event_df=event_df)
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")


def get_valid_ds(
    cfg: TrainConfig,
    event_df: pl.DataFrame,
    chunk_features: dict[str, np.ndarray],
) -> VALID_DATASET_TYPE:
    if cfg.dataset.name == "seg":
        return SegValidDataset(
            cfg=cfg,
            chunk_features=chunk_features,
            event_df=event_df,
        )
    elif cfg.dataset.name == "detr":
        return DETRValidDataset(
            cfg=cfg,
            chunk_features=chunk_features,
            event_df=event_df,
        )
    elif cfg.dataset.name == "centernet":
        return CenterNetValidDataset(
            cfg=cfg,
            chunk_features=chunk_features,
            event_df=event_df,
        )
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")


def get_test_ds(
    cfg: InferenceConfig,
    chunk_features: dict[str, np.ndarray],
) -> TEST_DATASET_TYPE:
    if cfg.dataset.name == "seg":
        return SegTestDataset(cfg=cfg, chunk_features=chunk_features)
    elif cfg.dataset.name == "detr":
        return DETRTestDataset(cfg=cfg, chunk_features=chunk_features)
    elif cfg.dataset.name == "centernet":
        return CenterNetTestDataset(cfg=cfg, chunk_features=chunk_features)
    else:
        raise ValueError(f"Invalid dataset name: {cfg.dataset.name}")
