import random

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from src.conf import InferenceConfig, TrainConfig
from src.utils.common import nearest_valid_size, negative_sampling, pad_if_needed, random_crop


###################
# Label
###################
def get_detr_label(
    this_event_df: pd.DataFrame,
    duration: int,
    start: int,
    end: int,
    max_det: int = 20,
) -> np.ndarray:
    # # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    label = np.zeros((max_det, 3))  # (num_frames, [objectness, onset, wakeup])
    # 1dbboxのラベルを作成
    for i, (onset, wakeup) in enumerate(this_event_df[["onset", "wakeup"]].to_numpy()):
        onset = (onset - start) / duration  # 相対時間に変換
        wakeup = (wakeup - start) / duration  # 相対時間に変換
        label[i] = np.array([1, onset, wakeup])

    # 0 <= onset, wakeup <= 1になるようにclip
    label[:, 1:] = np.clip(label[:, 1:], 0, 1)

    return label


class DETRTrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.max_det = cfg.model.params["max_det"]
        self.event_df: pd.DataFrame = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.features = features
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = self.event_df.at[idx, event]
        series_id = self.event_df.at[idx, "series_id"]
        self.event_df["series_id"]
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        n_steps = this_feature.shape[0]

        # sample background
        if random.random() < self.cfg.dataset.bg_sampling_rate:
            pos = negative_sampling(this_event_df, n_steps)

        # crop
        if n_steps > self.cfg.duration:
            start, end = random_crop(pos, self.cfg.duration, n_steps)
            feature = this_feature[start:end]
        else:
            start, end = 0, self.cfg.duration
            feature = pad_if_needed(this_feature, self.cfg.duration)

        # upsample
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        # from hard label to gaussian label
        self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_detr_label(this_event_df, self.cfg.duration, start, end, max_det=self.max_det)

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (max_det, [objectness, onset, wakeup])
        }


class DETRValidDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
        self.max_det = cfg.model.params["max_det"]
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.event_df = (
            event_df.pivot(index=["series_id", "night"], columns="event", values="step")
            .drop_nulls()
            .to_pandas()
        )
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        self.upsampled_num_frames // self.cfg.downsample_rate
        this_event_df = self.event_df.query("series_id == @series_id").reset_index(drop=True)
        label = get_detr_label(this_event_df, self.cfg.duration, start, end, max_det=self.max_det)
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class DETRTestDataset(Dataset):
    def __init__(
        self,
        cfg: InferenceConfig,
        chunk_features: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.num_features = len(cfg.features)
        self.upsampled_num_frames = nearest_valid_size(
            int(self.cfg.duration * self.cfg.upsample_rate), self.cfg.downsample_rate
        )

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        feature = torch.FloatTensor(feature.T).unsqueeze(0)  # (1, num_features, duration)
        feature = resize(
            feature,
            size=[self.num_features, self.upsampled_num_frames],
            antialias=False,
        ).squeeze(0)

        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
        }
