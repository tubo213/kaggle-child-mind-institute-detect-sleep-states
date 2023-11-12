import random

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import resize

from src.conf import InferenceConfig, TrainConfig
from src.utils.common import (
    gaussian_label,
    nearest_valid_size,
    negative_sampling,
    pad_if_needed,
    random_crop,
)


###################
# Label
###################
def get_centernet_label(
    this_event_df: pd.DataFrame, num_frames: int, duration: int, start: int, end: int
) -> np.ndarray:
    # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.query("@start <= wakeup & onset <= @end")

    # labelを作成
    # onset_pos, wakeup_pos, onset_offset, wakeup_offset, onset_bbox_size, wakeup_bbox_size
    label = np.zeros((num_frames, 6))
    for onset, wakeup in this_event_df[["onset", "wakeup"]].to_numpy():
        onset_pos = int((onset - start) / duration * num_frames)
        onset_offset = (onset - start) / duration * num_frames - onset_pos
        wakeup_pos = int((wakeup - start) / duration * num_frames)
        wakeup_offset = (wakeup - start) / duration * num_frames - wakeup_pos

        # 区間に入らない場合は、posをclipする.
        # e.g. num_frames=100, onset_pos=50, wakeup_pos=150
        # -> onset_pos=50, wakeup_pos=100, bbox_size=(100-50)/100=0.5
        bbox_size = (min(wakeup_pos, num_frames) - max(onset_pos, 0)) / num_frames

        if onset_pos >= 0 and onset_pos < num_frames:
            label[onset_pos, 0] = 1
            label[onset_pos, 2] = onset_offset
            label[onset_pos, 4] = bbox_size

        if wakeup_pos < num_frames and wakeup_pos >= 0:
            label[wakeup_pos, 1] = 1
            label[wakeup_pos, 3] = wakeup_offset
            label[wakeup_pos, 5] = bbox_size

    # org_pos = pred_pos + pred_offset, don't use bbox_size. it's for loss.
    return label


class CenterNetTrainDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
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
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_centernet_label(this_event_df, num_frames, self.cfg.duration, start, end)
        label[:, [0, 1]] = gaussian_label(
            label[:, [0, 1]], offset=self.cfg.dataset.offset, sigma=self.cfg.dataset.sigma
        )

        return {
            "series_id": series_id,
            "feature": feature,  # (num_features, upsampled_num_frames)
            "label": torch.FloatTensor(label),  # (pred_length, num_classes)
        }


class CenterNetValidDataset(Dataset):
    def __init__(
        self,
        cfg: TrainConfig,
        chunk_features: dict[str, np.ndarray],
        event_df: pl.DataFrame,
    ):
        self.cfg = cfg
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
        num_frames = self.upsampled_num_frames // self.cfg.downsample_rate
        label = get_centernet_label(
            self.event_df.query("series_id == @series_id").reset_index(drop=True),
            num_frames,
            self.cfg.duration,
            start,
            end,
        )
        return {
            "key": key,
            "feature": feature,  # (num_features, duration)
            "label": torch.FloatTensor(label),  # (duration, num_classes)
        }


class CenterNetTestDataset(Dataset):
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
