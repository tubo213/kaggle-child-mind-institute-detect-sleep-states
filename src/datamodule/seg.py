import random
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # RGB
IMAGENET_STD = [0.229, 0.224, 0.225]  # RGB


###################
# Load Functions
###################
def load_features(
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    train_or_test: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [
            series_dir.name
            for series_dir in (processed_dir / f"{train_or_test}/features").glob("*")
        ]

    for series_id in series_ids:
        series_dir = processed_dir / f"{train_or_test}/features/{series_id}"
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        features[series_dir.name] = np.stack(this_feature, axis=1)

    return features


def load_masks(series_ids: Optional[list[str]], processed_dir: Path) -> dict[str, np.ndarray]:
    masks = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / "train/labels").glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / f"train/labels/{series_id}"
        masks[series_id] = np.load(series_dir / "event_null.npy")

    return masks


def pad_if_needed(x: np.ndarray, max_len: int, pad_value: float = 0.0) -> np.ndarray:
    if len(x) == max_len:
        return x
    num_pad = max_len - len(x)
    n_dim = len(x.shape)
    pad_widths = [(0, num_pad)] + [(0, 0) for _ in range(n_dim - 1)]
    return np.pad(x, pad_width=pad_widths, mode="constant", constant_values=pad_value)


def load_chunk_features(
    duration: int,
    feature_names: list[str],
    series_ids: Optional[list[str]],
    processed_dir: Path,
    train_or_test: str,
) -> dict[str, np.ndarray]:
    features = {}

    if series_ids is None:
        series_ids = [
            series_dir.name
            for series_dir in (processed_dir / f"{train_or_test}/features").glob("*")
        ]

    for series_id in series_ids:
        series_dir = processed_dir / f"{train_or_test}/features/{series_id}"
        this_feature = []
        for feature_name in feature_names:
            this_feature.append(np.load(series_dir / f"{feature_name}.npy"))
        this_feature = np.stack(this_feature, axis=1)
        num_chunks = (len(this_feature) // duration) + 1
        for i in range(num_chunks):
            chunk_feature = this_feature[i * duration : (i + 1) * duration]
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)  # type: ignore
            features[f"{series_id}_{i:07}"] = chunk_feature

    return features  # type: ignore


def load_chunk_masks(
    duration: int, series_ids: Optional[list[str]], processed_dir: Path
) -> dict[str, np.ndarray]:
    masks = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / "train/labels").glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / f"train/labels/{series_id}"
        this_mask = np.load(series_dir / "event_null.npy")
        num_chunks = (len(this_mask) // duration) + 1
        for i in range(num_chunks):
            chunk_mask = this_mask[i * duration : (i + 1) * duration]
            chunk_mask = pad_if_needed(chunk_mask, duration, pad_value=1)  # maskは1で埋める
            masks[f"{series_id}_{i:07}"] = chunk_mask

    return masks


###################
# Augmentation
###################
def random_crop(pos: int, duration: int, max_end) -> tuple[int, int]:
    """Randomly crops with duration length including pos.
    However, 0<=start, end<=max_end
    """
    start = random.randint(max(0, pos - duration), min(pos, max_end - duration))
    end = start + duration
    return start, end


###################
# Label
###################
def get_label(
    this_event_df: pl.DataFrame, n_steps: int, hop_length: int, start: int, end: int
) -> np.ndarray:
    # (start, end)の範囲と(onset, wakeup)の範囲が重なるものを取得
    this_event_df = this_event_df.filter((pl.col("wakeup") >= start) & (pl.col("onset") <= end))
    # 位置を修正
    this_event_df = this_event_df.with_columns(
        ((pl.col("onset") - start) // hop_length + 1),
        ((pl.col("wakeup") - start) // hop_length + 1),
    )

    label = np.zeros((n_steps, 3))
    for onset, wakeup in this_event_df.select(["onset", "wakeup"]).to_numpy():
        if onset >= 0 and onset < n_steps:
            label[onset, 1] = 1
        if wakeup < n_steps and wakeup >= 0:
            label[wakeup, 2] = 1

        onset = max(0, onset)
        wakeup = min(n_steps, wakeup)
        label[onset:wakeup, 0] = 1  # sleep

    return label


# ref: https://www.kaggle.com/competitions/dfl-bundesliga-data-shootout/discussion/360236#2004730
def gaussian_kernel(length: int, sigma: int = 3) -> np.ndarray:
    x = np.ogrid[-length : length + 1]
    h = np.exp(-(x**2) / (2 * sigma * sigma))  # type: ignore
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def gaussian_label(label: np.ndarray, offset: int, sigma: int) -> np.ndarray:
    num_events = label.shape[1]
    for i in range(num_events):
        label[:, i] = np.convolve(label[:, i], gaussian_kernel(offset, sigma), mode="same")

    return label


###################
# Dataset
###################
class TrainDataset(Dataset):
    def __init__(
        self,
        cfg: DictConfig,
        event_df: pl.DataFrame,
        features: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.event_df = event_df.pivot(
            index=["series_id", "night"], columns="event", values="step"
        ).drop_nulls()
        self.features = features
        self.masks = masks
        self.wav_transform = Spectrogram(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        self.eps = 1e-6

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        event = np.random.choice(["onset", "wakeup"], p=[0.5, 0.5])
        pos = self.event_df[idx, event]
        series_id = self.event_df[idx, "series_id"]
        this_event_df = self.event_df.filter(pl.col("series_id") == series_id)

        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        this_mask = self.masks[series_id]  # (n_steps,)
        n_steps = this_feature.shape[0]

        # sample background
        if random.random() < self.cfg.bg_sampling_rate:
            bg_positions = np.where(this_mask == 1)[0].tolist()
            pos = random.sample(bg_positions, 1)[0]

        # crop
        start, end = random_crop(pos, self.cfg.duration, n_steps)
        feature = this_feature[start:end]  # (duration, num_features)

        # TODO: modelのとこでやる. wave to Spectrogram
        imgs = []
        for i in range(feature.shape[1]):
            img = self.wav_transform(torch.FloatTensor(feature[:, i]))
            img = librosa.power_to_db(img.numpy())
            # normalize 0-1 with min-max
            img = (img - img.mean()) / (img.std() + self.eps)
            imgs.append(img)

        # concat
        feature = np.stack(imgs, axis=0)  # (C, n_mels, duration // hop_length + 1)

        # from hard label to gaussian label
        label = get_label(this_event_df, feature.shape[-1], self.cfg.hop_length, start, end)
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], offset=self.cfg.offset, sigma=self.cfg.sigma
        )

        return {
            "series_id": series_id,
            "feature": torch.FloatTensor(feature),
            "label": torch.FloatTensor(label),
        }


class ValidDataset(Dataset):
    def __init__(
        self, cfg: DictConfig, event_df: pl.DataFrame, chunk_features: dict[str, np.ndarray]
    ):
        self.cfg = cfg
        self.event_df = event_df.pivot(
            index=["series_id", "night"], columns="event", values="step"
        ).drop_nulls()
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.wav_transform = Spectrogram(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        self.eps = 1e-6

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]

        # TODO: modelのとこでやる. wave to Spectrogram wave to Spectrogram
        imgs = []
        for i in range(feature.shape[1]):
            img = self.wav_transform(torch.FloatTensor(feature[:, i]))
            img = librosa.power_to_db(img.numpy())
            # normalize 0-1 with min-max
            img = (img - img.mean()) / (img.std() + self.eps)
            imgs.append(img)
        # concat
        feature = np.stack(imgs, axis=0)  # (C, n_mels, duration // hop_length + 1)

        series_id, chunk_id = key.split("_")
        chunk_id = int(chunk_id)
        start = chunk_id * self.cfg.duration
        end = start + self.cfg.duration
        label = get_label(
            self.event_df.filter(pl.col("series_id") == series_id),
            feature.shape[-1],
            self.cfg.hop_length,
            start,
            end,
        )

        return {
            "key": key,
            "feature": torch.FloatTensor(feature),
            "label": torch.FloatTensor(label),
        }


class TestDataset(Dataset):
    def __init__(self, cfg, chunk_features: dict[str, np.ndarray]):
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())
        self.wav_transform = Spectrogram(
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        self.eps = 1e-6

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]

        # TODO: modelのとこでやる. wave to Spectrogram wave to Spectrogram
        imgs = []
        for i in range(feature.shape[1]):
            img = self.wav_transform(torch.FloatTensor(feature[:, i]))
            img = librosa.power_to_db(img.numpy())
            # normalize 0-1 with min-max
            img = (img - img.mean()) / (img.std() + self.eps)
            imgs.append(img)
        # concat
        feature = np.stack(imgs, axis=0)  # (C, n_mels, duration // hop_length + 1)

        return {
            "key": key,
            "feature": torch.FloatTensor(feature),
        }


###################
# DataModule
###################
class SegDataModule(LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.data_dir = Path(cfg.dir.data_dir)
        self.processed_dir = Path(cfg.dir.processed_dir)
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv").drop_nulls()
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.train_series_ids)
        )
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.valid_series_ids)
        )
        # train data
        self.train_features = load_features(
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
            train_or_test="train",
        )
        self.train_masks = load_masks(
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
        )

        # valid data
        self.valid_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.valid_series_ids,
            processed_dir=self.processed_dir,
            train_or_test="train",
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
            masks=self.train_masks,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_dataset = ValidDataset(
            cfg=self.cfg,
            event_df=self.valid_event_df,
            chunk_features=self.valid_chunk_features,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader
