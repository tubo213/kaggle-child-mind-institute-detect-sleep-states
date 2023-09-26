import random
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset


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


def load_labels(
    label_names: str, series_ids: Optional[list[str]], processed_dir: Path
) -> dict[str, np.ndarray]:
    labels = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / "train/labels").glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / f"train/labels/{series_id}"
        this_label = []
        for label_name in label_names:
            this_label.append(np.load(series_dir / f"{label_name}.npy"))
        labels[series_id] = np.stack(this_label, axis=1)

    return labels


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


def load_chunk_labels(
    duration: int,
    label_names: list[str],
    series_ids: list[str],
    processed_dir: Path,
) -> dict[str, np.ndarray]:
    labels = {}

    if series_ids is None:
        series_ids = [series_dir.name for series_dir in (processed_dir / "train/labels").glob("*")]

    for series_id in series_ids:
        series_dir = processed_dir / f"train/labels/{series_id}"
        this_label = []
        for label_name in label_names:
            this_label.append(np.load(series_dir / f"{label_name}.npy"))
        this_label = np.stack(this_label, axis=1)
        num_chunks = (len(this_label) // duration) + 1
        for i in range(num_chunks):
            chunk_label = this_label[i * duration : (i + 1) * duration]
            chunk_label = pad_if_needed(chunk_label, duration, pad_value=0)  # type: ignore
            labels[f"{series_id}_{i:07}"] = chunk_label

    return labels  # type: ignore


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
        labels: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
    ):
        self.cfg = cfg
        self.event_df = event_df
        self.features = features
        self.labels = labels
        self.masks = masks

    def __len__(self):
        return len(self.event_df)

    def __getitem__(self, idx):
        pos = self.event_df[idx, "step"]
        series_id = self.event_df[idx, "series_id"]

        # extract data matching series_id
        this_feature = self.features[series_id]  # (n_steps, num_features)
        this_label = self.labels[series_id].astype(np.float64)  # (n_steps, 3)
        this_mask = self.masks[series_id]  # (n_steps,)
        n_steps = this_feature.shape[0]

        # sample background
        if random.random() < self.cfg.bg_sampling_rate:
            bg_positions = np.where(this_mask == 1)[0].tolist()
            pos = random.sample(bg_positions, 1)[0]

        # crop
        start, end = random_crop(pos, self.cfg.duration, n_steps)
        feature = this_feature[start:end]  # (duration, num_features)
        label = this_label[start:end]  # label has 1 at the event position, (duration, 3)

        # from hard label to gaussian label
        label[:, [1, 2]] = gaussian_label(
            label[:, [1, 2]], offset=self.cfg.offset, sigma=self.cfg.sigma
        )

        return {
            "series_id": series_id,
            "feature": torch.FloatTensor(feature),
            "label": torch.FloatTensor(label),
        }


class ValidDataset(Dataset):
    def __init__(self, chunk_features: dict[str, np.ndarray], chunk_labels: dict[str, np.ndarray]):
        self.chunk_features = chunk_features
        self.chunk_labels = chunk_labels
        self.keys = list(chunk_features.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]
        label = self.chunk_labels[key]

        return {
            "key": key,
            "feature": torch.FloatTensor(feature),
            "label": torch.FloatTensor(label),
        }


class TestDataset(Dataset):
    def __init__(self, chunk_features: dict[str, np.ndarray]):
        self.chunk_features = chunk_features
        self.keys = list(chunk_features.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        feature = self.chunk_features[key]

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

        # def setup(self, stage: str) -> None:
        #     if stage == "fit" or stage == "validate" or stage is None:
        self.event_df = pl.read_csv(self.data_dir / "train_events.csv")
        self.train_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.train_series_ids)
        ).drop_nulls()
        self.valid_event_df = self.event_df.filter(
            pl.col("series_id").is_in(self.cfg.split.valid_series_ids)
        ).drop_nulls()
        # train data
        self.train_features = load_features(
            feature_names=self.cfg.features,
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
            train_or_test="train",
        )
        self.train_labels = load_labels(
            label_names=self.cfg.labels,
            series_ids=self.cfg.split.train_series_ids,
            processed_dir=self.processed_dir,
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
        self.valid_chunk_labels = load_chunk_labels(
            duration=self.cfg.duration,
            label_names=self.cfg.labels,
            series_ids=self.cfg.split.valid_series_ids,
            processed_dir=self.processed_dir,
        )

        # if stage == "test" or stage == "predict" or stage is None:
        # test data
        self.test_chunk_features = load_chunk_features(
            duration=self.cfg.duration,
            feature_names=self.cfg.features,
            series_ids=None,
            processed_dir=self.processed_dir,
            train_or_test="test",
        )

    def train_dataloader(self):
        train_dataset = TrainDataset(
            cfg=self.cfg,
            event_df=self.train_event_df,
            features=self.train_features,
            labels=self.train_labels,
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
            chunk_features=self.valid_chunk_features,
            chunk_labels=self.valid_chunk_labels,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_dataset = TestDataset(
            chunk_features=self.test_chunk_features,
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )
        return test_loader

    def predict_dataloader(self):
        return self.test_dataloader()
