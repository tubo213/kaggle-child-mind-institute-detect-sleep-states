from dataclasses import dataclass
from typing import Any


@dataclass
class DirConfig:
    data_dir: str
    processed_dir: str
    output_dir: str
    model_dir: str
    sub_dir: str


@dataclass
class SplitConfig:
    name: str
    train_series_ids: list[str]
    valid_series_ids: list[str]


@dataclass
class FeatureExtractorConfig:
    name: str
    params: dict[str, Any]


@dataclass
class DecoderConfig:
    name: str
    params: dict[str, Any]


@dataclass
class ModelConfig:
    name: str
    params: dict[str, Any]


@dataclass
class TrainerConfig:
    epochs: int
    accelerator: str
    use_amp: bool
    debug: bool
    gradient_clip_val: float
    accumulate_grad_batches: int
    monitor: str
    monitor_mode: str
    check_val_every_n_epoch: int


@dataclass
class DatasetConfig:
    name: str
    batch_size: int
    num_workers: int
    offset: int
    sigma: int
    bg_sampling_rate: float


@dataclass
class AugmentationConfig:
    mixup_prob: float
    mixup_alpha: float
    cutmix_prob: float
    cutmix_alpha: float


@dataclass
class PostProcessConfig:
    score_th: float
    distance: int


@dataclass
class OptimizerConfig:
    lr: float


@dataclass
class SchedulerConfig:
    num_warmup_steps: int


@dataclass
class WeightConfig:
    exp_name: str
    run_name: str


@dataclass
class PrepareDataConfig:
    dir: DirConfig
    phase: str


@dataclass
class TrainConfig:
    exp_name: str
    seed: int
    batch_size: int
    num_workers: int
    duration: int
    downsample_rate: int
    upsample_rate: int
    labels: list[str]
    features: list[str]
    split: SplitConfig
    dir: DirConfig
    model: ModelConfig
    feature_extractor: FeatureExtractorConfig
    decoder: DecoderConfig
    trainer: TrainerConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    dataset: DatasetConfig
    aug: AugmentationConfig
    pp: PostProcessConfig


@dataclass
class InferenceConfig:
    exp_name: str
    phase: str
    seed: int
    batch_size: int
    num_workers: int
    duration: int
    downsample_rate: int
    upsample_rate: int
    use_amp: bool
    labels: list[str]
    features: list[str]
    dir: DirConfig
    model: ModelConfig
    feature_extractor: FeatureExtractorConfig
    decoder: DecoderConfig
    weight: WeightConfig
    dataset: DatasetConfig
    aug: AugmentationConfig
    pp: PostProcessConfig
