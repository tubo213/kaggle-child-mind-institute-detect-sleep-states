from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig
from tqdm import tqdm

from src.utils.common import pad_if_needed, trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "month_sin",
    "month_cos",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
]


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def make_feature_df(series_df: pl.DataFrame):
    series_df = series_df.with_columns(
        *to_coord(pl.col("month"), 12, "month"),
        *to_coord(pl.col("hour"), 24, "hour"),
        *to_coord(pl.col("minute"), 60, "minute"),
    )
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy()
        np.save(output_dir / f"{col_name}.npy", x)


def save_chunk_each_series(
    series_id: str,
    this_series_df: pl.DataFrame,
    columns: list[str],
    duration: int,
    output_dir: Path,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    num_chunks = (len(this_series_df) // duration) + 1
    for i in range(num_chunks):
        chunk_dir = output_dir / f"{series_id}_{i:07}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        chunk_df = this_series_df[i * duration : (i + 1) * duration]
        for col_name in columns:
            chunk_feature = chunk_df.get_column(col_name).to_numpy()
            chunk_feature = pad_if_needed(chunk_feature, duration, pad_value=0)
            np.save(chunk_dir / f"{col_name}.npy", chunk_feature)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    # Read series_df
    with trace("read series_df"):
        series_df = (
            pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.train_or_test_or_dev}_series.parquet",
                low_memory=True,
            )
            .with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp").dt.month().alias("month"),
                    pl.col("timestamp").dt.hour().alias("hour"),
                    pl.col("timestamp").dt.minute().alias("minute"),
                ]
            )
            .collect()
        )

    for series_id, this_series_df in tqdm(
        series_df.group_by("series_id"), desc="generate features"
    ):
        # 特徴量を追加
        feature_df = make_feature_df(this_series_df)

        # series_id毎に特徴量をそれぞれnpyで保存
        feature_output_dir = (
            Path(cfg.dir.processed_dir) / cfg.train_or_test_or_dev / "features" / series_id
        )
        if cfg.train_or_test_or_dev == "train":
            save_each_series(feature_df, FEATURE_NAMES, feature_output_dir)
        else:
            save_chunk_each_series(
                series_id,
                feature_df,
                FEATURE_NAMES,
                cfg.duration,
                Path(cfg.dir.processed_dir) / cfg.train_or_test_or_dev / "features",
            )


if __name__ == "__main__":
    main()
