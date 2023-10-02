import shutil
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


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
    ).select("series_id", *FEATURE_NAMES)
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
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase
    chunk_dir: Path = processed_dir / "chunk"
    all_dir: Path = processed_dir / "all"

    # ディレクトリが存在する場合は削除
    if chunk_dir.exists() and cfg.chunk:
        shutil.rmtree(chunk_dir)
        print(f"Removed Chunk dir: {chunk_dir}")
    if all_dir.exists() and not cfg.chunk:
        shutil.rmtree(all_dir)
        print(f"Removed All dir: {all_dir}")

    with trace("Load series"):
        series_df = (
            pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
            .with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
            )
            .select([pl.col("series_id"), pl.col("anglez"), pl.col("enmo"), pl.col("timestamp")])
            .collect(streaming=True)
        )
    unique_series_ids = series_df.select("series_id").unique().to_numpy().reshape(-1)
    with trace("Save features"):
        for series_id in tqdm(unique_series_ids):
            this_series_df = series_df.filter(pl.col("series_id") == series_id)
            this_series_df = add_feature(this_series_df)
            if cfg.chunk:
                # 特徴量をduration毎にchunkしてnpyで保存
                save_chunk_each_series(
                    series_id,
                    this_series_df,
                    FEATURE_NAMES,
                    cfg.duration,
                    chunk_dir,
                )
            else:
                # 特徴量をそれぞれnpyで保存
                save_each_series(this_series_df, FEATURE_NAMES, all_dir / series_id)

            # 保存したらメモリ解放
            series_df = series_df.filter(pl.col("series_id") != series_id)


if __name__ == "__main__":
    main()
