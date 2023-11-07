import shutil
from pathlib import Path

import hydra
import numpy as np
import polars as pl
from tqdm import tqdm

from src.conf import PrepareDataConfig
from src.utils.common import trace

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}


FEATURE_NAMES = [
    "anglez",
    "enmo",
    "step",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "minute_sin",
    "minute_cos",
    "anglez_sin",
    "anglez_cos",
]

ANGLEZ_MEAN = -8.810476
ANGLEZ_STD = 35.521877
ENMO_MEAN = 0.041315
ENMO_STD = 0.101829


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def deg_to_rad(x: pl.Expr) -> pl.Expr:
    return np.pi / 180 * x


def add_feature(series_df: pl.DataFrame) -> pl.DataFrame:
    series_df = (
        series_df.with_row_count("step")
        .with_columns(
            *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
            *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
            *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
            pl.col("step") / pl.count("step"),
            pl.col('anglez_rad').sin().alias('anglez_sin'),
            pl.col('anglez_rad').cos().alias('anglez_cos'),
        )
        .select("series_id", *FEATURE_NAMES)
    )
    return series_df


def save_each_series(this_series_df: pl.DataFrame, columns: list[str], output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    for col_name in columns:
        x = this_series_df.get_column(col_name).to_numpy(zero_copy_only=True)
        np.save(output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: PrepareDataConfig):
    processed_dir: Path = Path(cfg.dir.processed_dir) / cfg.phase

    # ディレクトリが存在する場合は削除
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
        print(f"Removed {cfg.phase} dir: {processed_dir}")

    with trace("Load series"):
        # scan parquet
        if cfg.phase in ["train", "test"]:
            series_lf = pl.scan_parquet(
                Path(cfg.dir.data_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        elif cfg.phase == "dev":
            series_lf = pl.scan_parquet(
                Path(cfg.dir.processed_dir) / f"{cfg.phase}_series.parquet",
                low_memory=True,
            )
        else:
            raise ValueError(f"Invalid phase: {cfg.phase}")

        # preprocess
        series_df = (
            series_lf.with_columns(
                pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
                deg_to_rad(pl.col("anglez")).alias("anglez_rad"),
                (pl.col("anglez") - ANGLEZ_MEAN) / ANGLEZ_STD,
                (pl.col("enmo") - ENMO_MEAN) / ENMO_STD,
            )
            .select(
                [
                    pl.col("series_id"),
                    pl.col("anglez"),
                    pl.col("enmo"),
                    pl.col("timestamp"),
                    pl.col("anglez_rad"),
                ]
            )
            .collect(streaming=True)
            .sort(by=["series_id", "timestamp"])
        )
        n_unique = series_df.get_column("series_id").n_unique()
    with trace("Save features"):
        for series_id, this_series_df in tqdm(series_df.group_by("series_id"), total=n_unique):
            # 特徴量を追加
            this_series_df = add_feature(this_series_df)

            # 特徴量をそれぞれnpyで保存
            series_dir = processed_dir / series_id  # type: ignore
            save_each_series(this_series_df, FEATURE_NAMES, series_dir)


if __name__ == "__main__":
    main()
