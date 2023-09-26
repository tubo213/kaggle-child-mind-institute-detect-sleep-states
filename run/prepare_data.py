from pathlib import Path

import hydra
import numpy as np
import polars as pl
from omegaconf import DictConfig

SERIES_SCHEMA = {
    "series_id": pl.Utf8,
    "step": pl.UInt32,
    "anglez": pl.Float32,
    "enmo": pl.Float32,
}

EVENT_SCHEMA = {
    "series_id": pl.Utf8,
    "night": pl.UInt32,
    "event": pl.Utf8,
    "step": pl.UInt32,
    "timestamp": pl.Utf8,
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

EVENT_NAMES = ["event_null", "event_onset", "event_wakeup", "awake"]


def to_coord(x: pl.Expr, max_: int, name: str) -> list[pl.Expr]:
    rad = 2 * np.pi * (x % max_) / max_
    x_sin = rad.sin()
    x_cos = rad.cos()

    return [x_sin.alias(f"{name}_sin"), x_cos.alias(f"{name}_cos")]


def make_feature_df(series_df: pl.DataFrame):
    series_df = series_df.with_columns(
        *to_coord(pl.col("timestamp").dt.month(), 12, "month"),
        *to_coord(pl.col("timestamp").dt.hour(), 24, "hour"),
        *to_coord(pl.col("timestamp").dt.minute(), 60, "minute"),
        (pl.col("anglez") - (-90)) / (90 - (-90)),
        (pl.col("enmo") - pl.col("enmo").min()) / (pl.col("enmo").max() - pl.col("enmo").min()),
    )
    return series_df


def make_label_df(series_df: pl.DataFrame, event_df: pl.DataFrame):
    label_df = series_df.select(["series_id", "step"]).join(
        event_df.select(["series_id", "step", "event", "awake"]),
        on=["series_id", "step"],
        how="left",
    )
    label_df = label_df.with_columns(
        pl.col("awake").fill_null(strategy="backward").over("series_id").fill_null(1).cast(pl.Int8)
    )
    label_df = label_df.to_dummies("event")
    return label_df


def save_for_each_series_id(df: pl.DataFrame, columns: list[str], output_dir: Path):
    for series_id, df in df.group_by("series_id"):
        this_output_dir = output_dir / series_id  # type: ignore
        this_output_dir.mkdir(parents=True, exist_ok=True)

        for col_name in columns:
            x = df.get_column(col_name).to_numpy()
            np.save(this_output_dir / f"{col_name}.npy", x)


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    # Read series_df
    series_df = pl.read_parquet(Path(cfg.dir.data_dir) / f"{cfg.train_or_test}_series.parquet")

    # cast series_df to appropriate type
    series_df = series_df.select(
        *[pl.col(col).cast(d_type) for col, d_type in SERIES_SCHEMA.items()],
        pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%z"),
    )

    # 特徴量を追加
    feature_df = make_feature_df(series_df)

    # series_id毎に特徴量をそれぞれnpyで保存
    feature_output_dir = Path(cfg.dir.processed_dir) / cfg.train_or_test / "features"
    save_for_each_series_id(feature_df, FEATURE_NAMES, feature_output_dir)

    # trainの場合はlabelを作成
    if cfg.train_or_test == "train":
        # Read event_df
        event_df = pl.read_csv(Path(cfg.dir.data_dir) / "train_events.csv", dtypes=EVENT_SCHEMA)

        # Drop null event
        event_df = event_df.drop_nulls()

        # awake追加
        event_df = event_df.with_columns(
            pl.col("event").map(lambda s: s == "onset").alias("awake")
        )

        # labelを作成
        label_df = make_label_df(series_df, event_df)

        # series_id毎にlabelをそれぞれnpyで保存
        label_output_dir = Path(cfg.dir.processed_dir) / cfg.train_or_test / "labels"
        save_for_each_series_id(label_df, EVENT_NAMES, label_output_dir)
    elif cfg.train_or_test == "test":
        pass
    else:
        raise ValueError(f"Invalid train_or_test: {cfg.train_or_test}")


if __name__ == "__main__":
    main()
