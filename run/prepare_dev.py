from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    data_dir = Path(cfg.dir.data_dir)
    series_df = pl.scan_parquet(data_dir / "train_series.parquet")
    event_df = pl.scan_csv(data_dir / "train_events.csv")
    sample_200_series_ids = (
        series_df.select("series_id")
        .unique()
        .collect(streaming=True)
        .sample(200)
        .get_column("series_id")
    )
    dev_series_df = series_df.filter(pl.col("series_id").is_in(sample_200_series_ids))
    dev_event_df = event_df.filter(pl.col("series_id").is_in(sample_200_series_ids))

    processed_dir = Path(cfg.dir.processed_dir)
    processed_dir.mkdir(exist_ok=True, parents=True)
    dev_series_df.sink_parquet(processed_dir / "dev_series.parquet")
    dev_event_df.sink_csv(processed_dir / "dev_events.csv")
    print("Done")


if __name__ == "__main__":
    main()
