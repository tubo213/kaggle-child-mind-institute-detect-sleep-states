from pathlib import Path

import hydra
import polars as pl
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="prepare_data", version_base="1.2")
def main(cfg: DictConfig):
    data_dir = Path(cfg.dir.data_dir)
    series_df = pl.scan_parquet(data_dir / "train_series.parquet")
    sample_200_series_ids = (
        series_df.select("series_id").unique().collect(streaming=True).sample(200).get_column("series_id")
    )
    dev_series_df = series_df.filter(pl.col("series_id").is_in(sample_200_series_ids))
    dev_series_df.sink_parquet(data_dir / "dev_series.parquet")
    print("Done")


if __name__ == "__main__":
    main()
