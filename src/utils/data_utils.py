import pandas as pd
from pathlib import Path


def load_dataset(processed=False, is_global_scaled=None):
    dir = "processed" if processed else "raw"
    if processed:
        dir = f"processed/"
        name = "dataset_gl_scaled.csv" if is_global_scaled else "dataset_lc_scaled.csv"
    else:
        dir = "raw"
        name = "dataset.csv"
    return pd.read_csv(Path(f"data/{dir}/{name}"), parse_dates=True, index_col="datetime")


def load_metadata():
    return pd.read_csv("data/raw/station_metadata.csv", index_col="id", dtype={"id": str})


def load_indices(split):
    return pd.read_csv(Path(f"data/processed/{split}_indices.csv"))


def get_time_steps(time, resolution="10min"):
    return int(pd.Timedelta(time) / pd.to_timedelta(resolution))
