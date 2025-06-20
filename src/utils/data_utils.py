import pandas as pd
from pathlib import Path
import numpy as np


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


def get_wind_dir_relative_freq(wind_dir_series, min_deg, max_deg):
    return (((wind_dir_series >= min_deg) & (wind_dir_series <= max_deg)).sum()) / len(
        wind_dir_series
    )


def get_stations_in_sec_rel_freq(ref_station, normal_stations, angle_min, angle_max):
    ref_point = ref_station.geometry.iloc[0]
    station_points = normal_stations.geometry

    dx = station_points.x - ref_point.x
    dy = station_points.y - ref_point.y

    angles_rad = np.arctan2(dx, dy)
    angles_deg = (np.degrees(angles_rad) + 360) % 360

    in_interval = (angles_deg > angle_min) & (angles_deg <= angle_max)
    count = in_interval.sum()
    relative = count / len(angles_deg)

    print(f"Stations in interval ({angle_min}°, {angle_max}°]: {count}")
    print(f"Relative frequency: {relative:.2%}")
