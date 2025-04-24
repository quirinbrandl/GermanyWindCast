import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import utils.constants as c
from pathlib import Path


class WindDataset(Dataset):

    def __init__(
        self,
        split,
        look_back_hours,
        resolution,
        station_ids,
        station_features,
        global_features,
        forecasting_horizon_hours,
    ):
        x_data = pd.read_csv(
            Path(f"data/processed/{resolution}res/{split}.csv"),
            parse_dates=True,
            index_col="datetime",
        )

        predictions_start_date = (
            x_data.index[0]
            - pd.to_timedelta(resolution)
            + pd.to_timedelta(look_back_hours, unit="h")
        )
        predictions_end_date = x_data.index[-1]
        self.predictions_time_range = pd.date_range(
            predictions_start_date, predictions_end_date, freq="1h", inclusive="right"
        )

        x_station_cols = [
            f"{feature}_{station_id}" for feature in station_features for station_id in station_ids
        ]

        x_station_data = x_data[x_station_cols]
        self.x_station_data = self.shape_station_data(x_station_data, station_features, station_ids)
        self.x_global_data = x_data[global_features].values if global_features else np.empty((0,))

        y_data = pd.read_csv(
            Path(f"data/processed/1hres/{split}.csv"), parse_dates=True, index_col="datetime"
        )
        target_col = f"wind_speed_{c.REFERENCE_STATION_ID}"
        y_data = y_data[target_col]
        self.y_data = y_data.values

        self.look_back_hours = look_back_hours
        self.rows_per_hour = int(pd.Timedelta("1h") / pd.to_timedelta(resolution))
        self.look_back_rows = self.rows_per_hour * look_back_hours

        self.forecasting_horizon_hours = forecasting_horizon_hours

        self.number_of_samples = len(self.y_data) - look_back_hours - forecasting_horizon_hours + 1

    def __len__(self):
        return self.number_of_samples

    def __getitem__(self, index):
        look_back_start = index * self.rows_per_hour
        look_back_end = look_back_start + self.look_back_rows

        x_station = self.x_station_data[look_back_start:look_back_end]
        x_global = self.x_global_data[look_back_start:look_back_end]

        forecasting_horizon_start = index + self.look_back_hours
        forecasting_horizon_end = forecasting_horizon_start + self.forecasting_horizon_hours
        y = self.y_data[forecasting_horizon_start:forecasting_horizon_end]

        x_station_tensor = torch.tensor(x_station, dtype=torch.float)
        x_global_tensor = torch.tensor(
            x_global,
            dtype=torch.float,
        )
        y_tensor = torch.tensor(y, dtype=torch.float)

        return x_station_tensor, x_global_tensor, y_tensor

    def shape_station_data(self, station_data, station_features, station_ids):
        reshaped_data = np.zeros((len(station_data), len(station_features), len(station_ids)))

        for feature_idx, feature_name in enumerate(station_features):
            for station_idx, station_id in enumerate(station_ids):
                col_name = f"{feature_name}_{station_id}"
                reshaped_data[:, feature_idx, station_idx] = station_data[col_name].values

        return reshaped_data


def get_data_loaders(
    look_back_hours,
    resolution,
    station_ids,
    station_features,
    global_features,
    forecasting_horizon_hours,
    splits,
    batch_size=1,
    num_workers=1,
):
    datasets = [
        (
            WindDataset(
                split,
                look_back_hours,
                resolution,
                station_ids,
                station_features,
                global_features,
                forecasting_horizon_hours,
            ),
            split == "train",
        )
        for split in splits
    ]

    return [
        DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        for (dataset, shuffle) in datasets
    ]
