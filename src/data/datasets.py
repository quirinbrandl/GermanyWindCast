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
        self.data = pd.read_csv(
            Path(f"data/processed/{resolution}res/{split}.csv"),
            parse_dates=True,
            index_col="datetime",
        )

        target_data = pd.read_csv(
            Path(f"data/processed/1hres/{split}.csv"), parse_dates=True, index_col="datetime"
        )
        target_col = f"wind_speed_{c.REFERENCE_STATION_ID}"
        self.target_data = target_data[target_col]

        self.start_date = self.data.index[0]
        self.look_back_hours = look_back_hours
        self.station_ids = station_ids
        self.resolution = resolution
        self.station_features = station_features
        self.global_features = global_features
        self.forecasting_horizon_hours = forecasting_horizon_hours

    def __len__(self):
        end_date = self.data.index[-1]
        effective_start_date = self.start_date - pd.to_timedelta(self.resolution)

        number_of_hours = int((end_date - effective_start_date).total_seconds() / 3600)

        return number_of_hours - self.look_back_hours - self.forecasting_horizon_hours + 1

    def __getitem__(self, index):
        x_station_cols = [
            f"{feature}_{station_id}"
            for feature in self.station_features
            for station_id in self.station_ids
        ]
        x_global_cols = self.global_features

        look_back_start_date = self.start_date + index * pd.to_timedelta(1, "h")

        ## Needed since the measured values are always the average w.r.t. the time since last timestamp
        look_back_effective_start_date = look_back_start_date - pd.to_timedelta(self.resolution)
        look_back_end_date = (
            look_back_effective_start_date + self.look_back_hours * pd.to_timedelta(1, "h")
        )

        x_rows = pd.date_range(look_back_start_date, look_back_end_date, freq=self.resolution)

        x_station_data = self.data.loc[x_rows, x_station_cols]
        x_global_data = self.data.loc[x_rows, x_global_cols]

        x_station_data_shaped = self.reshape_station_data(x_station_data)
        x_global_data_shaped = x_global_data.values

        x_station_tensor = torch.tensor(x_station_data_shaped, dtype=torch.float)
        x_global_tensor = torch.tensor(x_global_data_shaped, dtype=torch.float)

        forecast_horizon_end_date = look_back_end_date + pd.to_timedelta(
            self.forecasting_horizon_hours, "h"
        )
        y_hours = pd.date_range(
            look_back_end_date, forecast_horizon_end_date, freq="1h", inclusive="right"
        )

        y_data = self.target_data.loc[y_hours]
        y_shaped_data = y_data.values
        y_tensor = torch.tensor(y_shaped_data, dtype=torch.float)

        return x_station_tensor, x_global_tensor, y_tensor

    def reshape_station_data(self, data):
        reshaped_data = np.zeros((len(data), len(self.station_features), len(self.station_ids)))

        for feature_idx, feature_name in enumerate(self.station_features):
            for station_idx, station_id in enumerate(self.station_ids):
                col_name = f"{feature_name}_{station_id}"
                reshaped_data[:, feature_idx, station_idx] = data[col_name].values

        return reshaped_data


def get_data_loaders(
    look_back_hours,
    resolution,
    station_ids,
    station_features,
    global_features,
    forecasting_horizon_hours,
    batch_size,
    splits
):
    datasets = [
        WindDataset(
            split,
            look_back_hours,
            resolution,
            station_ids,
            station_features,
            global_features,
            forecasting_horizon_hours,
        )
        for split in splits
    ]

    return [DataLoader(dataset, batch_size=batch_size, shuffle=False) for dataset in datasets]
