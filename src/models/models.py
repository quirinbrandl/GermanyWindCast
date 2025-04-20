import torch
import torch.nn as nn
import utils.constants as c
import pandas as pd


class PersistenceModel(nn.Module):
    """Simple persistence model that predicts the last value observed."""

    def __init__(self, forecasting_hours, wind_speed_idx, reference_station_idx):
        super(PersistenceModel, self).__init__()
        self.forecasting_hours = forecasting_hours
        self.wind_speed_idx = wind_speed_idx
        self.reference_station_idx = reference_station_idx

    def forward(self, x_station_feats, x_global_feats):
        last_observed = x_station_feats[:, -1, self.wind_speed_idx, self.reference_station_idx]
        return last_observed.unsqueeze(1).repeat(1, self.forecasting_hours)


class MLP(nn.Module):

    def __init__(
        self,
        look_back_hours,
        forecasting_horizon,
        station_ids,
        station_features,
        global_features,
        num_hidden_layers,
        hidden_size,
        dropout_rate,
        resolution,
    ):
        super(MLP, self).__init__()

        look_back_window = int(pd.to_timedelta(look_back_hours, "h") / pd.to_timedelta(resolution))
        input_size = (
            (len(station_ids) * len(station_features)) + len(global_features)
        ) * look_back_window

        layers = []
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, forecasting_horizon))

        self.net = nn.Sequential(*layers)

    def forward(self, x_station_feats, x_global_feats):
        batch_size = x_station_feats.size(0)

        x_station_flat = x_station_feats.view(batch_size, -1)
        x_global_flat = x_global_feats.view(batch_size, -1)

        x = torch.cat([x_station_flat, x_global_flat], dim=1)

        return self.net(x)
