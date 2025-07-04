import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.nn import GAT, GCN

import utils.constants as c
from utils.data_utils import get_time_steps


class PersistenceModel(nn.Module):

    def __init__(self, forecasting_hours, station_ids, station_features):
        super(PersistenceModel, self).__init__()
        self.forecasting_hours = forecasting_hours
        self.wind_speed_idx = station_features.index("wind_speed")
        self.reference_station_idx = station_ids.index(c.REFERENCE_STATION_ID)
        self.num_stations = len(station_ids)
        self.num_features = len(station_features)

    def forward(self, x_tensor):
        last_observed_at_ref_station = x_tensor[
            :, -1, self.wind_speed_idx * self.num_stations + self.reference_station_idx
        ]
        return last_observed_at_ref_station.unsqueeze(1).repeat(1, self.forecasting_hours)


class BaseModel(nn.Module):
    def __init__(self, forecasting_hours, station_ids, station_features, global_features):
        super(BaseModel, self).__init__()
        self.forecasting_hours = forecasting_hours
        self.station_ids = station_ids
        self.station_features = station_features
        self.global_features = global_features
        self.use_global_features = bool(global_features) if global_features is not None else False
        self.num_global_features = len(global_features) if global_features else 0


class MLP(BaseModel):
    def __init__(
        self,
        look_back_hours,
        forecasting_hours,
        station_ids,
        station_features,
        global_features,
        num_hidden_dense_layers,
        dense_hidden_size,
        dropout_rate,
        resolution,
    ):
        super(MLP, self).__init__(forecasting_hours, station_ids, station_features, global_features)

        look_back_window = int(pd.to_timedelta(look_back_hours, "h") / pd.to_timedelta(resolution))
        input_size = (
            (len(station_ids) * len(station_features)) + self.num_global_features
        ) * look_back_window

        layers = []
        for _ in range(num_hidden_dense_layers):
            layers.append(nn.Linear(input_size, dense_hidden_size))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            input_size = dense_hidden_size

        layers.append(nn.Linear(dense_hidden_size, forecasting_hours))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        batch_size = x.size(0)
        x_station_flat = x.reshape(batch_size, -1)

        return self.net(x_station_flat)


class WindLSTM(BaseModel):
    def __init__(
        self,
        forecasting_hours,
        station_ids,
        station_features,
        global_features,
        num_hidden_lstm_layers,
        num_hidden_dense_layers,
        lstm_hidden_size,
        dense_hidden_size,
        dropout_rate,
    ):
        super(WindLSTM, self).__init__(forecasting_hours, station_ids, station_features, global_features)

        self.lstm = nn.LSTM(
            input_size=len(station_ids) * len(station_features) + self.num_global_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_hidden_lstm_layers,
            dropout=dropout_rate,
            batch_first=True,
        )

        dense_input_size = lstm_hidden_size
        dense_layers = []
        for _ in range(num_hidden_dense_layers):
            dense_layers.append(nn.Linear(dense_input_size, dense_hidden_size))
            dense_layers.append(nn.ReLU())
            if dropout_rate > 0:
                dense_layers.append(nn.Dropout(dropout_rate))
            dense_input_size = dense_hidden_size

        dense_layers.append(
            nn.Linear(
                dense_hidden_size if len(dense_layers) > 0 else lstm_hidden_size, forecasting_hours
            )
        )
        self.mlp_head = nn.Sequential(*dense_layers)

    def forward(self, x):
        lstm_output, _ = self.lstm(x)
        lstm_most_recent = lstm_output[:, -1, :]
        predictions = self.mlp_head(lstm_most_recent)

        return predictions


class BaseGNNModel(BaseModel):
    def __init__(
        self,
        forecasting_hours,
        station_ids,
        station_features,
        global_features,
        hidden_channels,
        resolution,
        look_back_hours,
        use_residual,
    ):
        super(BaseGNNModel, self).__init__(
            forecasting_hours, station_ids, station_features, global_features
        )

        self.look_back_window = int(
            pd.to_timedelta(look_back_hours, "h") / pd.to_timedelta(resolution)
        )
        self.use_residual = use_residual
        self.station_embedding = nn.Linear(len(station_features), hidden_channels)

    def process_gnn_output(self, gnn_output, batch_size):
        gcn_output_shaped = gnn_output.reshape(
            batch_size, self.look_back_window, len(self.station_ids), -1
        )
        ref_station_index = self.station_ids.index(c.REFERENCE_STATION_ID)
        gcn_output_ref_station = gcn_output_shaped[:, :, ref_station_index, :]

        return gcn_output_ref_station

    def get_lstm_input(self, gcn_output_ref_station, x_global):
        return (
            torch.cat([gcn_output_ref_station, x_global], dim=2)
            if self.use_global_features
            else gcn_output_ref_station
        )


class WindGCN(BaseGNNModel):
    def __init__(
        self,
        hidden_channels,
        dense_hidden_size,
        num_hidden_gcn_layers,
        num_hidden_dense_layers,
        dropout_rate,
        station_features,
        station_ids,
        global_features,
        resolution,
        look_back_hours,
        forecasting_hours,
        use_residual,
    ):
        super(WindGCN, self).__init__(
            forecasting_hours,
            station_ids,
            station_features,
            global_features,
            hidden_channels,
            resolution,
            look_back_hours,
            use_residual,
        )

        self.gcn = GCN(hidden_channels, hidden_channels, num_hidden_gcn_layers, dropout=dropout_rate)

        mlp_head_layers = []
        
        look_back_steps = get_time_steps(f"{look_back_hours}h", resolution=resolution)
        input_size = (
            (hidden_channels + self.num_global_features) * look_back_steps
        )

        for _ in range(num_hidden_dense_layers):
            mlp_head_layers.append(nn.Linear(input_size, dense_hidden_size))
            mlp_head_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_head_layers.append(nn.Dropout(dropout_rate))

            input_size = dense_hidden_size
        mlp_head_layers.append(nn.Linear(dense_hidden_size, forecasting_hours))
        self.mlp_head = nn.Sequential(*mlp_head_layers)

    def forward(self, data):
        batch_size = data.num_graphs
        x_global = data.x_global
        x_station = data.x
        edge_index = data.edge_index
        weights = data.weights

        x_station_embedded = self.station_embedding(x_station)
        gcn_output = self.gcn(x_station_embedded, edge_index, edge_weight=weights)
        if self.use_residual:
            gcn_output += x_station_embedded

        gcn_output_ref_station = self.process_gnn_output(gcn_output, batch_size)

        linear_input = gcn_output_ref_station.reshape(batch_size, -1)
        if self.use_global_features:
            linear_input_global = x_global.reshape(batch_size, -1)
            linear_input = torch.cat([linear_input, linear_input_global], dim=1)

        predictions = self.mlp_head(linear_input)
        return predictions


class BaseGNNLSTM(BaseGNNModel):
    def __init__(
        self,
        forecasting_hours,
        station_ids,
        station_features,
        global_features,
        hidden_channels,
        dropout_rate,
        resolution,
        look_back_hours,
        use_residual,
        num_hidden_lstm_layers,
        lstm_hidden_size,
        num_hidden_dense_layers,
        dense_hidden_size,
    ):
        super(BaseGNNLSTM, self).__init__(
            forecasting_hours,
            station_ids,
            station_features,
            global_features,
            hidden_channels,
            resolution,
            look_back_hours,
            use_residual,
        )

        self.lstm = nn.LSTM(
            hidden_channels + self.num_global_features,
            lstm_hidden_size,
            num_hidden_lstm_layers,
            batch_first=True,
            dropout=dropout_rate,
        )


        mlp_head_layers = []
        
        input_size = lstm_hidden_size

        for _ in range(num_hidden_dense_layers):
            mlp_head_layers.append(nn.Linear(input_size, dense_hidden_size))
            mlp_head_layers.append(nn.ReLU())
            if dropout_rate > 0:
                mlp_head_layers.append(nn.Dropout(dropout_rate))

            input_size = dense_hidden_size
        mlp_head_layers.append(nn.Linear(dense_hidden_size if len(mlp_head_layers) > 0 else input_size, forecasting_hours))
        self.mlp_head = nn.Sequential(*mlp_head_layers)

    def forward(self, data):
        batch_size = data.num_graphs
        x_global = data.x_global
        x_station = data.x
        edge_index = data.edge_index
        weights = data.weights

        x_station_embedded = self.station_embedding(x_station)
        gnn_output = self.apply_gnn(x_station_embedded, edge_index, weights)
        if self.use_residual:
            gnn_output += x_station_embedded

        gcn_output_ref_station = self.process_gnn_output(gnn_output, batch_size)

        lstm_input = self.get_lstm_input(gcn_output_ref_station, x_global)
        lstm_output, _ = self.lstm(lstm_input)
        lstm_most_recent = lstm_output[:, -1, :]

        predictions = self.mlp_head(lstm_most_recent)
        return predictions

    def apply_gnn(self, x_station_embedded, edge_index, weights):
        raise NotImplementedError("Subclasses must implement this method")


class GCNLSTM(BaseGNNLSTM):
    def __init__(
        self,
        hidden_channels,
        num_hidden_gcn_layers,
        dropout_rate,
        station_features,
        station_ids,
        global_features,
        resolution,
        look_back_hours,
        forecasting_hours,
        num_hidden_lstm_layers,
        lstm_hidden_size,
        use_residual,
        num_hidden_dense_layers,
        dense_hidden_size,
    ):
        super(GCNLSTM, self).__init__(
            forecasting_hours,
            station_ids,
            station_features,
            global_features,
            hidden_channels,
            dropout_rate,
            resolution,
            look_back_hours,
            use_residual,
            num_hidden_lstm_layers,
            lstm_hidden_size,
            num_hidden_dense_layers,
            dense_hidden_size,
        )

        self.gcn = GCN(hidden_channels, hidden_channels, num_hidden_gcn_layers, dropout=dropout_rate)

    def apply_gnn(self, x_station_embedded, edge_index, weights):
        return self.gcn(x_station_embedded, edge_index, edge_weight=weights)


class GATLSTM(BaseGNNLSTM):
    def __init__(
        self,
        hidden_channels,
        lstm_hidden_size,
        use_residual,
        num_hidden_gat_layers,
        dropout_rate,
        station_features,
        station_ids,
        global_features,
        resolution,
        look_back_hours,
        forecasting_hours,
        num_hidden_lstm_layers,
        heads,
    ):
        super(GATLSTM, self).__init__(
            forecasting_hours,
            station_ids,
            station_features,
            global_features,
            hidden_channels,
            dropout_rate,
            resolution,
            look_back_hours,
            use_residual,
            num_hidden_lstm_layers,
            lstm_hidden_size,
        )

        self.heads = heads
        self.gat = GAT(
            hidden_channels,
            hidden_channels,
            num_hidden_gat_layers,
            dropout=dropout_rate,
            heads=heads,
        )

    def apply_gnn(self, x_station_embedded, edge_index, weights):
        return self.gat(x_station_embedded, edge_index)
