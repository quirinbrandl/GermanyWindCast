from pathlib import Path

import numpy as np
import pandas as pd
import torch
from haversine import Unit, haversine_vector
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data as GraphData
from torch_geometric.data import Dataset as GraphDataset
from torch_geometric.loader import DataLoader as GraphDataLoader

import utils.constants as c
from utils.data_utils import load_dataset, load_indices, get_time_steps


class BaseWindDataset:
    def __init__(
        self,
        split,
        look_back_hours,
        resolution,
        station_ids,
        station_features,
        global_features,
        forecasting_horizon_hours,
        use_global_scaling,
    ):
        self.step_size_x = get_time_steps(resolution)
        self.step_size_y = get_time_steps("1h")

        self.idxs = load_indices(split)["index"].values
        self.look_back_rows = get_time_steps(f"{look_back_hours}h")
        self.forecasting_rows = get_time_steps(f"{forecasting_horizon_hours}h")
        self.look_back_steps = get_time_steps(f"{look_back_hours}h", resolution=resolution)

        self.x_station_data, self.x_global_data, self.y_data = self._load_data(
            resolution,
            station_features,
            station_ids,
            global_features,
            use_globally_scaled=use_global_scaling,
        )

    def _load_data(
        self, resolution, station_features, station_ids, global_features, use_globally_scaled
    ):
        x_data = load_dataset(
            processed=True, is_global_scaled=use_globally_scaled, resolution=resolution
        )
        y_data = load_dataset(processed=True, is_global_scaled=use_globally_scaled, resolution="1h")
        x_station_cols = [
            f"{feature}_{station_id}" for feature in station_features for station_id in station_ids
        ]
        x_station_data = x_data[x_station_cols]

        x_global_data = x_data[global_features].values if global_features else np.empty((0,))

        y_data = y_data[f"wind_speed_{c.REFERENCE_STATION_ID}"].values

        return x_station_data, x_global_data, y_data


class WindDataset(BaseWindDataset, Dataset):

    def __init__(
        self,
        split,
        look_back_hours,
        resolution,
        station_ids,
        station_features,
        global_features,
        forecasting_horizon_hours,
        use_global_scaling,
    ):
        super().__init__(
            split=split,
            look_back_hours=look_back_hours,
            resolution=resolution,
            station_ids=station_ids,
            station_features=station_features,
            global_features=global_features,
            forecasting_horizon_hours=forecasting_horizon_hours,
            use_global_scaling=use_global_scaling,
        )

        self.x_station_data = self.x_station_data.values

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        look_back_start = self.idxs[index]
        look_back_end = look_back_start + self.look_back_rows

        x_station = self.x_station_data[look_back_start : look_back_end : self.step_size_x]
        x_station_tensor = torch.tensor(x_station, dtype=torch.float)

        x_global = self.x_global_data[look_back_start : look_back_end : self.step_size_x]
        x_global_tensor = torch.tensor(
            x_global,
            dtype=torch.float,
        )

        x_tensor = torch.concat((x_station_tensor, x_global_tensor), dim=1)

        forecasting_horizon_end = look_back_end + self.forecasting_rows
        y = self.y_data[look_back_end : forecasting_horizon_end : self.step_size_y]
        y_tensor = torch.tensor(y, dtype=torch.float)

        return x_tensor, y_tensor


class WindDatasetSpatial(BaseWindDataset, GraphDataset):

    def __init__(
        self,
        split,
        look_back_hours,
        resolution,
        station_ids,
        station_features,
        global_features,
        forecasting_horizon_hours,
        weighting,
        knns,
        use_global_scaling,
    ):
        BaseWindDataset.__init__(
            self,
            split=split,
            look_back_hours=look_back_hours,
            resolution=resolution,
            station_ids=station_ids,
            station_features=station_features,
            global_features=global_features,
            forecasting_horizon_hours=forecasting_horizon_hours,
            use_global_scaling=use_global_scaling,
        )
        GraphDataset.__init__(self)

        self.x_station_data = self._shape_station_data(
            self.x_station_data, station_features, station_ids
        )

        self.num_stations = len(station_ids)
        self.num_station_features = len(station_features)

        if weighting:
            weight_matrix = self._calculate_weight_matr(weighting, station_ids)

        self.edge_index = (
            self._build_knn_edge_index(weight_matrix, knns) if knns else self._build_fc_edge_index()
        )

        if weighting:
            self.weights = self._build_weights(weight_matrix)

    def len(self):
        return len(self.idxs)

    def get(self, index):
        look_back_start = self.idxs[index]
        look_back_end = look_back_start + self.look_back_rows

        x_station = self.x_station_data[look_back_start : look_back_end : self.step_size_x]
        x_station = x_station.reshape(
            self.look_back_steps * self.num_stations, self.num_station_features
        )
        x_station_tensor = torch.tensor(x_station, dtype=torch.float)

        x_global = self.x_global_data[look_back_start : look_back_end : self.step_size_x]
        x_global_tensor = torch.tensor(
            x_global,
            dtype=torch.float,
        ).unsqueeze(
            0
        )  # PyG concatenates along dimension 0 when creating batches

        forecasting_horizon_end = look_back_end + self.forecasting_rows
        y = self.y_data[look_back_end : forecasting_horizon_end : self.step_size_y]
        y_tensor = torch.tensor(y, dtype=torch.float).unsqueeze(0)

        data = GraphData(
            x=x_station_tensor,
            edge_index=self.edge_index,
            weights=self.weights,
            y=y_tensor,
            x_global=x_global_tensor,
        )

        return data

    def _shape_station_data(self, station_data, station_features, station_ids):
        look_back_rows = len(station_data)
        num_station_features = len(station_features)
        num_stations = len(station_ids)

        shaped_data = np.zeros((look_back_rows, num_stations, num_station_features))

        for feature_idx, feature_name in enumerate(station_features):
            for station_idx, station_id in enumerate(station_ids):
                col_name = f"{feature_name}_{station_id}"
                col_vals = station_data[col_name].values
                shaped_data[:, station_idx, feature_idx] = col_vals

        return shaped_data

    def _calculate_dist_matr_km(self, station_ids):
        coords = self._load_station_coords(station_ids)
        dist_matr = torch.from_numpy(
            haversine_vector(coords, coords, unit=Unit.KILOMETERS, comb=True)
        )

        return dist_matr

    def _calculate_gaussian_sigma_km(self, dist_matr):
        i_upper, j_upper = np.triu_indices(dist_matr.shape[0], k=1)
        pairwise_dists = dist_matr[i_upper, j_upper]

        sigma = torch.std(pairwise_dists, unbiased=False).item()
        return float(sigma)

    def _calculate_similarity_matr(self, station_ids, weighting):
        train_1h_res_df = pd.read_csv("data/processed/1hres/train.csv")
        train_1h_res_df = train_1h_res_df[
            [f"wind_speed_{station_id}" for station_id in station_ids]
        ]

        if "pearson" in weighting:
            corr = train_1h_res_df.corr(method="pearson").values
        elif "spearman" in weighting:
            corr = train_1h_res_df.corr(method="spearman").values
        elif "cos" in weighting:
            X = train_1h_res_df.values.astype(float)
            norms = np.linalg.norm(X, axis=0)
            norms[norms == 0] = 1.0
            dotprods = X.T @ X
            corr = dotprods / np.outer(norms, norms)
        else:
            raise ValueError(
                f"The weighting method {weighting} does not include a supported method for calculating similarity."
            )

        # rescale to [0, 1] for use with PyG
        W = (corr + 1.0) / 2.0
        return torch.from_numpy(W).float()

    def _calculate_weight_matr(self, weighting, station_ids):
        if weighting == "distance":
            dist_matr = self._calculate_dist_matr_km(station_ids)
            gaussian_sigma_km = self._calculate_gaussian_sigma_km(dist_matr)

            return torch.exp(-(dist_matr**2) / (2 * gaussian_sigma_km**2))
        elif "similarity" in weighting:
            return self._calculate_similarity_matr(station_ids, weighting)
        else:
            raise ValueError(f"{weighting} is not a supported weighting method.")

    def _build_similarity_weights(self, station_ids):
        matr = self._calculate_similarity_matr(station_ids)
        src_nodes_station_idxs = self._get_station_idx(self.edge_index[0])
        trgt_nodes_station_idxs = self._get_station_idx(self.edge_index[1])
        weights = matr[src_nodes_station_idxs, trgt_nodes_station_idxs]

        return weights.float()

    def _build_weights(self, weight_matr):
        src_nodes_station_idxs = self._get_station_idx(self.edge_index[0])
        trgt_nodes_station_idxs = self._get_station_idx(self.edge_index[1])
        weights = weight_matr[src_nodes_station_idxs, trgt_nodes_station_idxs]

        return weights.float()

    def _build_fc_edge_index(self):
        src, dst = [], []
        for i in range(self.num_stations):
            for j in range(self.num_stations):
                if i != j:
                    src.append(i)
                    dst.append(j)

        edge_index = self._expand_to_edge_index(src, dst)
        return edge_index

    def _build_knn_edge_index(self, weight_matr, k):
        topk = torch.topk(
            weight_matr, k + 1, dim=1  # includes self at pos0 (which we want to exclude)
        ).indices
        src, dst = [], []
        for i in range(self.num_stations):
            neighbors = topk[i][topk[i] != i][:k]
            for j in neighbors:
                src.append(i)
                dst.append(j.item())

        edge_index = self._expand_to_edge_index(src, dst)
        return edge_index

    def _expand_to_edge_index(self, src, dst):
        src = torch.tensor(src, dtype=torch.long)
        dst = torch.tensor(dst, dtype=torch.long)

        all_src, all_dst = [], []
        for t in range(self.look_back_rows):
            offset = t * self.num_stations
            all_src.append(src + offset)
            all_dst.append(dst + offset)

        edge_index = torch.vstack([torch.cat(all_src), torch.cat(all_dst)])
        return edge_index

    def _load_station_coords(self, station_ids):
        station_data_path = Path("data/raw/station_metadata.csv")
        station_data = pd.read_csv(station_data_path, index_col="id", dtype={"id": str})
        station_data = station_data.loc[station_ids]

        return list(zip(station_data["geographic_latitude"], station_data["geographic_longitude"]))

    def _get_time_step(self, node_idx):
        return node_idx // self.num_stations

    def _get_station_idx(self, node_idx):
        return node_idx % self.num_stations


def get_data_loaders(
    look_back_hours,
    resolution,
    station_ids,
    station_features,
    global_features,
    forecasting_horizon_hours,
    splits,
    use_global_scaling,
    batch_size=1,
    num_workers=1,
    is_spatial=False,
    knns=None,
    weighting=None,
):
    DatasetType = WindDatasetSpatial if is_spatial else WindDataset
    DataLoaderType = GraphDataLoader if is_spatial else DataLoader

    ds_kwargs = dict(
        look_back_hours=look_back_hours,
        resolution=resolution,
        station_ids=station_ids,
        station_features=station_features,
        global_features=global_features,
        forecasting_horizon_hours=forecasting_horizon_hours,
        use_global_scaling=use_global_scaling,
    )
    if is_spatial:
        ds_kwargs["knns"] = knns
        ds_kwargs["weighting"] = weighting

    datasets = [
        (
            DatasetType(split=split, **ds_kwargs),
            split == "train",
        )
        for split in splits
    ]

    return [
        DataLoaderType(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            pin_memory=True,
            num_workers=num_workers,
            persistent_workers=True,
        )
        for (dataset, shuffle) in datasets
    ]
