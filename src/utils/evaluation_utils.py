from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import utils.constants as c
from data.datasets import get_data_loaders
from utils.model_utils import load_best_model, load_hyperparameters


def inverse_scale_numpy_array(wind_speeds, resolution):
    scalers_dict = joblib.load(Path(f"data/processed/{resolution}res/scalers.pkl"))
    scaler = scalers_dict["wind_speed"]

    inverse_scaled = np.zeros_like(wind_speeds)
    inverse_scaled = scaler.inverse_transform(wind_speeds.reshape(-1, 1)).reshape(wind_speeds.shape)

    return inverse_scaled


def inverse_scale_batch_tensor(y_batch, resolution):
    scalers_dict = joblib.load(Path(f"data/processed/{resolution}res/scalers.pkl"))
    scaler = scalers_dict["wind_speed"]

    y_batch_np = (
        y_batch.detach().cpu().numpy()
    )  # scikit learn's scaler works only with np.array on cpu
    device = y_batch.device

    inverse_scaled = np.zeros_like(y_batch_np)
    inverse_scaled = scaler.inverse_transform(y_batch_np.reshape(-1, 1)).reshape(y_batch_np.shape)

    return torch.tensor(inverse_scaled, device=device, dtype=y_batch.dtype)


def add_ground_truth_to_df(df, ground_truth_resolutions, split):
    df_copy = df.copy(deep=True)

    for resolution in ground_truth_resolutions:
        data_path = Path(f"data/processed/{resolution}res/{split}.csv")
        dataset_df = pd.read_csv(data_path, parse_dates=True, index_col="datetime")

        ground_truth_scaled = dataset_df[f"wind_speed_{c.REFERENCE_STATION_ID}"]
        ground_truth_unscaled = inverse_scale_numpy_array(ground_truth_scaled.values, resolution)

        ground_truth_index = ground_truth_scaled.index
        ground_truth = pd.Series(ground_truth_unscaled, index=ground_truth_index)

        df_copy = df_copy.reindex(df_copy.index.union(ground_truth_index))
        df_copy[f"ground_truth_{resolution}_resolution"] = ground_truth

    return df_copy


def get_inference_df(run_ids, forecasting_hours, ground_truth_resolutions=["1h"], split="eval"):
    inference_df = add_ground_truth_to_df(pd.DataFrame(), ground_truth_resolutions, split)

    for run_id in run_ids:
        hyperparameters = load_hyperparameters(run_id, forecasting_hours)
        look_back_hours = hyperparameters["look_back_hours"]

        model = load_best_model(run_id, forecasting_hours, "cpu")

        dataloader = get_data_loaders(
            look_back_hours=look_back_hours,
            resolution=hyperparameters["resolution"],
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            forecasting_horizon_hours=forecasting_hours,
            splits=["eval"],
        )[0]

        predictions = []
        with torch.no_grad():
            for x_station, x_global, y in dataloader:
                prediction = torch.flatten(
                    inverse_scale_batch_tensor(
                        model(x_station, x_global), hyperparameters["resolution"]
                    )
                ).item()
                predictions.append(prediction)

        predictions_index = dataloader.dataset.predictions_time_range
        prediction_series = pd.Series(predictions, index=predictions_index)

        inference_df[f"run_{run_id}_predictions"] = prediction_series

    return inference_df

