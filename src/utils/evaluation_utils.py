from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import utils.constants as c
from data.datasets import get_data_loaders
from utils.model_utils import load_best_model, load_hyperparameters, load_general_config
from utils.data_utils import load_dataset


def inverse_scale_numpy_array(wind_speeds):
    scalers_dict = joblib.load(Path(f"data/processed/scalers.pkl"))
    scaler = scalers_dict["wind_speed"]

    inverse_scaled = np.zeros_like(wind_speeds)
    inverse_scaled = scaler.inverse_transform(wind_speeds.reshape(-1, 1)).reshape(wind_speeds.shape)

    return inverse_scaled


def inverse_scale_batch_tensor(y_batch, use_global_scaler):
    if use_global_scaler:
        file_name = "global_scalers.pkl"
        scaler_key = "wind_speed"
    else:
        file_name = "local_scalers.pkl"
        scaler_key = f"wind_speed_{c.REFERENCE_STATION_ID}"
    
    scalers_dict = joblib.load(Path(f"data/processed/{file_name}"))
    scaler = scalers_dict[scaler_key]

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
        dataset_df = load_dataset(resolution, split)

        ground_truth_scaled = dataset_df[f"wind_speed_{c.REFERENCE_STATION_ID}"]
        ground_truth_unscaled = inverse_scale_numpy_array(ground_truth_scaled.values)

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

        outs = []
        with torch.no_grad():
            for idx, (x_station, x_global, y) in enumerate(dataloader):
                out = inverse_scale_batch_tensor(model(x_station, x_global)).cpu()
                if idx % forecasting_hours == 0:
                    outs.append(out)

        if not outs:
            final_preds = []
        else:
            stack = torch.cat(outs, dim=0)
            final_preds = stack.view(-1).tolist()

        predictions = final_preds
        predictions_index = calculate_predictions_time_range(
            split, hyperparameters["look_back_hours"]
        )
        prediction_series = pd.Series(predictions, index=predictions_index)

        inference_df[f"run_{run_id}_predictions"] = prediction_series

    return inference_df


def calculate_predictions_time_range(split, look_back_hours):
    df = pd.read_csv(f"data/processed/1hres/{split}.csv", index_col="datetime", parse_dates=True)

    predictions_start_date = df.index[0] + pd.to_timedelta(look_back_hours, unit="h")
    predictions_end_date = df.index[-1]
    predictions_time_range = pd.date_range(
        predictions_start_date, predictions_end_date, freq="1h", inclusive="left"
    )
    return predictions_time_range


def get_feature_selection_results(run_ids, forecasting_hours, wandb_api, base_features=None):
    general_config = load_general_config()
    run_names = [
        f"{forecasting_hours}_hour_forecasting_run_{id}"
        for id in run_ids
    ]
    runs = wandb_api.runs(
        path=f"{general_config['wandb_entity']}/{general_config['wandb_project']}",
        filters={"display_name": {"$in": run_names}},
    )
    records = []

    for run in runs:
        rmse = run.summary.get("best_model_val_rmse_original_domain")
        station_features = run.config.get("station_features") or []
        global_features = run.config.get("global_features") or []

        records.append({
            "run_name": run.name,
            "station_features": tuple(sorted(station_features)),
            "global_features": tuple(sorted(global_features)),
            "RMSE": rmse,
        })

    df = pd.DataFrame(records)
    df["round"] = df["run_name"].apply(
        lambda name: run_names.index(name) if name in run_names else -1
    )
    df = df[df["round"] >= 0]
    df = df.sort_values("round")

    winners = df.groupby("round").apply(lambda g: g.nsmallest(1, "RMSE")).reset_index(drop=True)

    winners["improvement_over_last(%)"] = winners["RMSE"].shift(1) - winners["RMSE"]
    winners["improvement_over_last(%)"] = (
        winners["improvement_over_last(%)"] / winners["RMSE"].shift(1) * 100
    ).round(2)

    winners["station_features"] = winners["station_features"].apply(lambda x: list(x) if x else [])
    winners["global_features"] = winners["global_features"].apply(lambda x: list(x) if x else [])

    return winners.reset_index(drop=True)

