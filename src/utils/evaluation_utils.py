from pathlib import Path
from sklearn.metrics import root_mean_squared_error
import joblib
import wandb
import numpy as np
import pandas as pd
import torch

import utils.constants as c
from data.datasets import get_data_loaders
from utils.model_utils import (
    create_model,
    is_model_spatial,
    load_best_model,
    load_hyperparameters,
    load_general_config,
    make_predictions,
)
from utils.data_utils import load_dataset, load_indices, load_metadata
import openmeteo_requests

import pandas as pd
import requests_cache
from retry_requests import retry


def get_scaler(use_global_scaler):
    if use_global_scaler:
        file_name = "global_scalers.pkl"
        scaler_key = "wind_speed"
    else:
        file_name = "local_scalers.pkl"
        scaler_key = f"wind_speed_{c.REFERENCE_STATION_ID}"

    scalers_dict = joblib.load(Path(f"data/processed/{file_name}"))
    return scalers_dict[scaler_key]


def inverse_scale_batch_tensor(y_batch, use_global_scaler):
    scaler = get_scaler(use_global_scaler)

    y_batch_np = (
        y_batch.detach().cpu().numpy()
    )  # scikit learn's scaler works only with np.array on cpu
    device = y_batch.device

    inverse_scaled = np.zeros_like(y_batch_np)
    inverse_scaled = scaler.inverse_transform(y_batch_np.reshape(-1, 1)).reshape(y_batch_np.shape)

    return torch.tensor(inverse_scaled, device=device, dtype=y_batch.dtype)


def inverse_scale_numpy_array(arr, use_global_scaler):
    scaler = get_scaler(use_global_scaler)
    return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(arr.shape)


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
    run_names = [f"{forecasting_hours}_hour_forecasting_run_{id}" for id in run_ids]
    runs = wandb_api.runs(
        path=f"{general_config['wandb_entity']}/{general_config['wandb_project']}",
        filters={"display_name": {"$in": run_names}},
    )
    records = []

    for run in runs:
        rmse = run.summary.get("best_model_val_rmse_original_domain")
        station_features = run.config.get("station_features") or []
        global_features = run.config.get("global_features") or []

        records.append(
            {
                "run_name": run.name,
                "station_features": tuple(sorted(station_features)),
                "global_features": tuple(sorted(global_features)),
                "RMSE": rmse,
            }
        )

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


def get_temporal_analysis_results(forecasting_hours_to_run_id_mapping):
    wandb.login()
    api = wandb.Api()
    general_config = load_general_config()
    wandb_base_path = f"{general_config["wandb_entity"]}/{general_config["wandb_project"]}"

    records = []
    for forecasting_hours in forecasting_hours_to_run_id_mapping.keys():
        run_names = [
            f"{forecasting_hours}_hour_forecasting_run_{run_id}"
            for run_id in forecasting_hours_to_run_id_mapping[forecasting_hours]
        ]

        runs = api.runs(path=wandb_base_path, filters={"display_name": {"$in": run_names}})
        for run in runs:
            forecasting_hours = run.name.split("_")[0]
            run_id = run.name.split("_")[-1]
            rmse = run.summary.get("best_model_val_rmse_original_domain")
            resolution = run.config.get("resolution")
            look_back_hours = run.config.get("look_back_hours")
            model_architecture = run.config.get("model_architecture")

            records.append(
                {
                    "forecasting_hours": forecasting_hours,
                    "run_id": run_id,
                    "resolution": resolution,
                    "look_back_hours": look_back_hours,
                    "model_architecture": model_architecture,
                    "RMSE": rmse,
                }
            )

    df = pd.DataFrame(records)
    return df


def get_spatial_analysis_results(forecasting_hours_to_run_id_mapping):
    wandb.login()
    api = wandb.Api()
    general_config = load_general_config()
    wandb_base_path = f"{general_config["wandb_entity"]}/{general_config["wandb_project"]}"

    records = []
    for forecasting_hours in forecasting_hours_to_run_id_mapping.keys():
        run_names = [
            f"{forecasting_hours}_hour_forecasting_run_{run_id}"
            for run_id in forecasting_hours_to_run_id_mapping[forecasting_hours]
        ]

        runs = api.runs(path=wandb_base_path, filters={"display_name": {"$in": run_names}})
        for run in runs:
            forecasting_hours = run.name.split("_")[0]
            run_id = run.name.split("_")[-1]
            rmse = run.summary.get("best_model_val_rmse_original_domain")
            model_architecture = run.config.get("model_architecture")
            station_ids = run.config.get("station_ids")

            records.append(
                {
                    "forecasting_hours": int(forecasting_hours),
                    "run_id": run_id,
                    "model_architecture": model_architecture,
                    "station_ids": station_ids,
                    "number_of_stations": len(station_ids),
                    "RMSE": rmse,
                }
            )

    df = pd.DataFrame(records)
    return df


def calculate_percentage_improvement(baseline_rmse, model_rmse):
    return (1 - model_rmse / baseline_rmse) * 100


def get_rmse_for_subset(
    run_id, forecasting_hours, device, split, indices=None, batch_size=1, num_workers=1
):
    model = load_best_model(run_id, forecasting_hours, device)

    hyperparameters = load_hyperparameters(run_id, forecasting_hours)
    use_global_scaling = hyperparameters["use_global_scaling"]
    test_data_loader = get_data_loaders(
        look_back_hours=hyperparameters["look_back_hours"],
        resolution=hyperparameters["resolution"],
        station_ids=hyperparameters["station_ids"],
        station_features=hyperparameters["station_features"],
        global_features=hyperparameters["global_features"],
        forecasting_horizon_hours=forecasting_hours,
        splits=[split],
        is_spatial=is_model_spatial(model),
        weighting=hyperparameters.get("weighting"),
        knns=hyperparameters.get("knns"),
        use_global_scaling=use_global_scaling,
        num_workers=num_workers,
        batch_size=batch_size,
        indices=indices,
    )[0]

    return run_evaluation_and_get_rmse(test_data_loader, model, device, use_global_scaling)


def run_evaluation_and_get_rmse(data_loader, model, device, use_global_scaling):
    mse = torch.nn.MSELoss()
    mse_losses = []
    with torch.no_grad():

        for batch in data_loader:
            predictions, y = make_predictions(batch, model, device)

            y = inverse_scale_batch_tensor(y, use_global_scaling)
            predictions = inverse_scale_batch_tensor(predictions, use_global_scaling)

            mse_losses.append(mse(predictions, y).item())

    total_mse = sum(mse_losses) / len(mse_losses)
    return np.sqrt(total_mse)


def get_persistence_rmse(forecasting_hours, split):
    station_ids = [c.REFERENCE_STATION_ID]
    station_features = [f"wind_speed"]
    model_architecture = "persistence"
    persistence_model = create_model(
        {
            "station_ids": station_ids,
            "station_features": station_features,
            "model_architecture": model_architecture,
        },
        forecasting_hours,
    )

    test_data_loader = get_data_loaders(
        look_back_hours=1,
        resolution="10min",
        station_ids=[c.REFERENCE_STATION_ID],
        station_features=station_features,
        global_features=[],
        forecasting_horizon_hours=forecasting_hours,
        splits=[split],
        is_spatial=False,
        use_global_scaling=True,
    )[0]

    return run_evaluation_and_get_rmse(test_data_loader, persistence_model, "cpu", True)


def get_icon_rmse(forecasting_hours, split, variant):
    indices = load_indices(split)["valid_indices"]
    processed_data = load_dataset(processed=True, is_global_scaled=True)
    time_indices = processed_data.iloc[indices].index
    processed_data["target"] = inverse_scale_numpy_array(
        processed_data["target"].values, use_global_scaler=True
    )

    icon_forecast = get_icon_forecast(time_indices, variant).dropna()
    targets = processed_data["target"]

    predictions_arr = icon_forecast.iloc[:, :forecasting_hours].to_numpy()
    observed_blocks = []
    for h in range(forecasting_hours):
        offset = pd.Timedelta("10min") + h * pd.Timedelta("60min")
        observed = targets.reindex(icon_forecast.index + offset).to_numpy()
        observed_blocks.append(observed.reshape(-1, 1))

    observed_arr = np.hstack(observed_blocks)

    rmse = root_mean_squared_error(observed_arr.ravel(), predictions_arr.ravel())
    return rmse


def get_icon_forecast(time_indices, variant):
    start_date = time_indices[0]
    end_date = time_indices[-1]
    icon_data_path = Path(f"data/nwp/icon_{variant}.csv")

    if not icon_data_path.exists() or not (
        start_date
        <= pd.read_csv(icon_data_path, parse_dates=True, index_col="datetime").index[0]
        <= end_date
    ):
        metadata = load_metadata()
        reference_location = metadata.loc[
            c.REFERENCE_STATION_ID, ["geographic_latitude", "geographic_longitude"]
        ]

        cache_session = requests_cache.CachedSession(".cache", expire_after=3600)
        retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
        openmeteo = openmeteo_requests.Client(session=retry_session)

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": reference_location["geographic_latitude"],
            "longitude": reference_location["geographic_longitude"],
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": "wind_speed_10m",
            "wind_speed_unit": "ms",
            "models": f"icon_{variant}",
        }
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()

        hourly_data = {
            "datetime": pd.date_range(
                start=pd.to_datetime(hourly.Time(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left",
            )
        }

        hourly_data["wind_speed"] = hourly_wind_speed_10m

        hourly_df = pd.DataFrame(hourly_data)
        hourly_df.set_index("datetime", inplace=True)

        final_df = transform_icon_forecast(hourly_df, time_indices, variant)
        final_df.to_csv(icon_data_path)

    return pd.read_csv(icon_data_path, parse_dates=True, index_col="datetime")


def transform_icon_forecast(icon_df, time_indices, variant):
    icon_indices = time_indices[
        (time_indices.minute == 0) & time_indices.hour.isin(get_icon_variant_run_hours(variant))
    ]
    cols = [f"{h}h" for h in range(1, 9)]
    out = pd.DataFrame(index=icon_indices, columns=cols, dtype=float)
    measured_data = load_dataset()

    for t0 in icon_indices:
        wind_speed_now = measured_data.at[t0, f"wind_speed_{c.REFERENCE_STATION_ID}"]
        icon_next = icon_df.at[t0 + pd.Timedelta(hours=1), "wind_speed"]
        out.at[t0, "1h"] = (wind_speed_now + icon_next) / 2.0

        for h in range(2, 9):
            first = icon_df.at[t0 + pd.Timedelta(hours=h - 1), "wind_speed"]
            second = icon_df.at[t0 + pd.Timedelta(hours=h), "wind_speed"]
            out.at[t0, f"{h}h"] = (first + second) / 2.0
    return out


def get_icon_indices(split, variant):
    indices = load_indices(split)["valid_indices"]
    processed_data = load_dataset(processed=True, is_global_scaled=True)

    out = []
    for counter, idx in enumerate(indices):
        timestamp = processed_data.index[idx]
        if (timestamp.minute == 0) and (timestamp.hour in get_icon_variant_run_hours(variant)):
            out.append(counter)

    return out


def get_icon_variant_run_hours(variant):
    if variant in ["d2", "eu"]:
        return [3 * i for i in range(8)]
    else:
        return [6 * i for i in range(4)]
