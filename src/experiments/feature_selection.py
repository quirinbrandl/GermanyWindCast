import subprocess
import sys

import wandb
from utils.evaluation_utils import get_feature_selection_results
from utils.model_utils import (
    calculate_next_run_id,
    load_general_config,
)
from utils.experiment_utils import run_training_with_config
from experiments.config import OPTIMAL_FEATURES, MODEL_ARCHS


def run_training_with_features(station_features, global_features):
    hyperparameters = {"station_features": station_features, "global_features": global_features}
    run_training_with_config(hyperparameters)


def run_feature_selection(
    baseline_station_features, station_features_to_test, global_features_to_test
):

    forecasting_hours = load_general_config()["forecasting_hours"]
    next_run_id = calculate_next_run_id(forecasting_hours)
    last_round_best_rmse = float("inf")

    curr_round_best_rmse, curr_round_best_station_features, curr_round_best_global_features = (
        perform_baseline_run(baseline_station_features, forecasting_hours, next_run_id)
    )
    next_run_id += 1

    while curr_round_best_rmse < last_round_best_rmse:
        remaining_station_features_to_test = get_remaining_features(
            station_features_to_test, curr_round_best_station_features
        )
        remaining_global_features_to_test = get_remaining_features(
            global_features_to_test, curr_round_best_global_features
        )
        if (
            len(remaining_station_features_to_test) == 0
            and len(remaining_global_features_to_test) == 0
        ):
            break

        station_run_ids, next_run_id = run_station_feature_selection(
            curr_round_best_station_features,
            curr_round_best_global_features,
            remaining_station_features_to_test,
            next_run_id,
        )

        global_run_ids, next_run_id = run_global_feature_selection(
            curr_round_best_station_features,
            curr_round_best_global_features,
            remaining_global_features_to_test,
            next_run_id,
        )

        run_ids = station_run_ids + global_run_ids
        result = get_feature_selection_results(run_ids, forecasting_hours, wandb.Api())
        best_index = result["RMSE"].idxmin()

        last_round_best_rmse = curr_round_best_rmse
        curr_round_best_rmse = result.loc[best_index, "RMSE"]
        curr_round_best_station_features, curr_round_best_global_features = result.loc[
            best_index, ["station_features", "global_features"]
        ]


def get_remaining_features(features_to_test, best_features):
    return [
        feature_to_test
        for feature_to_test in features_to_test
        if feature_to_test[0] not in best_features
    ]


def run_station_feature_selection(
    base_station_features, base_global_features, station_features_to_test, next_run_id
):
    run_ids = []

    for station_feature in station_features_to_test:
        run_training_with_features(base_station_features + station_feature, base_global_features)
        run_ids.append(next_run_id)
        next_run_id += 1

    return run_ids, next_run_id


def perform_baseline_run(baseline_station_features, forecasting_hours, next_run_id):
    baseline_global_features = []
    baseline_run_id = next_run_id

    run_training_with_features(baseline_station_features, baseline_global_features)

    baseline_df = get_feature_selection_results([baseline_run_id], forecasting_hours, wandb.Api())
    curr_round_best_rmse = float(baseline_df["RMSE"].iloc[0])

    return curr_round_best_rmse, baseline_station_features, baseline_global_features


def run_global_feature_selection(
    base_station_features, base_global_features, global_features_to_test, next_run_id
):
    run_ids = []

    for global_feature in global_features_to_test:
        run_training_with_features(base_station_features, base_global_features + global_feature)
        run_ids.append(next_run_id)
        next_run_id += 1

    return run_ids, next_run_id

def run_global_feat_ablation_experiments(models, forecasting_hours_to_test):
    for model in models:
        for forecasting_hours in forecasting_hours_to_test:
            for remove_global_features in [True, False]:
                hyperparameters = {"model_architecture": model}
                hyperparameters.update(OPTIMAL_FEATURES[forecasting_hours])
                hyperparameters.update(MODEL_ARCHS[model])
                if remove_global_features:
                    hyperparameters.update({"global_features": []})

                general_config = {"forecasting_hours": forecasting_hours}

                run_training_with_config(hyperparameters, general_config)




if __name__ == "__main__":
    # wandb.login()

    # features_to_test = [
    #     ["air_temperature"],
    #     ["relative_humidity"],
    #     ["air_pressure"],
    #     ["dew_point"],
    # ]

    # global_features_to_test = [
    #     ["sin_1y", "cos_1y"],
    #     ["sin_1d", "cos_1d"],
    # ]

    # baseline_station_features = ["wind_speed", "wind_direction_sin", "wind_direction_cos"]

    # run_feature_selection(baseline_station_features, features_to_test, global_features_to_test)

    run_global_feat_ablation_experiments(["gcn_lstm"], [1, 4, 8])
