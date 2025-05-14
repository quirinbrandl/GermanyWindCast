import subprocess
import sys

from utils.model_utils import load_hyperparameters, save_hyperparameters


def run_training_with_features(station_features, global_features):
    hyperparameters = load_hyperparameters()
    hyperparameters["station_features"] = station_features
    hyperparameters["global_features"] = global_features

    save_hyperparameters(hyperparameters)

    subprocess.run([sys.executable, "src/train.py"])


if __name__ == "__main__":

    base_station_features = ["wind_speed"]
    base_global_features = []

    station_options_to_test = [
        ["dew_point"],
        ["air_pressure"],
        ["air_temperature"],
        ["wind_direction_sin", "wind_direction_cos"],
        ["relative_humidity"],
    ]
    global_features_to_test = [
        ["dew_point_03897"],
        ["air_pressure_03897"],
        ["air_temperature_03897"],
        ["wind_direction_sin_03897", "wind_direction_cos_03897"],
        ["relative_humidity_03897"],
        ["day_sin", "day_cos"],
        ["hour_cos", "hour_sin"],
    ]

    for feat in station_options_to_test:
        station_feats = base_station_features + feat
        run_training_with_features(station_feats, base_global_features)

    for feat in global_features_to_test:
        global_feats = feat + base_global_features
        run_training_with_features(base_station_features, global_feats)
