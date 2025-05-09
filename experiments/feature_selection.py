import subprocess

from utils.model_utils import save_hyperparameters, load_hyperparameters


def run_training_with_features(station_features, global_features):
    hyperparameters = load_hyperparameters()
    hyperparameters["station_features"] = station_features
    hyperparameters["global_features"] = global_features

    save_hyperparameters(hyperparameters)

    subprocess.run(["python", "src/train.py"])


if __name__ == "__main__":

    base_station_features = ["wind_speed", "wind_direction_sin", "wind_direction_cos", "relative_humidity"]
    base_global_features = ["hour_sin", "hour_cos", "day_sin", "day_cos"]

    station_options_to_test = [
        ["dew_point"],
        ["air_pressure"],
        ["air_temperature"],
    ]
    global_features_to_test = []

    for feat in station_options_to_test:
        station_feats = base_station_features + feat
        run_training_with_features(station_feats, base_global_features)

    for feat in global_features_to_test:
        global_feats = feat + base_global_features
        run_training_with_features(base_station_features, global_feats)
