import numpy as np
from utils.model_utils import is_model_spatial
from utils.data_utils import find_nearest_stations
from utils.experiment_utils import run_training_with_config
from experiments.config import OPTIMAL_FEATURES, MODEL_ARCHS, OPTIMAL_TEMPORAL_CONFIG

def run_spatial_analysis(numbers_of_stations_to_test, models_to_test, forecasting_hours_to_test):
    for forecasting_hours in forecasting_hours_to_test:
        for model in models_to_test:
            for num_stations in numbers_of_stations_to_test:
                if is_model_spatial(model) and (num_stations == 1):
                    continue
            
                hyperparameters = {
                            "model_architecture": model,
                            "station_ids": find_nearest_stations(num_stations),
                        }
                hyperparameters.update(OPTIMAL_FEATURES[forecasting_hours])
                hyperparameters.update(MODEL_ARCHS[model])
                hyperparameters.update(OPTIMAL_TEMPORAL_CONFIG[(forecasting_hours, model)])

                general_config = {"forecasting_hours": forecasting_hours}

                run_training_with_config(hyperparameters, general_config)


if __name__ == "__main__":
    numbers_of_stations_to_test = [3, 4, 5, 6, 8, 9, 10, 11, 13, 14, 15, 16]
    forecasting_hours_to_test = [1, 4, 8]
    models_to_test = ["mlp", "lstm", "gcn", "gcn_lstm"]
    run_spatial_analysis(numbers_of_stations_to_test, models_to_test, forecasting_hours_to_test)