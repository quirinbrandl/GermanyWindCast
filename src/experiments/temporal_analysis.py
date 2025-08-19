import wandb
from utils.experiment_utils import run_training_with_config
from experiments.config import OPTIMAL_FEATURES, MODEL_ARCHS


def run_temporal_analysis(
    resolutions_to_test, look_back_hours_to_test, models_to_test, forecasting_hours_to_test
):
    for forecasting_hours in forecasting_hours_to_test:
        for model in models_to_test:
            for res in resolutions_to_test:
                for look_back_hours in look_back_hours_to_test:
                    hyperparameters = {
                        "look_back_hours": look_back_hours,
                        "resolution": res,
                        "model_architecture": model,
                    }
                    hyperparameters.update(OPTIMAL_FEATURES[forecasting_hours])
                    hyperparameters.update(MODEL_ARCHS[model])

                    general_config = {"forecasting_hours": forecasting_hours}

                    run_training_with_config(hyperparameters, general_config)


if __name__ == "__main__":
    wandb.login()

    models_to_test = ["gcn_lstm"]
    resolutions_to_test = ["10min"]
    look_back_hours_to_test = [11]
    forecasting_hours_to_test = [1]

    run_temporal_analysis(
        resolutions_to_test, look_back_hours_to_test, models_to_test, forecasting_hours_to_test
    )
