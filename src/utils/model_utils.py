from models.models import PersistenceModel, MLP
from pathlib import Path
import yaml
import torch


def load_hyperparameters(config_dir=Path("config")):
    with open(config_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    return hyperparameters


def create_model(hyperparameters, general_config):
    model_architecture = hyperparameters["model_architecture"]
    forecasting_horizon_hours = general_config["forecasting_hours"]

    if model_architecture == "persistence":
        return PersistenceModel(forecasting_horizon_hours, 0, 0)
    elif model_architecture == "mlp":
        return MLP(
            hyperparameters["look_back_hours"],
            forecasting_horizon_hours,
            hyperparameters["station_ids"],
            hyperparameters["station_features"],
            hyperparameters["global_features"],
            hyperparameters["num_hidden_layers"],
            hyperparameters["hidden_size"],
            hyperparameters["dropout_rate"],
            hyperparameters["resolution"],
        )
    else:
        raise ValueError(f"Unknown  architecture: {model_architecture}")


def load_best_model(run_id, device):
    model_dir = Path(f"runs/{run_id}")
    model_hyperparameters = load_hyperparameters(model_dir)

    model = create_model(model_hyperparameters)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    return model
