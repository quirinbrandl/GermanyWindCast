from models.models import PersistenceModel, MLP
from pathlib import Path
import yaml
import torch


def load_general_config():
    general_config_path = Path("config/general_config.yaml")
    with open(general_config_path, "r") as f:
        general_config = yaml.safe_load(f)

    return general_config


def load_hyperparameters(run_id=None, forecasting_hours=None):
    config_dir = (
        Path("config")
        if run_id is None
        else Path(f"runs/{forecasting_hours}_hour_forecasting/{run_id}")
    )
    with open(config_dir / "hyperparameters.yaml", "r") as f:
        hyperparameters = yaml.safe_load(f)

    return hyperparameters


def save_hyperparameters(hyperparameters, output_dir=None):
    dir = Path(output_dir) if output_dir else Path("config")
    with open(dir / "hyperparameters.yaml", "w") as f:
        yaml.dump(hyperparameters, f)


def create_model(hyperparameters, forecasting_horizon_hours):
    model_architecture = hyperparameters["model_architecture"]

    if model_architecture == "persistence":
        return PersistenceModel(forecasting_horizon_hours, 0, 0)
    elif model_architecture == "mlp":
        return MLP(
            look_back_hours=hyperparameters["look_back_hours"],
            forecasting_horizon=forecasting_horizon_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            num_hidden_layers=hyperparameters["num_hidden_layers"],
            hidden_size=hyperparameters["hidden_size"],
            dropout_rate=hyperparameters["dropout_rate"],
            resolution=hyperparameters["resolution"],
        )
    else:
        raise ValueError(f"Unknown  architecture: {model_architecture}")


def load_best_model(run_id, forecasting_hours, device):
    model_hyperparameters = load_hyperparameters(run_id, forecasting_hours)

    model_dir = Path(f"runs/{forecasting_hours}_hour_forecasting/{run_id}")
    model = create_model(model_hyperparameters, forecasting_hours)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    return model
