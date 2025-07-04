import pandas as pd
from models.models import PersistenceModel, MLP, WindLSTM, WindGCN, GCNLSTM, GATLSTM
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


def create_model(hyperparameters, forecasting_hours):
    model_architecture = hyperparameters["model_architecture"]

    if model_architecture == "persistence":
        return PersistenceModel(
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
        )
    elif model_architecture == "mlp":
        return MLP(
            look_back_hours=hyperparameters["look_back_hours"],
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            num_hidden_dense_layers=hyperparameters["num_hidden_dense_layers"],
            dense_hidden_size=hyperparameters["dense_hidden_size"],
            dropout_rate=hyperparameters["dropout_rate"],
            resolution=hyperparameters["resolution"],
        )
    elif model_architecture == "lstm":
        return WindLSTM(
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            num_hidden_lstm_layers=hyperparameters["num_hidden_lstm_layers"],
            num_hidden_dense_layers=hyperparameters["num_hidden_dense_layers"],
            lstm_hidden_size=hyperparameters["lstm_hidden_size"],
            dense_hidden_size=hyperparameters["dense_hidden_size"],
            dropout_rate=hyperparameters["dropout_rate"],
        )
    elif model_architecture == "gcn":
        return WindGCN(
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            hidden_channels=hyperparameters["hidden_channels"],
            dropout_rate=hyperparameters["dropout_rate"],
            resolution=hyperparameters["resolution"],
            look_back_hours=hyperparameters["look_back_hours"],
            use_residual=hyperparameters["use_residual"],
            dense_hidden_size=hyperparameters["dense_hidden_size"],
            num_hidden_gcn_layers=hyperparameters["num_hidden_gnn_layers"],
            num_hidden_dense_layers=hyperparameters["num_hidden_dense_layers"],
        )
    elif model_architecture == "gcn_lstm":
        return GCNLSTM(
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            hidden_channels=hyperparameters["hidden_channels"],
            dropout_rate=hyperparameters["dropout_rate"],
            resolution=hyperparameters["resolution"],
            look_back_hours=hyperparameters["look_back_hours"],
            use_residual=hyperparameters["use_residual"],
            num_hidden_lstm_layers=hyperparameters["num_hidden_lstm_layers"],
            lstm_hidden_size=hyperparameters["lstm_hidden_size"],
            num_hidden_gcn_layers=hyperparameters["num_hidden_gnn_layers"],
            num_hidden_dense_layers=hyperparameters["num_hidden_dense_layers"],
            dense_hidden_size=hyperparameters["dense_hidden_size"],
        )
    elif model_architecture == "gat_lstm":
        return GATLSTM(
            forecasting_hours=forecasting_hours,
            station_ids=hyperparameters["station_ids"],
            station_features=hyperparameters["station_features"],
            global_features=hyperparameters["global_features"],
            hidden_channels=hyperparameters["hidden_channels"],
            dropout_rate=hyperparameters["dropout_rate"],
            resolution=hyperparameters["resolution"],
            look_back_hours=hyperparameters["look_back_hours"],
            use_residual=hyperparameters["use_residual"],
            num_hidden_lstm_layers=hyperparameters["num_hidden_lstm_layers"],
            lstm_hidden_size=hyperparameters["lstm_hidden_size"],
            num_hidden_gat_layers=hyperparameters["num_hidden_gnn_layers"],
            heads=hyperparameters["heads"],
        )
    else:
        raise ValueError(f"Unknown architecture: {model_architecture}")


def is_model_spatial(model):
    return isinstance(model, (WindGCN, GCNLSTM, GCNLSTM))


def load_best_model(run_id, forecasting_hours, device):
    model_hyperparameters = load_hyperparameters(run_id, forecasting_hours)

    model_dir = Path(f"runs/{forecasting_hours}_hour_forecasting/{run_id}")
    model = create_model(model_hyperparameters, forecasting_hours)
    model.load_state_dict(torch.load(model_dir / "best_model.pt", map_location=device))

    return model

def calculate_next_run_id(forecasting_hours):
    base_run_directory = get_base_run_dir(forecasting_hours)
    existing_run_ids = [int(run.name) for run in base_run_directory.iterdir() if run.name.isdigit()]

    return max(existing_run_ids) + 1 if existing_run_ids else 0

def get_base_run_dir(forecasting_hours):
    return Path(f"runs/{forecasting_hours}_hour_forecasting")
