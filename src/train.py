import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch as GraphBatch
from torch_geometric.nn import summary as graph_summary
from torchinfo import summary

import wandb
from data.datasets import get_data_loaders
from models.models import PersistenceModel
from utils.evaluation_utils import inverse_scale_batch_tensor
from utils.model_utils import (
    create_model,
    load_general_config,
    load_hyperparameters,
    save_hyperparameters,
    is_model_spatial,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_run_directory(hyperparameters, general_config):
    base_run_directory = Path(f"runs/{general_config["forecasting_hours"]}_hour_forecasting")
    base_run_directory.mkdir(exist_ok=True)

    existing_run_ids = [int(run.name) for run in base_run_directory.iterdir() if run.name.isdigit()]
    next_id = max(existing_run_ids) + 1 if existing_run_ids else 0

    run_directory = base_run_directory / str(next_id)
    run_directory.mkdir(parents=True, exist_ok=True)

    save_hyperparameters(hyperparameters, run_directory)

    return run_directory


def evaluate_model_during_train(model, data_loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in data_loader:
            predictions, y = make_predictions(batch, model, device)

            loss = criterion(predictions, y)
            losses.append(loss.item())

    return sum(losses) / len(losses)


def evaluate_final_model(model, data_loader, device, resolution):
    model.eval()

    mse = torch.nn.MSELoss()
    mae = torch.nn.L1Loss()

    mse_losses = []
    mae_losses = []

    with torch.no_grad():
        for batch in data_loader:
            predictions, y = make_predictions(batch, model, device)

            y = inverse_scale_batch_tensor(y, resolution)
            predictions = inverse_scale_batch_tensor(predictions, resolution)

            mse_losses.append(mse(predictions, y).item())
            mae_losses.append(mae(predictions, y).item())

    total_mse = sum(mse_losses) / len(mse_losses)
    total_mae = sum(mae_losses) / len(mae_losses)

    return total_mse, total_mae


def make_predictions(batch, model, device):
    if isinstance(batch, GraphBatch):
        batch.to(device)
        predictions = model(batch)
        y = batch.y
    else:
        x_station, x_global, y = batch

        x_global = x_global.to(device)
        x_station = x_station.to(device)
        y = y.to(device)

        predictions = model(x_station, x_global)

    return predictions, y


def log_metrics(use_wandb, metrics, epoch):
    if use_wandb:
        wandb.log(metrics, step=epoch)

    log_str = " - ".join(
        [
            f"{metric}: {value:.4f}" if isinstance(value, float) else f"{metric}: {value}"
            for metric, value in metrics.items()
        ]
    )
    print(f"Epoch-{epoch}: {log_str}")


def save_model(model, run_directory, use_wandb, run_id):
    model_path = run_directory / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    print(f"Best model saved to {model_path}")

    if use_wandb:
        artifact = wandb.Artifact(f"best_model_run_{run_id}", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)


def perform_training_loop_spatial(
    model, train_loader, optimizer, criterion, val_loader, general_config
):
    device = general_config["device"]
    use_wandb = general_config["use_wandb"]

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    stopped_epoch = None
    best_epoch = None

    print("Starting training loop...")
    for epoch in range(hyperparameters["max_epochs"]):
        model.train()
        epoch_losses = []

        for batch in train_loader:
            optimizer.zero_grad()
            loss = criterion(*make_predictions(batch, model, device))
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_val_loss = evaluate_model_during_train(
            model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        metrics = {
            "train_loss": mean_train_loss,
            "val_loss": mean_val_loss,
        }
        log_metrics(use_wandb, metrics, epoch)

        # Early stopping check
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0

        else:
            patience_counter += 1

        if patience_counter >= hyperparameters["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    return best_model_state, best_epoch, stopped_epoch


def print_model_information(model, data_loader, device):
    print("General information on the model:")

    sample_batch = next(iter(data_loader))
    if isinstance(sample_batch, GraphBatch):
        sample_batch_on_device = sample_batch.to(device)
        print(graph_summary(model, data=sample_batch_on_device))
    else:
        sample_batch_on_device = tuple(x.to(device) for x in sample_batch[:2])
        summary(model, input_data=sample_batch_on_device)

    print(model)


def init_wandb(hyperparameters, general_config, run_id):
    if general_config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=general_config["wandb_project"],
            name=f"{general_config["forecasting_hours"]}_hour_forecasting_run_{run_id}",
            config=hyperparameters,
        )


def log_run_summary_errors(use_wandb, mse, rmse, mae):
    print(
        "Best model has following errors on evaluation set with inverse scaling (in original domain):"
    )
    print(f"MSE: {mse}")
    print(f"RMSE: {rmse}")
    print(f"MAE: {mae}")

    if use_wandb:
        wandb.run.summary["best_model_val_mse_original_domain"] = mse
        wandb.run.summary["best_model_val_rmse_original_domain"] = rmse
        wandb.run.summary["best_model_val_mae_original_domain"] = mae


def log_run_summary_epochs(use_wandb, stopped_epoch, best_epoch):
    if use_wandb:
        wandb.run.summary["stopped_epoch"] = stopped_epoch
        wandb.run.summary["best_epoch"] = best_epoch


def log_training_start(run_id, general_config, hyperparameters):
    print(f"Starting run number {run_id}.")
    print(f"Used general configuration: {general_config}")
    print(f"Used hyperparameters: {hyperparameters}")


def execute_training_pipeline(general_config, hyperparameters):
    run_directory = init_run_directory(hyperparameters, general_config)
    run_id = run_directory.name

    use_wandb = general_config["use_wandb"]
    init_wandb(hyperparameters, general_config, run_id)
    log_training_start(run_id, general_config, hyperparameters)

    device = general_config["device"]
    model = create_model(hyperparameters, general_config["forecasting_hours"]).to(device)

    print("Loading the dataset...")
    train_loader, val_loader = get_data_loaders(
        look_back_hours=hyperparameters["look_back_hours"],
        resolution=hyperparameters["resolution"],
        station_ids=hyperparameters["station_ids"],
        station_features=hyperparameters["station_features"],
        global_features=hyperparameters["global_features"],
        forecasting_horizon_hours=general_config["forecasting_hours"],
        batch_size=hyperparameters["batch_size"],
        splits=["train", "eval"],
        num_workers=general_config["num_workers"],
        is_spatial=is_model_spatial(model),
        weighting=hyperparameters.get("weighting"),
        knns=hyperparameters.get("knns")
    )
    print_model_information(model, train_loader, device)

    if use_wandb:
        wandb.watch(model, log="all")

    criterion = torch.nn.MSELoss()

    # Persistence model does not need training
    if not isinstance(model, PersistenceModel):
        Optimizer = getattr(torch.optim, hyperparameters["optimizer"])
        optimizer = Optimizer(model.parameters(), lr=hyperparameters["learning_rate"])

        best_model_state, best_epoch, stopped_epoch = perform_training_loop_spatial(
            model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            val_loader=val_loader,
            general_config=general_config,
        )

        log_run_summary_epochs(use_wandb, stopped_epoch, best_epoch)
        model.load_state_dict(best_model_state)

    mse, mae = evaluate_final_model(model, val_loader, device, hyperparameters["resolution"])
    rmse = np.sqrt(mse)
    log_run_summary_errors(use_wandb, mse, rmse, mae)
    save_model(model, run_directory, use_wandb, run_id)

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    general_config = load_general_config()
    hyperparameters = load_hyperparameters(forecasting_hours=general_config["forecasting_hours"])
    execute_training_pipeline(general_config=general_config, hyperparameters=hyperparameters)
