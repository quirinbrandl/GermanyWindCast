import copy
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torchinfo import summary

import wandb
from data.datasets import get_data_loaders
from models.models import PersistenceModel
from utils.model_utils import (
    create_model,
    load_hyperparameters,
    save_hyperparameters,
    load_general_config,
)
from utils.evaluation_utils import inverse_scale_batch_tensor


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
        for x_station, x_global, y in data_loader:
            y = y.to(device)

            x_station = x_station.to(device)
            x_global = x_global.to(device)
            predictions = model(x_station, x_global)

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
        for x_station, x_global, y in data_loader:
            y = inverse_scale_batch_tensor(y.to(device), resolution)

            x_station = x_station.to(device)
            x_global = x_global.to(device)
            predictions = inverse_scale_batch_tensor(model(x_station, x_global), resolution)

            mse_losses.append(mse(predictions, y).item())
            mae_losses.append(mae(predictions, y).item())

    total_mse = sum(mse_losses) / len(mse_losses)
    total_mae = sum(mae_losses) / len(mae_losses)

    return total_mse, total_mae


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


def save_model(model, run_directory):
    model_path = run_directory / "best_model.pt"
    torch.save(model.state_dict(), model_path)
    return model_path


def perform_training_loop(model, train_loader, optimizer, criterion, val_loader, general_config):
    device = general_config["device"]
    use_wandb = general_config["use_wandb"]

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    stopped_epoch = None
    best_epoch = None

    print("Starting training loop...")
    for epoch in range(hyperparameters["max_epochs"]):

        # Training phase
        model.train()
        epoch_losses = []

        for x_station, x_global, y in train_loader:
            x_station = x_station.to(device)
            x_global = x_global.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            predictions = model(x_station, x_global)
            loss = criterion(predictions, y)
            loss.backward()
            optimizer.step()

            epoch_losses.append(loss.item())

        # Evaluation phase
        mean_train_loss = sum(epoch_losses) / len(epoch_losses)
        mean_val_loss = evaluate_model_during_train(
            model,
            data_loader=val_loader,
            criterion=criterion,
            device=device,
        )

        # Log metrics
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


def execute_training_pipeline(general_config, hyperparameters):
    run_directory = init_run_directory(hyperparameters, general_config)
    run_id = run_directory.name

    use_wandb = general_config["use_wandb"]
    init_wandb(hyperparameters, general_config, run_id)

    print(f"Starting run number {run_id}.")
    print(f"Used general configuration: {general_config}")
    print(f"Used hyperparameters: {hyperparameters}")

    # Load datasets
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
    )

    # Create model
    device = general_config["device"]
    model = create_model(hyperparameters, general_config["forecasting_hours"]).to(device)

    if use_wandb:
        wandb.watch(model, log="all")

    print_model_information(model, train_loader, device)

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()

    # Persistence model does not need training
    if not isinstance(model, PersistenceModel):
        Optimizer = getattr(torch.optim, hyperparameters["optimizer"])
        optimizer = Optimizer(model.parameters(), lr=hyperparameters["learning_rate"])

        best_model_state, best_epoch, stopped_epoch = perform_training_loop(
            model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            val_loader=val_loader,
            general_config=general_config,
        )

        if use_wandb:
            wandb.run.summary["stopped_epoch"] = stopped_epoch
            wandb.run.summary["best_epoch"] = best_epoch

        model.load_state_dict(best_model_state)

    # Evaluate best model and log results (on original domain for better interpretability -> inverse scaling)
    mse, mae = evaluate_final_model(model, val_loader, device, hyperparameters["resolution"])
    rmse = np.sqrt(mse)

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

    # Save the best model to file system
    model_path = save_model(model, run_directory)
    print(f"Best model saved to {model_path}")

    # Save the best model to wandb
    if use_wandb:
        artifact = wandb.Artifact(f"best_model_run_{run_id}", type="model")
        artifact.add_file(str(model_path))
        wandb.log_artifact(artifact)

    # Finish WandB logging
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    general_config = load_general_config()
    hyperparameters = load_hyperparameters(forecasting_hours=general_config["forecasting_hours"])
    execute_training_pipeline(general_config=general_config, hyperparameters=hyperparameters)
