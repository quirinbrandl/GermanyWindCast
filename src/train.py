import torch
from models.models import PersistenceModel
from utils.model_utils import load_hyperparameters, create_model
import wandb
from data.datasets import get_data_loaders
from torchinfo import summary
from pathlib import Path
import yaml
import copy
import random
import numpy as np
import joblib


def load_general_config():
    general_config_path = Path("config/general_config.yaml")
    with open(general_config_path / "", "r") as f:
        general_config = yaml.safe_load(f)

    return general_config


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_run_directory(hyperparameters):
    base_run_directory = Path("runs")

    existing_run_ids = [int(run.name) for run in base_run_directory.iterdir() if run.name.isdigit()]
    next_id = max(existing_run_ids) + 1 if existing_run_ids else 0

    run_directory = base_run_directory / str(next_id)
    run_directory.mkdir(parents=True, exist_ok=True)

    with open(run_directory / "hyperparameters.yaml", "w") as f:
        yaml.dump(hyperparameters, f)

    return run_directory


def inverseScaling(y_batch, resolution):
    scalers_dict = joblib.load(Path(f"data/processed/{resolution}res/scalers.pkl"))
    scaler = scalers_dict["wind_speed"]

    device = y_batch.device
    y_batch_np = (
        y_batch.detach().cpu().numpy()
    )  # scikit learn's scaler works only with np.array on cpu

    inverse_scaled = np.zeros_like(y_batch)
    inverse_scaled = scaler.inverse_transform(y_batch_np.reshape(-1, 1)).reshape(y_batch_np.shape)

    return torch.tensor(inverse_scaled, device=device, dtype=y_batch.dtype)


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
            y = inverseScaling(y.to(device), resolution)

            x_station = x_station.to(device)
            x_global = x_global.to(device)
            predictions = inverseScaling(model(x_station, x_global), resolution)

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


def perform_training_loop(
    model, train_loader, optimizer, criterion, val_loader, general_config):
    device = general_config["device"]
    use_wandb = general_config["use_wandb"]

    best_val_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    stopped_epoch = None
    best_epoch = None

    for epoch in range(general_config["max_epochs"]):
        print("Starting training loop...")

        # Training phase
        model.train()
        epoch_losses = []

        for x_station, x_global, y in train_loader:
            x_station = x_station.to()
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

        if patience_counter >= general_config["early_stopping_patience"]:
            print(f"Early stopping triggered after {epoch} epochs")
            break

    return best_model_state, best_epoch, stopped_epoch


def execute_training_pipeline(general_config, hyperparameters):
    run_directory = init_run_directory(hyperparameters)
    run_id = run_directory.name

    print(f"Starting run number {run_id}.")
    print(f"Used general configuration: {general_config}")
    print(f"Used hyperparameters: {hyperparameters}")

    # Initialize WandB
    use_wandb = general_config["use_wandb"]
    forecasting_hours = general_config["forecasting_hours"]
    if use_wandb:
        wandb.login()
        wandb.init(
            project=general_config["wandb_project"],
            name=f"run_{run_id}",
            config=hyperparameters,
            group=f"{forecasting_hours}_forecasting_hours",
        )

    # Load datasets
    print("Loading the dataset...")
    train_loader, val_loader = get_data_loaders(
        look_back_hours=hyperparameters["look_back_hours"],
        resolution=hyperparameters["resolution"],
        station_ids=hyperparameters["station_ids"],
        station_features=hyperparameters["station_features"],
        global_features=hyperparameters["global_features"],
        forecasting_horizon_hours=general_config["forecasting_hours"],
        batch_size=general_config["batch_size"],
        splits=["train", "eval"],
    )

    # Create model
    device = general_config["device"]
    model = create_model(hyperparameters, general_config).to(device)

    if use_wandb:
        wandb.watch(model, log="all")

    print("General information on the model:")
    summary(model, input_data=next(iter(train_loader))[:2])

    # Define loss and optimizer
    criterion = torch.nn.MSELoss()
    Optimizer = getattr(torch.optim, general_config["optimizer"])
    optimizer = Optimizer(model.parameters(), lr=general_config["learning_rate"])

    if not isinstance(model, PersistenceModel):
        best_model_state, best_epoch, stopped_epoch = perform_training_loop(
            model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            val_loader=val_loader,
            general_config=general_config
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

    # Save the best model
    model_path = save_model(model, run_directory)
    print(f"Best model saved to {model_path}")

    if use_wandb:
        wandb.save(str(model_path))

    # Finish WandB logging
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    set_seed(42)
    hyperparameters = load_hyperparameters()
    general_config = load_general_config()
    execute_training_pipeline(general_config=general_config, hyperparameters=hyperparameters)
