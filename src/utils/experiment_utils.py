import subprocess
import sys
import yaml

from utils.evaluation_utils import load_general_config, load_hyperparameters
from utils.model_utils import save_hyperparameters 


def run_training_with_config(hyperparameters, general_config):
    file_hyperparameters = load_hyperparameters()
    file_general_config = load_general_config()

    file_hyperparameters.update(hyperparameters)
    file_general_config.update(general_config)

    save_hyperparameters(file_hyperparameters)
    save_general_config(file_general_config)

    subprocess.run([sys.executable, "src/train.py"])

def save_general_config(general_config):
    with open("config/general_config.yaml", "w") as f:
        yaml.dump(general_config, f)
