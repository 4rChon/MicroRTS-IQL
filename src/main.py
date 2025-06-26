# modified from source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import wandb
import yaml

from iql.train_config import TrainConfig
from iql.train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--note",
        type=str,
        default="",
        help="Note for the experiment",
        required=True
    )

    parser.add_argument(
        "--config",
        type=str,
        default="../config/hyperparams_iql.yaml",
        help="Path to hyperparams config file"
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)

    config = TrainConfig.from_dict(config_dict)
    config.note = args.note

    time = datetime.now().strftime("%H_%M_%S")
    date = datetime.now().strftime("%Y_%m_%d")
    config.environment.name = f"{config.environment.name}_{time}"
    prefix = f"{config.environment.save_dir}/{config.environment.group}"

    experiment_dir = Path(
        f"{prefix}/{date}/{config.environment.name}"
    )

    wandb.init(
        config=asdict(config),
        project=config.environment.project,
        group=config.environment.group,
        notes=config.note,
        dir=experiment_dir,
        name=config.environment.name,
    )

    wandb.save("hyperparams.yaml")

    train(config, experiment_dir)

    wandb.finish()
