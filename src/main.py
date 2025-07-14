import argparse
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyrallis
import torch
import wandb

from iql.train import train
from iql.train_config import TrainConfig

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
        help="Path to the hyperparameters YAML file"
    )

    args = parser.parse_args()
    pyrallis.decode.register(  # type: ignore
        torch.device,
        lambda x: torch.device(x) if x else torch.device("cuda")
    )
    config = pyrallis.parse(  # type: ignore
        config_class=TrainConfig, config_path=args.config
    )

    time = datetime.now().strftime("%H_%M_%S")
    date = datetime.now().strftime("%Y_%m_%d")
    config.environment.name = f"{config.environment.name}_{time}"
    prefix = Path(f"{config.environment.save_dir}/{config.environment.group}")

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
