from dataclasses import asdict
from datetime import datetime
import os
from pathlib import Path

import pyrallis
import torch

import wandb
from iql.train import train
from iql.train_config import TrainConfig

pyrallis.decode.register(  # type: ignore
    torch.device,
    lambda x: torch.device(x) if x else torch.device("cuda")
)


@pyrallis.wrap()  # type: ignore
def main(config: TrainConfig):
    if not config.note:
        raise ValueError("Please provide a note for the experiment.")

    time = datetime.now().strftime("%H_%M_%S")
    date = datetime.now().strftime("%Y_%m_%d")
    id = wandb.util.generate_id()  # type: ignore
    experiment_dir = Path(
        f"{config.environment.save_dir}/{config.environment.group}/{date}/{id}"
    )

    with wandb.init(
        project=config.environment.project,
        group=config.environment.group,
        notes=config.note,
        dir=experiment_dir,
        id=id,
    ) as run:
        run.name = f"{config.environment.group}-{run.name} [{date} {time}]"
        config.environment.name = run.name  # type: ignore
        run.config.update(asdict(config), allow_val_change=True)

        wandb.save(os.path.join(experiment_dir, "*.pt"))

        train(config, experiment_dir)


if __name__ == "__main__":
    main()  # type: ignore
