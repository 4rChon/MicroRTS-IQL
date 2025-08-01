import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pyrallis
import torch
import wandb
import wandb.util

from iql_microrts.train_config import TrainConfig

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
    id = wandb.util.generate_id()
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
        run.config.update(asdict(config), allow_val_change=True)

        wandb.save(os.path.join(experiment_dir, "*.pt"))

        if config.eval.eval:
            from iql_microrts.eval import eval
            start_seed = config.seed
            print(f"Evaluating IQL with seeds from {start_seed} to "
                  f"{start_seed + config.eval.seed_count - 1}")
            for seed in range(start_seed, start_seed + config.eval.seed_count):
                config.seed = seed
                eval(config, experiment_dir)
        else:
            from iql_microrts.train import train
            train(config, experiment_dir)


if __name__ == "__main__":
    main()  # type: ignore
