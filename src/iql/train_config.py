from dataclasses import dataclass, field
from pathlib import Path

import torch


@dataclass
class EvalConfig:
    eval: bool = False
    envs_num: int = 8
    rollouts_num: int = 200


@dataclass
class EnvConfig:
    save_dir: str = "_exp"
    project: str = "PhD"
    group: str = "iql"
    name: str = "microrts"
    episode_steps_max: int = 2000


@dataclass
class DataConfig:
    buffer_path: Path = Path("data/8x8/1v1/transitions/train")
    save_interval: int = 10000
    log_interval: int = 10
    num_workers: int = 12


@dataclass
class IQLModelConfig:
    hidden_dim: int = 32
    hidden_layers: int = 2
    dropout: float = 0.1
    load: bool = False
    load_path: str = "model/model.pt"


@dataclass
class IQLTrainingConfig:
    train: bool = True
    tau: float = 0.005
    beta: float = 3.0
    iql_tau: float = 0.7
    discount: float = 0.99
    max_timesteps: int = 1000000
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    batch_size: int = 64
    reward_scale: float = 1.0
    warmup_pct: float = 0.45


@dataclass
class IQLEvalConfig:
    episodes_num: int = 10
    longer_episodes_num: int = 100
    eval_freq: int = 1000


@dataclass
class IQLConfig:
    model: IQLModelConfig = field(default_factory=IQLModelConfig)
    training: IQLTrainingConfig = field(default_factory=IQLTrainingConfig)
    eval: IQLEvalConfig = field(default_factory=IQLEvalConfig)


@dataclass
class TrainConfig:
    note: str = ""
    seed: int = 42
    debug: bool = False
    device: torch.device = torch.device("cuda")
    render: bool = False
    eval: EvalConfig = field(default_factory=EvalConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)
