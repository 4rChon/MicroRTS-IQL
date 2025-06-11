from dataclasses import dataclass, field
from typing import Any

@dataclass
class EvalConfig:
    eval: bool = True
    envs_num: int = 8
    rollouts: int = 200

@dataclass
class EnvConfig:
    save_dir: str = "_exp"
    project: str = "PhD"
    group: str = "iql"
    name: str = "microrts"
    max_steps: int = 2000
    self_play: bool = False

@dataclass
class DataConfig:
    buffer_path: str = "data/8x8/1v1/replay_buffer_3k"
    save_interval: int = 10000
    log_interval: int = 10

@dataclass
class IQLModelConfig:
    hidden_dim: int = 256
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
    max_timesteps: int = 1e6
    vf_lr: float = 3e-4
    qf_lr: float = 3e-4
    actor_lr: float = 3e-4
    batch_size: int = 64
    buffer_size: int = 3000
    reward_scale: float = 1.0

@dataclass
class IQLEvalConfig:
    num_episodes: int = 10
    num_longer_episodes: int = 100
    eval_freq: int = 1000

@dataclass
class IQLConfig:
    model: IQLModelConfig = field(default_factory=IQLModelConfig)
    training: IQLTrainingConfig = field(default_factory=IQLTrainingConfig)
    eval: IQLEvalConfig = field(default_factory=IQLEvalConfig)

@dataclass
class TrainConfig:
    seed: int = 42
    debug: bool = False
    device: str = "cuda"
    eval: EvalConfig = field(default_factory=EvalConfig)
    environment: EnvConfig = field(default_factory=EnvConfig)
    data: DataConfig = field(default_factory=DataConfig)
    iql: IQLConfig = field(default_factory=IQLConfig)

    @classmethod
    def from_dict(cls, config_dict: dict[str: Any]) -> 'TrainConfig':
        config_dict['eval'] = EvalConfig(**config_dict['eval'])
        config_dict['environment'] = EnvConfig(**config_dict['environment'])
        config_dict['data'] = DataConfig(**config_dict['data'])
        iql_dict = config_dict['iql']
        iql_dict['model'] = IQLModelConfig(**iql_dict['model'])
        iql_dict['training'] = IQLTrainingConfig(**iql_dict['training'])
        iql_dict['eval'] = IQLEvalConfig(**iql_dict['eval'])
        config_dict['iql'] = IQLConfig(**iql_dict)

        return cls(**config_dict)