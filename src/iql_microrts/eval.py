from pathlib import Path
from typing import Any

import gym_microrts
import gym_microrts.envs
import gym_microrts.envs.vec_env
import gym_microrts.microrts_ai
import numpy as np
import torch

from env_utils import get_env_spec, make_eval_env, sample_maps
from iql_microrts.iql_model import ActorPolicy, IQLNetwork
from iql_microrts.train_config import TrainConfig
from utils import set_seed_everywhere

TensorBatch = list[torch.Tensor]

EXP_ADV_MAX = 100.0


@torch.no_grad()
def eval_actor(
    env: gym_microrts.envs.vec_env.MicroRTSGridModeVecEnv,
    actor: ActorPolicy,
    device: torch.device,
    episodes_num: int,
    render: bool = False,
) -> np.ndarray[Any, np.dtype[np.float32]]:
    actor.eval()
    episode_rewards = []
    for _ in range(episodes_num):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            if render:
                env.render()
            action_mask = env.get_action_mask()
            action_mask = torch.tensor(
                action_mask, dtype=torch.float32, device=device
            )
            state = torch.tensor(state, dtype=torch.float32, device=device)
            action = actor.act(state, action_mask).detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append((episode_reward + 1) / 2)
        print(f"Total rewards: {sum(episode_rewards)}/{len(episode_rewards)}")

    actor.train()
    return np.asarray(episode_rewards)


def eval(config: TrainConfig, save_path: Path):
    seed = config.seed
    set_seed_everywhere(seed)
    state_dim, action_dim, _ = get_env_spec(
        config.environment.episode_steps_max
    )

    iql_network = IQLNetwork(
        state_dim, action_dim, config.iql.model, config.device,
    )

    actor = iql_network.actor

    print("Loading IQL model from", config.iql.model.load_path)
    state_dict = torch.load(
        config.iql.model.load_path, map_location=config.device
    )
    actor.load_state_dict(state_dict["actor"])
    actor.eval()

    print("---------------------------------------")
    print(f"Evaluating IQL, Env: Gym-MicroRTS, Seed: {seed}")
    print("---------------------------------------")

    print("---- Actor Evaluation Start ----")
    enemies = [
        gym_microrts.microrts_ai.coacAI,
        gym_microrts.microrts_ai.mayari,
        gym_microrts.microrts_ai.workerRushAI,
        gym_microrts.microrts_ai.lightRushAI
    ]

    rollouts = config.eval.rollouts_num
    max_steps = config.environment.episode_steps_max

    maps = sample_maps(config.eval.rollouts_num, "eval")

    win_rates = {}
    for ai2 in enemies:
        win_rates[ai2.__name__] = 0
        eval_envs, _, _ = make_eval_env(
            max_steps, 1, maps, 100 + config.seed, [ai2]
        )

        rewards = eval_actor(
            eval_envs, actor, config.device, rollouts,
            config.render
        ).mean()
        win_rates[ai2.__name__] = rewards.sum()
        print(win_rates)
    print("--- Actor Evaluation Complete ---")

    with open(f"{save_path}/results_{seed}.txt", "w") as f:
        f.write(str(win_rates))
        print(f"Results saved to {save_path}/results_{seed}.txt")
