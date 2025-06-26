import os
from pathlib import Path
import random
from typing import Callable

import numpy as np
from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

BASE_DIR = Path("./lib/gym_microrts/gym_microrts")
EVAL_MAPS_DIR = Path(f"{BASE_DIR}/microrts/maps/eval_maps")
ONLINE_TRAIN_MAPS_DIR = Path(f"{BASE_DIR}/microrts/maps/online_train_maps")
OFFLINE_TRAIN_MAPS_DIR = Path(f"{BASE_DIR}/microrts/maps/offline_train_maps")
maps = {
    "eval": [
        f"maps/eval_maps/{map}" for map in
        os.listdir(EVAL_MAPS_DIR)
    ],
    "online": [
        f"maps/online_train_maps/{map}" for map in
        os.listdir(ONLINE_TRAIN_MAPS_DIR)
    ],
    "offline": [
        f"maps/offline_train_maps/{map}" for map in
        os.listdir(OFFLINE_TRAIN_MAPS_DIR)
    ]
}

DEFAULT_AIS = [
    microrts_ai.coacAI,
    microrts_ai.workerRushAI,
    microrts_ai.lightRushAI,
    microrts_ai.mayari,
    microrts_ai.coacAI,
    microrts_ai.coacAI,
    microrts_ai.mayari,
    microrts_ai.mayari
]


def make_env(
    max_steps: int,
    num_envs: int,
    seed: int,
    map_set: str,
    ais: list[Callable] | None = None
) -> tuple[MicroRTSGridModeVecEnv, list[Callable], list[str]]:
    """
    Create a MicroRTS environment with specified parameters.
    Args:
        max_steps (int): Maximum number of steps per episode.
        num_envs (int): Number of environments to create.
        seed (int): Random seed for reproducibility.
        map_set (str): The set of maps to use. Options are "eval", "online",
                       or "offline".
        ais (list[Callable] | None): List of AI functions to use. If None,
                                     defaults to a predefined set of AIs.
    Returns:
        tuple[MicroRTSGridModeVecEnv, list[Callable], list[str]]:
            A tuple containing the created environment, the list of AIs used,
            and the list of map paths.
    """
    if ais is None:
        ais = DEFAULT_AIS
    ai2s = random.sample(ais, num_envs)
    map_paths = random.sample(maps[map_set], num_envs)
    print("Creating new environment with the following maps and ais:")
    print("Maps:")
    [print(f"\t{map}") for map in map_paths]
    print("AIs:")
    [print(f"\t{ai2.__name__}") for ai2 in ai2s]
    print("\n\n")
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=num_envs,
        max_steps=max_steps,
        ai2s=ai2s,
        map_paths=map_paths,
        cycle_maps=map_paths,
        # reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        reward_weight=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        autobuild=False,
    )
    env.reset()
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, ai2s, map_paths


def sample_maps(num_maps: int, map_set: str = "eval") -> list[str]:
    """
    Sample a specified number of maps from the given map set.
    Args:
        num_maps (int): The number of maps to sample.
        map_set (str): The set of maps to sample from. Options are "eval",
                       "online", or "offline".
    Returns:
        list[str]: A list of sampled map paths.
    """
    return random.sample(maps[map_set], num_maps)


def make_eval_env(
    max_steps: int,
    num_envs: int,
    map_paths: list[str],
    seed: int = 0,
    ais: list[Callable] | None = None
) -> tuple[MicroRTSGridModeVecEnv, list[Callable], list[str]]:
    """
    Create a MicroRTS evaluation environment with specified parameters.
    Args:
        max_steps (int): Maximum number of steps per episode.
        num_envs (int): Number of environments to create.
        map_paths (list[str]): List of map paths to use for the environments.
        seed (int): Random seed for reproducibility.
        ais (list[Callable] | None): List of AI functions to use. If None,
                                     defaults to a predefined set of AIs.
    Returns:
        tuple[MicroRTSGridModeVecEnv, list[Callable], list[str]]:
            A tuple containing the created environment, the list of AIs used,
            and the list of map paths.
    """
    if ais is None:
        ais = DEFAULT_AIS
    ai2s = random.sample(ais, num_envs)
    print("Creating new environment with the following maps and ais:")
    print("Maps:")
    [print(f"\t{map}") for map in map_paths]
    print("AIs:")
    [print(f"\t{ai2.__name__}") for ai2 in ai2s]
    print("\n\n")
    env = MicroRTSGridModeVecEnv(
        num_selfplay_envs=0,
        num_bot_envs=num_envs,
        max_steps=max_steps,
        ai2s=ai2s,
        map_paths=map_paths[:num_envs],
        cycle_maps=map_paths,
        # reward_weight=np.array([10.0, 1.0, 1.0, 0.2, 1.0, 4.0]),
        reward_weight=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        autobuild=False,
    )
    env.reset()
    env.action_space.seed(seed)
    env.observation_space.seed(seed)

    return env, ai2s, map_paths


def get_env_spec(max_steps: int) -> tuple[
    tuple[int, int, int], list[int], MicroRTSGridModeVecEnv
]:
    env, _, _ = make_env(max_steps, 1, 0, "eval")
    h, w, c = env.observation_space.shape
    state_dim = (h, w, c)
    action_dim = env.action_plane_space.nvec.tolist()

    print("\n\n------------------------")
    print("Env Spec:")
    print("\tState Dim:", state_dim)
    print("\tAction Dim:", action_dim)
    print("------------------------\n\n")

    return state_dim, action_dim, env
