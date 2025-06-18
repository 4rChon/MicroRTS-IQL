import os
import random
import numpy as np

from gym_microrts import microrts_ai
from gym_microrts.envs.vec_env import MicroRTSGridModeVecEnv

eval_maps_dir = "C:\\Users\\bendb\\Desktop\\Code\\MicroRTS-Py\\gym_microrts\\microrts\\maps\\eval_maps"
online_train_maps_dir = "C:\\Users\\bendb\\Desktop\\Code\\MicroRTS-Py\\gym_microrts\\microrts\\maps\\online_train_maps"
offline_train_maps_dir = "C:\\Users\\bendb\\Desktop\\Code\\MicroRTS-Py\\gym_microrts\\microrts\\maps\\offline_train_maps"
maps = {
    "eval": [f"maps/eval_maps/{map}" for map in os.listdir(eval_maps_dir)],
    "online": [f"maps/online_train_maps/{map}" for map in os.listdir(online_train_maps_dir)],
    "offline": [f"maps/offline_train_maps/{map}" for map in os.listdir(offline_train_maps_dir)]
}

# ais = [microrts_ai.coacAI, microrts_ai.randomAI, microrts_ai.workerRushAI, microrts_ai.lightRushAI, microrts_ai.mayari]
default_ais = [microrts_ai.coacAI, microrts_ai.workerRushAI, microrts_ai.lightRushAI, microrts_ai.mayari,
       microrts_ai.coacAI, microrts_ai.coacAI, microrts_ai.mayari, microrts_ai.mayari]

def make_env(max_steps, num_envs, seed, map_set, ais=None):
    if ais is None:
        ais = default_ais
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


def sample_maps(num_maps, map_set):
    return random.sample(maps[map_set], num_maps)

def make_eval_env(max_steps, num_envs, map_paths, seed, ais=None):
    if ais is None:
        ais = default_ais
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

def get_env_spec(max_steps):
    env, _, _ = make_env(max_steps, 1, 0, "eval")
    state_dim = env.observation_space.shape
    action_dim = env.action_plane_space.nvec.tolist()

    print("\n\n------------------------")
    print("Env Spec:")
    print("\tState Dim:", state_dim)
    print("\tAction Dim:", action_dim)
    print("------------------------\n\n")

    return state_dim, action_dim, env