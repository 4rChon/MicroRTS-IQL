# modified from source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
from pathlib import Path
from time import time
from typing import Any

import gym_microrts
import gym_microrts.envs
import gym_microrts.envs.vec_env
import gym_microrts.microrts_ai
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from env_utils import get_env_spec, make_eval_env, sample_maps
from iql_microrts.iql_model import (ActorPolicy, IQLNetwork, TwinQ,
                                    ValueFunction)
from iql_microrts.train_config import IQLTrainingConfig, TrainConfig
from iql_microrts.transition_set import TransitionDataLoader, TransitionSet
from utils import set_seed_everywhere

TensorBatch = list[torch.Tensor]

EXP_ADV_MAX = 100.0


def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(
        target.parameters(), source.parameters()
    ):
        target_param.data.copy_(
            (1 - tau) * target_param.data + tau * source_param.data
        )


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
                action_mask,
                dtype=torch.float32,
                device=device
            )
            state = torch.tensor(state, dtype=torch.float32, device=device)
            action = actor.act(state, action_mask).detach().cpu().numpy()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
        episode_rewards.append(episode_reward)

    actor.train()
    return np.asarray(episode_rewards)


def asymmetric_l2_loss(u: torch.Tensor, tau: float) -> torch.Tensor:
    return torch.mean(torch.abs(tau - (u < 0).float()) * u**2)


class ImplicitQLearning:
    def __init__(
        self,
        actor: ActorPolicy,
        actor_optimizer: torch.optim.Optimizer,
        q_network: TwinQ,
        q_optimizer: torch.optim.Optimizer,
        v_network: ValueFunction,
        v_optimizer: torch.optim.Optimizer,
        config: IQLTrainingConfig,
        device: torch.device,
    ):
        print("Initializing IQL Trainer...")
        self.iql_tau = config.iql_tau
        self.beta = config.beta
        self.discount = config.discount
        self.tau = config.tau

        self.qf = q_network
        self.q_target = copy.deepcopy(self.qf).requires_grad_(False).to(device)
        self.vf = v_network
        self.actor = actor
        self.v_optimizer = v_optimizer
        self.q_optimizer = q_optimizer
        self.actor_optimizer = actor_optimizer
        self.actor_lr_schedule = torch.optim.lr_scheduler.OneCycleLR(
            self.actor_optimizer,
            max_lr=actor_optimizer.param_groups[0]["lr"],
            total_steps=config.max_timesteps,
            pct_start=config.warmup_pct
        )

        self.total_steps = 0
        self.device = device

    def train(self, batch: TensorBatch) -> dict[str, Any]:
        (
            observations,
            next_observations,
            actions,
            rewards,
            dones,
            action_masks,
        ) = batch
        log_dict: dict[str, Any] = {}

        # Calculate value loss
        with torch.no_grad():
            min_Q = self.q_target(observations, actions)
        v = self.vf(observations)
        adv = min_Q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)

        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()

        # Calculate Q-value loss
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        with torch.no_grad():
            next_v = self.vf(next_observations)
        q1, q2 = self.qf.both(observations, actions)
        target = rewards + (1.0 - dones.float()) * self.discount * next_v
        q_loss = (F.mse_loss(q1, target) + F.mse_loss(q2, target)) * 0.5

        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        soft_update(self.q_target, self.qf, self.tau)

        # Calculate actor loss
        prob = self.actor(observations, action_masks)
        log_prob = torch.log(prob + 1e-8)
        exp_a = torch.exp(adv.detach() * self.beta).clamp(max=EXP_ADV_MAX)
        nll = -(log_prob * actions).sum(dim=(1, 2, 3))
        actor_loss = (exp_a * nll).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()

        log_dict["actor_loss"] = actor_loss.item()
        log_dict["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]
        log_dict["value_loss"] = v_loss.item()
        log_dict["q_loss"] = q_loss.item()

        self.total_steps += 1

        return log_dict

    def state_dict(self) -> dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_steps,
        }

    def load_state_dict(self, state_dict: dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_steps = state_dict["total_it"]


def train(config: TrainConfig, save_path: Path):
    seed = config.seed
    set_seed_everywhere(seed)
    state_dim, action_dim, _ = get_env_spec(
        config.environment.episode_steps_max
    )

    replay_buffer = TransitionSet(
        config.data.buffer_path,
        seed=seed,
        map_size_gb=200,
        state_dim=state_dim,
        action_dim=action_dim,
    )

    dataloader = TransitionDataLoader(
        replay_buffer,
        batch_size=config.iql.training.batch_size,
        num_samples=config.iql.training.max_timesteps,
        num_workers=config.data.num_workers,
    )

    print(f"Replay buffer size: {len(replay_buffer)}")

    iql_network = IQLNetwork(
        state_dim, action_dim, config.iql.model, config.device,
    )

    q_network = iql_network.critic
    v_network = iql_network.value
    actor = iql_network.actor

    v_optimizer = torch.optim.AdamW(
        v_network.parameters(), lr=config.iql.training.vf_lr, fused=True
    )
    q_optimizer = torch.optim.AdamW(
        q_network.parameters(), lr=config.iql.training.qf_lr, fused=True
    )
    actor_optimizer = torch.optim.AdamW(
        actor.parameters(), lr=config.iql.training.actor_lr, fused=True
    )

    print("---------------------------------------")
    print(f"Training IQL, Env: Gym-MicroRTS, Seed: {seed}")
    print("---------------------------------------")

    trainer = ImplicitQLearning(
        actor=actor,
        actor_optimizer=actor_optimizer,
        q_network=q_network,
        q_optimizer=q_optimizer,
        v_network=v_network,
        v_optimizer=v_optimizer,
        config=config.iql.training,
        device=config.device,
    )

    maps = sample_maps(config.iql.eval.longer_episodes_num, "eval")
    env, _, _ = make_eval_env(
        config.environment.episode_steps_max,
        1,
        maps,
        seed,
        ais=[gym_microrts.microrts_ai.coacAI]
    )

    max_step_time = config.environment.time_seconds_max
    step_time = 0
    start_time = time()
    for step, batch in enumerate(dataloader):
        step_start_time = time()
        batch = [
            b.to(config.device, non_blocking=True)
            for b in batch
        ]
        log_dict = trainer.train(batch)
        step_time += time() - step_start_time

        if (step + 1) % config.data.log_interval == 0:
            log_dict["step_time"] = step_time / config.data.log_interval
            log_dict["total_time"] = (time() - start_time)
            if config.environment.time_seconds_max > 0:
                max_step_time = max(0, max_step_time - step_time)
                log_dict["remaining_time"] = max_step_time
            step_time = 0
            wandb.log(log_dict, step=trainer.total_steps)
            print(f"Training step: {trainer.total_steps}")
        if (step + 1) % config.data.save_interval == 0:
            print(f"Saving model at step {trainer.total_steps} to {save_path}")
            torch.save(
                trainer.state_dict(),
                f"{save_path}/model_{trainer.total_steps}.pt"
            )

            eval_score = eval_actor(
                env, actor, config.device, config.iql.eval.longer_episodes_num,
                config.render
            ).mean()
            wandb.log(
                {"eval_score_long": eval_score}, step=trainer.total_steps
            )
        elif (step + 1) % config.iql.eval.eval_freq == 0:
            eval_score = eval_actor(
                env, actor, config.device, config.iql.eval.episodes_num,
                config.render
            ).mean()
            wandb.log({"eval_score": eval_score}, step=trainer.total_steps)
        if max_step_time == 0:
            print("Max step time reached, stopping training.")
            break

    replay_buffer.close()
    env.close()
