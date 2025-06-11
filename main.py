# modified from source: https://github.com/gwthomas/IQL-PyTorch
# https://arxiv.org/pdf/2110.06169.pdf
import copy
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import asdict
from typing import Any, Dict, List

import gym_microrts
import gym_microrts.envs.vec_env
import gym_microrts.microrts_ai
import gym_microrts.envs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from env_utils import get_env_spec, make_eval_env, sample_maps
from experiment.iql.iql_model import IQLNetwork
from experiment.iql.transition_set import TransitionSet
from experiment.iql.train_config import IQLTrainingConfig, TrainConfig
import yaml

from utils import set_seed_everywhere

TensorBatch = List[torch.Tensor]

EXP_ADV_MAX = 100.0

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

def wandb_init(config: TrainConfig) -> None:
    wandb.init(
        config=asdict(config),
        project=config.environment.project,
        group=config.environment.group,
    )

    # get run name from wandb
    run_name = wandb.run.name
    config.environment.name = run_name
    print(f"Run name: {run_name}")

@torch.no_grad()
def eval_actor(
    env: gym_microrts.envs.vec_env.MicroRTSGridModeVecEnv, actor: nn.Module, device: str, n_episodes: int, seed: int
) -> np.ndarray:
    actor.eval()
    episode_rewards = []
    for _ in range(n_episodes):
        state, done = env.reset(), False
        episode_reward = 0.0
        while not done:
            env.render()
            action_mask = env.get_action_mask()
            action_mask = torch.tensor(action_mask, dtype=torch.float32, device=device)
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
        actor: nn.Module,
        actor_optimizer: torch.optim.Optimizer,
        q_network: nn.Module,
        q_optimizer: torch.optim.Optimizer,
        v_network: nn.Module,
        v_optimizer: torch.optim.Optimizer,
        config: IQLTrainingConfig,
        device: str = "cpu",
    ):
        
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
            pct_start=0.45
        )
        # self.actor_lr_schedule = CosineAnnealingLR(self.actor_optimizer, max_steps)

        self.total_it = 0
        self.device = device

    def _update_v(self, observations, actions, log_dict) -> torch.Tensor:
        # Update value function
        with torch.no_grad():
            target_q = self.q_target(observations, actions)

        v = self.vf(observations)
        adv = target_q - v
        v_loss = asymmetric_l2_loss(adv, self.iql_tau)
        log_dict["value_loss"] = v_loss.item()
        self.v_optimizer.zero_grad()
        v_loss.backward()
        self.v_optimizer.step()
        return adv

    def _update_q(
        self,
        next_v: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminals: torch.Tensor,
        log_dict: Dict,
    ):
        targets = rewards + (1.0 - terminals.float()) * self.discount * next_v.detach()
        qs = self.qf.both(observations, actions)
        q_loss = sum(F.mse_loss(q, targets, reduction="sum") for q in qs) / len(qs)
        # q_loss = sum(F.mse_loss(q, targets) for q in qs) / len(qs)
        log_dict["q_loss"] = q_loss.item()
        self.q_optimizer.zero_grad()
        q_loss.backward()
        self.q_optimizer.step()

        # Update target Q network
        soft_update(self.q_target, self.qf, self.tau)

    def _update_policy(
        self,
        adv: torch.Tensor,
        observations: torch.Tensor,
        actions: torch.Tensor,
        action_masks: torch.Tensor,
        log_dict: Dict,
    ):
        exp_adv = torch.exp(self.beta * adv.detach()).clamp(max=EXP_ADV_MAX)
        policy_out = self.actor(observations, action_masks)
        # bc_losses = ((policy_out - actions) ** 2).sum(dim=(-1)).mean(dim=(-1, -2))

        batch_size = observations.shape[0]

        log_probs = torch.log(policy_out + 1e-8)
        nll = (-(actions * log_probs)).view(batch_size, -1)
        exp_adv = exp_adv.view(batch_size, 1)
        action_masks = action_masks.view(batch_size, -1)
        policy_loss = (exp_adv * nll)[action_masks != 0]
        policy_loss = policy_loss.sum() / exp_adv.shape[0]

        log_dict["actor_loss"] = policy_loss.item()
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.actor_lr_schedule.step()
        log_dict["actor_lr"] = self.actor_optimizer.param_groups[0]["lr"]

    def train(self, batch: TensorBatch) -> Dict[str, float]:
        self.total_it += 1
        (
            observations,
            actions,
            action_masks,
            rewards,
            next_observations,
            dones,
        ) = batch
        log_dict = {}

        with torch.no_grad():
            next_v = self.vf(next_observations)
        # Update value function
        adv = self._update_v(observations, actions, log_dict)
        rewards = rewards.squeeze(dim=-1)
        dones = dones.squeeze(dim=-1)
        # Update Q function
        self._update_q(next_v, observations, actions, rewards, dones, log_dict)
        # Update actor
        self._update_policy(adv, observations, actions, action_masks, log_dict)

        return log_dict

    def state_dict(self) -> Dict[str, Any]:
        return {
            "qf": self.qf.state_dict(),
            "q_optimizer": self.q_optimizer.state_dict(),
            "vf": self.vf.state_dict(),
            "v_optimizer": self.v_optimizer.state_dict(),
            "actor": self.actor.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "actor_lr_schedule": self.actor_lr_schedule.state_dict(),
            "total_it": self.total_it,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.qf.load_state_dict(state_dict["qf"])
        self.q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self.q_target = copy.deepcopy(self.qf)

        self.vf.load_state_dict(state_dict["vf"])
        self.v_optimizer.load_state_dict(state_dict["v_optimizer"])

        self.actor.load_state_dict(state_dict["actor"])
        self.actor_optimizer.load_state_dict(state_dict["actor_optimizer"])
        self.actor_lr_schedule.load_state_dict(state_dict["actor_lr_schedule"])

        self.total_it = state_dict["total_it"]

def train(config: TrainConfig):
    # Set seeds
    seed = config.seed
    set_seed_everywhere(seed)

    state_dim, action_dim, _ = get_env_spec({
        "max_steps": 2000,
        "seed": seed,
    })

    replay_buffer = TransitionSet(
        config.data.buffer_path,
        map_size_gb=150,
    )

    print(f"Replay buffer size: {len(replay_buffer)}")

    iql_network = IQLNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        config=config.iql.model,
        device=config.device,
    )

    q_network = iql_network.critic
    v_network = iql_network.value
    actor = iql_network.actor

    v_optimizer = torch.optim.Adam(v_network.parameters(), lr=config.iql.training.vf_lr)
    q_optimizer = torch.optim.Adam(q_network.parameters(), lr=config.iql.training.qf_lr)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.iql.training.actor_lr)

    print("---------------------------------------")
    print(f"Training IQL, Env: GymMicroRTS, Seed: {seed}")
    print("---------------------------------------")

    # Initialize actor
    trainer = ImplicitQLearning(
        actor = actor,
        actor_optimizer = actor_optimizer,
        q_network = q_network,
        q_optimizer = q_optimizer,
        v_network = v_network,
        v_optimizer = v_optimizer,
        config = config.iql.training,
        device = config.device,
    )

    wandb_init(config)

    maps = sample_maps(config.iql.eval.num_longer_episodes, "eval")
    env, _, _ = make_eval_env({ "max_steps": 2000 }, 1, maps, seed, ais=[gym_microrts.microrts_ai.coacAI])

    now = datetime.now().strftime("%Y.%m.%d/%H%M%S")
    save_path = Path(f"{config.environment.save_dir}/{config.environment.group}/{now}-{config.environment.name}")
    save_path.mkdir(parents=True, exist_ok=True)

    for t in range(int(config.iql.training.max_timesteps)):
        # batch = replay_buffer.sample(reward_scale=config.reward_scale, device=config.device)
        batch = replay_buffer.sample_next(batch_size=config.iql.training.batch_size, reward_scale=config.iql.training.reward_scale)
        batch = [torch.tensor(b, dtype=torch.float32, device=config.device) for b in batch]
        log_dict = trainer.train(batch)
        wandb.log(log_dict, step=trainer.total_it)
        # Evaluate episode
        if (t + 1) % config.data.log_interval == 0:
            print(f"Training step: {trainer.total_it}")
            print(f"Actor loss: {log_dict['actor_loss']:.3f}")
            print(f"Q loss: {log_dict['q_loss']:.3f}")
            print(f"Value loss: {log_dict['value_loss']:.3f}")
            print("---------------")
        if (t + 1) % config.data.save_interval == 0:
            # print(f"Saving model at step {trainer.total_it}")
            # wandb.save("*.pth")
            print(f"Saving model at step {trainer.total_it} to {save_path}")
            torch.save(
                trainer.state_dict(),
                f"{save_path}/model_{trainer.total_it}.pth",
            )

            # perform longer evaluation
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.iql.eval.num_longer_episodes,
                seed=seed,
            )
            eval_score = eval_scores.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.iql.eval.num_longer_episodes} episodes: "
                f"{eval_score:.3f}"
            )
            print("---------------------------------------")
            wandb.log(
                {"eval_score_long": eval_score}, step=trainer.total_it
            )
        elif (t + 1) % config.iql.eval.eval_freq == 0:
            print(f"Time steps: {t + 1}")
            eval_scores = eval_actor(
                env,
                actor,
                device=config.device,
                n_episodes=config.iql.eval.num_episodes,
                seed=config.seed,
            )
            eval_score = eval_scores.mean()
            print("---------------------------------------")
            print(
                f"Evaluation over {config.iql.eval.num_episodes} episodes: "
                f"{eval_score:.3f}"
            )
            print("---------------------------------------")
            wandb.log(
                {"eval_score": eval_score}, step=trainer.total_it
            )

    replay_buffer.close()
    env.close()
    wandb.finish()


if __name__ == "__main__":
    with open("hyperparams.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainConfig.from_dict(config_dict)
    train(config)