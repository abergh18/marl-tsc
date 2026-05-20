"""Minimal educational MAPPO trainer for the traffic signal project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _stack_agent_observations(observations: dict[str, np.ndarray], agent_ids: tuple[str, ...]) -> np.ndarray:
    return np.stack([np.asarray(observations[agent], dtype=np.float32) for agent in agent_ids], axis=0)


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0

    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        next_non_terminal = 1.0 - float(dones[step])
        delta = rewards[step] + gamma * next_value * next_non_terminal - values[step]
        gae = delta + gamma * gae_lambda * next_non_terminal * gae
        advantages[step] = gae

    returns = advantages + values
    return advantages, returns


@dataclass
class MappoModel:
    """Trained MAPPO actor/critic bundle used for evaluation."""

    actor: Any
    critic: Any
    traffic_light_ids: tuple[str, ...]
    obs_dim: int
    action_dim: int
    config: dict[str, Any]
    device: str = "cpu"
    policy_name: str = "MAPPO"

    def _act_single(self, observation: np.ndarray, deterministic: bool = True) -> int:
        import torch
        from torch.distributions import Categorical

        obs_tensor = torch.as_tensor(np.asarray(observation, dtype=np.float32), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            logits = self.actor(obs_tensor.unsqueeze(0)).squeeze(0)
            if deterministic:
                action_tensor = torch.argmax(logits, dim=-1)
            else:
                action_tensor = Categorical(logits=logits).sample()

        return int(action_tensor.item())

    def act(self, observations: dict[str, np.ndarray], deterministic: bool = True) -> dict[str, int]:
        return {
            agent_id: self._act_single(observations[agent_id], deterministic=deterministic)
            for agent_id in self.traffic_light_ids
        }

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        """SB3-style compatibility for single-agent evaluation paths."""

        return self._act_single(observation, deterministic=deterministic), None

    def __call__(self, observation: np.ndarray, deterministic: bool = True):
        """Extra compatibility for older notebook cells or ad hoc usage."""

        return self.predict(observation, deterministic=deterministic)


def train_mappo(
    config_file,
    traffic_light_ids,
    output_dir,
    total_timesteps=50_000,
    rollout_steps=256,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
):
    """Train a small shared-actor / centralized-critic MAPPO baseline."""

    from marl_tsc.traffic_env import SumoTrafficEnv
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from torch.optim import Adam

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )

    observations, _ = env.reset(seed=seed)
    agent_ids = tuple(traffic_light_ids)
    if not agent_ids:
        env.close()
        raise ValueError("traffic_light_ids must contain at least one agent.")

    obs_dim = int(np.asarray(observations[agent_ids[0]], dtype=np.float32).shape[0])
    action_dim = int(env.action_space(agent_ids[0]).n)
    num_agents = len(agent_ids)
    hidden_size = 64

    class Actor(nn.Module):
        def __init__(self, obs_size: int, action_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, action_size),
            )

        def forward(self, obs: torch.Tensor) -> torch.Tensor:
            return self.net(obs)

    class Critic(nn.Module):
        def __init__(self, central_obs_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(central_obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1),
            )

        def forward(self, central_obs: torch.Tensor) -> torch.Tensor:
            return self.net(central_obs).squeeze(-1)

    actor = Actor(obs_dim, action_dim).to(device)
    critic = Critic(obs_dim * num_agents).to(device)
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=3e-4)

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 4
    entropy_coef = 0.01
    value_coef = 0.5

    history: list[dict[str, Any]] = []
    global_steps = 0
    episode_index = 0

    while global_steps < total_timesteps:
        rollout_local_obs: list[np.ndarray] = []
        rollout_central_obs: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        rollout_log_probs: list[np.ndarray] = []
        rollout_rewards: list[float] = []
        rollout_values: list[float] = []
        rollout_dones: list[bool] = []

        while len(rollout_rewards) < rollout_steps and global_steps < total_timesteps:
            local_obs = _stack_agent_observations(observations, agent_ids)
            central_obs = local_obs.reshape(-1)

            with torch.no_grad():
                local_obs_tensor = torch.as_tensor(local_obs, dtype=torch.float32, device=device)
                central_obs_tensor = torch.as_tensor(central_obs, dtype=torch.float32, device=device)
                logits = actor(local_obs_tensor)
                dist = Categorical(logits=logits)
                actions_tensor = dist.sample()
                log_probs_tensor = dist.log_prob(actions_tensor)
                value_tensor = critic(central_obs_tensor)

            actions = {
                agent_id: int(action)
                for agent_id, action in zip(agent_ids, actions_tensor.cpu().tolist())
            }
            next_observations, rewards, terminations, truncations, _ = env.step(actions)
            team_reward = float(np.mean(list(rewards.values()))) if rewards else 0.0
            done = bool(any(terminations.values()) or any(truncations.values()))

            rollout_local_obs.append(local_obs)
            rollout_central_obs.append(central_obs)
            rollout_actions.append(np.asarray(actions_tensor.cpu().tolist(), dtype=np.int64))
            rollout_log_probs.append(np.asarray(log_probs_tensor.cpu().tolist(), dtype=np.float32))
            rollout_rewards.append(team_reward)
            rollout_values.append(float(value_tensor.item()))
            rollout_dones.append(done)

            global_steps += 1
            observations = next_observations

            if done:
                episode_index += 1
                observations, _ = env.reset(seed=seed + episode_index)

        if not rollout_rewards:
            break

        if rollout_dones[-1]:
            last_value = 0.0
        else:
            with torch.no_grad():
                next_local_obs = _stack_agent_observations(observations, agent_ids)
                next_central_obs = next_local_obs.reshape(-1)
                next_central_obs_tensor = torch.as_tensor(next_central_obs, dtype=torch.float32, device=device)
                last_value = float(critic(next_central_obs_tensor).item())

        rewards_array = np.asarray(rollout_rewards, dtype=np.float32)
        values_array = np.asarray(rollout_values, dtype=np.float32)
        dones_array = np.asarray(rollout_dones, dtype=np.bool_)
        advantages, returns = _compute_gae(
            rewards_array,
            values_array,
            dones_array,
            last_value,
            gamma,
            gae_lambda,
        )
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        local_obs_batch = np.concatenate(rollout_local_obs, axis=0)
        action_batch = np.concatenate(rollout_actions, axis=0)
        old_log_prob_batch = np.concatenate(rollout_log_probs, axis=0)
        advantage_batch = np.repeat(advantages, num_agents)
        return_batch = torch.as_tensor(returns, dtype=torch.float32, device=device)
        central_obs_batch = torch.as_tensor(np.asarray(rollout_central_obs), dtype=torch.float32, device=device)

        actor_obs_batch = torch.as_tensor(local_obs_batch, dtype=torch.float32, device=device)
        actor_action_batch = torch.as_tensor(action_batch, dtype=torch.int64, device=device)
        old_log_prob_batch_tensor = torch.as_tensor(old_log_prob_batch, dtype=torch.float32, device=device)
        advantage_batch_tensor = torch.as_tensor(advantage_batch, dtype=torch.float32, device=device)

        for _ in range(update_epochs):
            logits = actor(actor_obs_batch)
            dist = Categorical(logits=logits)
            new_log_probs = dist.log_prob(actor_action_batch)
            ratio = torch.exp(new_log_probs - old_log_prob_batch_tensor)
            unclipped = ratio * advantage_batch_tensor
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage_batch_tensor
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = dist.entropy().mean()

            predicted_values = critic(central_obs_batch)
            value_loss = F.mse_loss(predicted_values, return_batch)
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()), 0.5)
            optimizer.step()

        history.append(
            {
                "algorithm": "mappo",
                "timestep": global_steps,
                "mean_training_reward": float(np.mean(rewards_array)),
            }
        )

    model = MappoModel(
        actor=actor,
        critic=critic,
        traffic_light_ids=agent_ids,
        obs_dim=obs_dim,
        action_dim=action_dim,
        config={
            "algorithm_name": "mappo",
            "traffic_light_ids": list(agent_ids),
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "num_agents": num_agents,
            "total_timesteps": total_timesteps,
            "rollout_steps": rollout_steps,
            "max_steps": max_steps,
            "seed": seed,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "learning_rate": 3e-4,
            "update_epochs": update_epochs,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "env_kwargs": env_options,
        },
        device=str(device),
    )

    output_dir = Path(output_dir)
    model_path = output_dir / "models" / "mappo.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "config": model.config,
        },
        model_path,
    )

    env.close()
    return model, history, model_path
