from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from marl_tsc.wrappers import PeerRewardingWrapper


def _stack_agent_observations(
    observations: dict[str, np.ndarray],
    agent_ids: tuple[str, ...],
) -> np.ndarray:
    rows = [np.asarray(observations[agent], dtype=np.float32) for agent in agent_ids]
    return np.stack(rows, axis=0)


def _stack_agent_masks(
    infos: dict[str, dict] | None,
    agent_ids: tuple[str, ...],
    total_action_dim: int,
) -> np.ndarray:
    """Build a (num_agents, total_action_dim) mask from infos."""
    if not infos:
        return np.ones((len(agent_ids), total_action_dim), dtype=np.float32)

    rows = []
    for agent in agent_ids:
        info = infos.get(agent) or {}
        mask = info.get("action_mask")
        if mask is None:
            rows.append(np.ones(total_action_dim, dtype=np.float32))
        else:
            rows.append(np.asarray(mask, dtype=np.float32))
    return np.stack(rows, axis=0)


def _mask_logits(logits, mask):
    """Set logits at masked-out actions to -inf so the Categorical zeroes them out."""
    import torch

    very_negative = torch.finfo(logits.dtype).min
    return torch.where(mask > 0.5, logits, torch.full_like(logits, very_negative))


class RunningMeanStd:
    """Welford-style running statistics over a stream of scalar values."""

    def __init__(self) -> None:
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, batch: np.ndarray) -> None:
        batch = np.asarray(batch, dtype=np.float64).reshape(-1)
        n = batch.size
        if n == 0:
            return
        batch_mean = float(batch.mean())
        batch_var = float(batch.var())
        delta = batch_mean - self.mean
        new_count = self.count + n
        self.mean = self.mean + delta * n / new_count
        m_a = self.var * self.count
        m_b = batch_var * n
        m2 = m_a + m_b + (delta ** 2) * self.count * n / new_count
        self.var = m2 / new_count
        self.count = new_count

    @property
    def std(self) -> float:
        return float(np.sqrt(max(self.var, 1e-8)))


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalised Advantage Estimation targets."""
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
    action_dims: list[int]
    config: dict[str, Any]
    device: str = "cpu"
    policy_name: str = "MAPPO"

    def _act_single(
        self,
        observation: np.ndarray,
        mask: np.ndarray | None = None,
        deterministic: bool = True,
    ) -> list[int]:
        import torch
        from torch.distributions import Categorical

        obs_tensor = torch.as_tensor(
            np.asarray(observation, dtype=np.float32), 
            dtype=torch.float32, 
            device=self.device
        )

        with torch.no_grad():
            # Get the list of logits (one for traffic, one for sharing)
            logits_list = self.actor(obs_tensor.unsqueeze(0))
            traffic_dim = self.action_dims[0]

            if mask is not None:
                mask_tensor = torch.as_tensor(
                    np.asarray(mask, dtype=np.float32), 
                    dtype=torch.float32, 
                    device=self.device
                )
                t_mask = mask_tensor[..., :traffic_dim]
                logits_list[0] = _mask_logits(logits_list[0].squeeze(0), t_mask)
                if len(logits_list) > 1:
                    s_mask = mask_tensor[..., traffic_dim:]
                    logits_list[1] = _mask_logits(logits_list[1].squeeze(0), s_mask)
            else:
                logits_list[0] = logits_list[0].squeeze(0)
                if len(logits_list) > 1:
                    logits_list[1] = logits_list[1].squeeze(0)

            actions = []
            for logits in logits_list:
                if deterministic:
                    actions.append(int(torch.argmax(logits, dim=-1).item()))
                else:
                    actions.append(int(Categorical(logits=logits).sample().item()))

        return actions if len(actions) > 1 else actions[0]

    def act(
        self,
        observations: dict[str, np.ndarray],
        infos: dict[str, dict] | None = None,
        deterministic: bool = True,
    ) -> dict[str, list[int]]:
        actions: dict[str, list[int]] = {}
        for agent_id in self.traffic_light_ids:
            mask = None
            if infos is not None:
                info = infos.get(agent_id) or {}
                mask = info.get("action_mask")
            actions[agent_id] = self._act_single(
                observations[agent_id], mask=mask, deterministic=deterministic
            )
        return actions

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        return self._act_single(observation, deterministic=deterministic), None

    def __call__(self, observation: np.ndarray, deterministic: bool = True):
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
    use_peer_reward=True,
):
    """Train a shared-actor / centralised-critic MAPPO baseline with peer rewarding."""

    from marl_tsc.traffic_env import SumoTrafficEnv
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Categorical
    from torch.optim import Adam
    from collections import deque

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )
    
    # Apply peer rewarding for MAPPO training
    if use_peer_reward:
        env = PeerRewardingWrapper(env, division=10)
    observations, infos = env.reset(seed=seed)
    agent_ids = tuple(traffic_light_ids or env.agents)
    
    if not agent_ids:
        env.close()
        raise ValueError("traffic_light_ids must contain at least one agent.")

    obs_dim = int(np.asarray(observations[agent_ids[0]], dtype=np.float32).shape[0])
    
    # Extract multiple action dimensions for the two branches
    action_space = env.action_space(agent_ids[0])
    if hasattr(action_space, 'nvec'):
        action_dims = action_space.nvec.tolist()
    else:
        action_dims = [action_space.n]
    total_action_dim = sum(action_dims)
    num_agents = len(agent_ids)
    
    hidden_size = 256
    critic_hidden_size = 256

    class Actor(nn.Module):
        def __init__(
            self,
            obs_size: int,
            action_dims: list[int],
            phase_queue_start: int | None = None,
        ) -> None:
            super().__init__()
            self.action_dims = action_dims
            self.phase_queue_start = phase_queue_start
            self.queue_logit_scale = nn.Parameter(torch.tensor(2.0))
            
            # Base network shared by both action branches
            self.base = nn.Sequential(
                nn.Linear(obs_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
            )
            
            # Create a separate branch for traffic phase and sharing percentage
            self.branches = nn.ModuleList([
                nn.Linear(hidden_size, dim) for dim in action_dims
            ])

        def forward(self, obs: torch.Tensor) -> list[torch.Tensor]:
            x = self.base(obs)
            logits = [branch(x) for branch in self.branches]

            # Apply queue logic ONLY to the traffic branch (index 0)
            if self.phase_queue_start is not None:
                traffic_dim = self.action_dims[0]
                phase_queues = obs[
                    ..., self.phase_queue_start : self.phase_queue_start + traffic_dim
                ]
                logits[0] = logits[0] + self.queue_logit_scale.clamp(0.0, 5.0) * phase_queues
                
            return logits

    class Critic(nn.Module):
        def __init__(self, central_obs_size: int) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(central_obs_size, critic_hidden_size),
                nn.ReLU(),
                nn.Linear(critic_hidden_size, critic_hidden_size),
                nn.ReLU(),
                nn.Linear(critic_hidden_size, num_agents),
            )

        def forward(self, central_obs: torch.Tensor) -> torch.Tensor:
            return self.net(central_obs)

    phase_queue_start = None
    if env_options.get("include_phase_queue_features", True):
        phase_queue_start = int(env.max_lanes_per_tls) + 2

    actor = Actor(obs_dim, action_dims, phase_queue_start=phase_queue_start).to(device)
    critic = Critic(obs_dim * num_agents).to(device)
    
    learning_rate = 3e-5
    optimizer = Adam(list(actor.parameters()) + list(critic.parameters()), lr=learning_rate)

    gamma = 0.99
    gae_lambda = 0.95
    clip_coef = 0.2
    update_epochs = 10
    entropy_coef = 0.05
    value_coef = 0.5
    local_reward_weight = 0.7

    value_norm = RunningMeanStd()

    history: list[dict[str, Any]] = []
    global_steps = 0
    episode_index = 0

    episode_returns = deque(maxlen=100)
    current_episode_return = 0.0

    while global_steps < total_timesteps:
        rollout_local_obs: list[np.ndarray] = []
        rollout_central_obs: list[np.ndarray] = []
        rollout_actions: list[np.ndarray] = []
        rollout_log_probs: list[np.ndarray] = []
        rollout_masks: list[np.ndarray] = []
        rollout_rewards: list[np.ndarray] = []
        rollout_values: list[np.ndarray] = []
        rollout_dones: list[bool] = []

        while len(rollout_rewards) < rollout_steps and global_steps < total_timesteps:
            local_obs = _stack_agent_observations(observations, agent_ids)
            central_obs = local_obs.reshape(-1)
            # Fetch the combined masks for both actions
            mask_array = _stack_agent_masks(infos, agent_ids, total_action_dim)

            with torch.no_grad():
                local_obs_tensor = torch.as_tensor(local_obs, dtype=torch.float32, device=device)
                local_obs_tensor = (local_obs_tensor - local_obs_tensor.mean(dim=-1, keepdim=True)) / (local_obs_tensor.std(dim=-1, keepdim=True) + 1e-8)
                central_obs_tensor = torch.as_tensor(central_obs, dtype=torch.float32, device=device)
                mask_tensor = torch.as_tensor(mask_array, dtype=torch.float32, device=device)
                
                # Retrieve logits for both branches
                logits_list = actor(local_obs_tensor)
                
                traffic_dim = action_dims[0]
                t_mask = mask_tensor[:, :traffic_dim]
                s_mask = mask_tensor[:, traffic_dim:]
                
                masked_t_logits = _mask_logits(logits_list[0], t_mask)
                dist_t = Categorical(logits=masked_t_logits)
                
                if len(action_dims) > 1:
                    masked_s_logits = _mask_logits(logits_list[1], s_mask)
                    dist_s = Categorical(logits=masked_s_logits)
                else:
                    dist_s = None
                
                act_t = dist_t.sample()
                act_s = dist_s.sample() if dist_s is not None else torch.zeros_like(act_t)
                
                # Sum the log probabilities from both branches
                log_probs_tensor = dist_t.log_prob(act_t) + (dist_s.log_prob(act_s) if dist_s is not None else 0.0)
                value_tensor = critic(central_obs_tensor)

            actions = {
                agent_id: [int(t), int(s)] if dist_s is not None else int(t)
                for agent_id, t, s in zip(agent_ids, act_t.cpu().tolist(), act_s.cpu().tolist())
            }
            
            next_observations, rewards, terminations, truncations, next_infos = env.step(actions)
            local_rewards = np.asarray([float(rewards.get(agent_id, 0.0)) for agent_id in agent_ids], dtype=np.float32)
            global_reward = float(np.mean(local_rewards)) if local_rewards.size else 0.0
            blended_rewards = local_reward_weight * local_rewards + (1.0 - local_reward_weight) * global_reward
            done = bool(any(terminations.values()) or any(truncations.values()))

            value_denorm = value_tensor.detach().cpu().numpy().astype(np.float32) * value_norm.std + value_norm.mean

            rollout_local_obs.append(local_obs)
            rollout_central_obs.append(central_obs)
            
            # Store both actions in a unified tensor
            combined_actions = torch.stack([act_t, act_s], dim=-1)
            rollout_actions.append(np.asarray(combined_actions.cpu().tolist(), dtype=np.int64))
            
            rollout_log_probs.append(np.asarray(log_probs_tensor.cpu().tolist(), dtype=np.float32))
            rollout_masks.append(mask_array)
            rollout_rewards.append(blended_rewards)
            rollout_values.append(value_denorm)
            rollout_dones.append(done)

            global_steps += 1
            observations = next_observations
            infos = next_infos
            current_episode_return += float(np.mean(blended_rewards))

            if done:
                episode_index += 1
                episode_returns.append(current_episode_return)
                print(f"[MAPPO] Episode {episode_index} | Return: {current_episode_return:.2f} | Total Steps: {global_steps}")
                current_episode_return = 0.0
                observations, infos = env.reset(seed=seed + episode_index)

        if not rollout_rewards:
            break

        if rollout_dones[-1]:
            last_value = np.zeros(num_agents, dtype=np.float32)
        else:
            with torch.no_grad():
                next_local_obs = _stack_agent_observations(observations, agent_ids)
                next_central_obs = next_local_obs.reshape(-1)
                next_central_obs_tensor = torch.as_tensor(next_central_obs, dtype=torch.float32, device=device)
                raw_value = critic(next_central_obs_tensor).detach().cpu().numpy().astype(np.float32)
                last_value = raw_value * value_norm.std + value_norm.mean

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
        advantages = np.clip(advantages, -10.0, 10.0)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        value_norm.update(returns)
        normalized_returns = (returns - value_norm.mean) / value_norm.std

        local_obs_batch = np.concatenate(rollout_local_obs, axis=0)
        action_batch = np.concatenate(rollout_actions, axis=0)
        old_log_prob_batch = np.concatenate(rollout_log_probs, axis=0)
        mask_batch = np.concatenate(rollout_masks, axis=0)
        advantage_batch = advantages.reshape(-1)
        return_batch = torch.as_tensor(normalized_returns, dtype=torch.float32, device=device)
        central_obs_batch = torch.as_tensor(
            np.asarray(rollout_central_obs),
            dtype=torch.float32,
            device=device,
        )

        actor_obs_batch = torch.as_tensor(local_obs_batch, dtype=torch.float32, device=device)
        actor_action_batch = torch.as_tensor(action_batch, dtype=torch.int64, device=device)
        old_log_prob_batch_tensor = torch.as_tensor(
            old_log_prob_batch,
            dtype=torch.float32,
            device=device,
        )
        advantage_batch_tensor = torch.as_tensor(
            advantage_batch,
            dtype=torch.float32,
            device=device,
        )
        mask_batch_tensor = torch.as_tensor(mask_batch, dtype=torch.float32, device=device)

        for _ in range(update_epochs):
            logits_list = actor(actor_obs_batch)
            
            traffic_dim = action_dims[0]
            t_mask = mask_batch_tensor[:, :traffic_dim]
            s_mask = mask_batch_tensor[:, traffic_dim:]
            
            masked_t_logits = _mask_logits(logits_list[0], t_mask)
            dist_t = Categorical(logits=masked_t_logits)
            
            if len(action_dims) > 1:
                masked_s_logits = _mask_logits(logits_list[1], s_mask)
                dist_s = Categorical(logits=masked_s_logits)
            else:
                dist_s = None
            
            # Read the actions back out from the batch array
            act_t = actor_action_batch[:, 0]
            act_s = actor_action_batch[:, 1]
            
            # Sum the probabilities and entropies
            new_log_probs = dist_t.log_prob(act_t) + (dist_s.log_prob(act_s) if dist_s is not None else 0.0)
            entropy_loss = dist_t.entropy().mean() + (dist_s.entropy().mean() if dist_s is not None else 0.0)
            
            ratio = torch.exp(new_log_probs - old_log_prob_batch_tensor)
            unclipped = ratio * advantage_batch_tensor
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantage_batch_tensor

            policy_loss = -torch.min(unclipped, clipped).mean()

            predicted_values = critic(central_obs_batch)
            value_loss = F.mse_loss(predicted_values, return_batch)
            
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                0.5,
            )
            optimizer.step()

        history.append(
            {
                "algorithm": "mappo",
                "timestep": global_steps,
                "mean_training_reward": float(np.mean(rewards_array)),
                "moving_avg_episode_return": (
                    float(np.mean(episode_returns)) if episode_returns else 0.0
                ),
                "episodes_completed": episode_index,
            }
        )
        
    policy_label = "mappo_peer_reward" if use_peer_reward else "mappo"
    model = MappoModel(
        actor=actor,
        critic=critic,
        traffic_light_ids=agent_ids,
        obs_dim=obs_dim,
        action_dims=action_dims,
        config={
            "algorithm_name": policy_label,
            "traffic_light_ids": list(agent_ids),
            "obs_dim": obs_dim,
            "action_dims": action_dims,
            "num_agents": num_agents,
            "total_timesteps": total_timesteps,
            "rollout_steps": rollout_steps,
            "max_steps": max_steps,
            "seed": seed,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_coef": clip_coef,
            "learning_rate": learning_rate,
            "update_epochs": update_epochs,
            "entropy_coef": entropy_coef,
            "value_coef": value_coef,
            "local_reward_weight": local_reward_weight,
            "phase_queue_start": phase_queue_start,
            "env_kwargs": env_options,
        },
        device=str(device),
    )

    output_dir = Path(output_dir)
    model_path = output_dir / "models" / f"{policy_label}.pt"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "value_norm_mean": value_norm.mean,
            "value_norm_var": value_norm.var,
            "value_norm_count": value_norm.count,
            "config": model.config,
        },
        model_path,
    )

    env.close()
    return model, history, model_path
