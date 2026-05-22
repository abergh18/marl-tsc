"""Training and evaluation helpers for the notebook.

Shared PPO use parameter sharing through the PettingZoo/SuperSuit/SB3
wrapper. This is not true MAPPO because the critic is not centralized.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np


def _make_vec_env(env):
    import supersuit as ss

    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")

    target_env = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    if not hasattr(target_env, "seed"):

        def seed_env(seed_value=None):
            try:
                target_env.reset(seed=seed_value)
            except TypeError:
                target_env.reset()

            return [seed_value for _ in range(getattr(target_env, "num_envs", 1))]

        target_env.seed = seed_env

    return vec_env


def _make_reward_logger_callback(algorithm_name: str):
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardLoggerCallback(BaseCallback):
        def __init__(self):
            super().__init__()
            self.history: list[dict[str, Any]] = []

        def _on_step(self):
            rewards = self.locals.get("rewards")
            if rewards is not None:
                reward_values = [float(reward) for reward in rewards]
                mean_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
                self.history.append(
                    {
                        "algorithm": algorithm_name,
                        "timestep": int(self.num_timesteps),
                        "mean_training_reward": mean_reward,
                    }
                )
            return True

    return RewardLoggerCallback()


def train_ppo(
    config_file,
    traffic_light_ids,
    output_dir,
    total_timesteps=50_000,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
):
    """Train a shared-policy SB3 model and return the model, history, and path."""

    from stable_baselines3 import PPO
    from marl_tsc.traffic_env import SumoTrafficEnv

    algorithm = "ppo"

    env_options = dict(env_kwargs or {})
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )

    vec_env = _make_vec_env(env)
    model = PPO("MlpPolicy", vec_env, verbose=0, seed=seed, n_steps=1024, batch_size=256)

    reward_logger = _make_reward_logger_callback(algorithm)
    output_dir = Path(output_dir)
    model_path = output_dir / f"{algorithm}_sumo_traffic"

    try:
        model.learn(total_timesteps=total_timesteps, callback=reward_logger)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    finally:
        vec_env.close()

    return model, reward_logger.history, model_path.with_suffix(".zip")


def _policy_name(policy) -> str:
    policy_name = getattr(policy, "policy_name", None)
    if policy_name:
        return str(policy_name)

    if hasattr(policy, "predict"):
        return policy.__class__.__name__
    return getattr(policy, "__name__", policy.__class__.__name__)


def _actions_from_policy(
    policy,
    env,
    observations,
    step_index: int,
    infos: dict[str, dict] | None = None,
) -> dict[str, int]:
    if hasattr(policy, "act") and callable(policy.act):
        try:
            return policy.act(observations, infos=infos, deterministic=True)
        except TypeError:
            return policy.act(observations, deterministic=True)

    if hasattr(policy, "predict"):
        actions = {}
        for agent in env.agents:
            action, _ = policy.predict(observations[agent], deterministic=True)
            actions[agent] = int(action)
        return actions

    actions = policy(env, step_index)
    if isinstance(actions, dict):
        return {agent: int(actions.get(agent, 0)) for agent in env.agents}

    return {agent: int(actions) for agent in env.agents}


def evaluate_policy(
    config_file,
    traffic_light_ids,
    policy,
    episodes=3,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
):
    """Evaluate an SB3 model or a baseline action helper."""

    from marl_tsc.traffic_env import SumoTrafficEnv

    env_options = dict(env_kwargs or {})
    episode_rewards = []
    episode_queues = []
    episode_switches = []
    total_completed_steps = 0

    for episode_index in range(episodes):
        env = SumoTrafficEnv(
            config_file,
            possible_agents=traffic_light_ids,
            max_steps=max_steps,
            seed=seed + episode_index,
            **env_options,
        )
        observations, infos = env.reset(seed=seed + episode_index)

        episode_reward = 0.0
        queue_sum = 0.0
        queue_count = 0
        switch_count = 0
        completed_steps = 0

        try:
            for step_index in range(max_steps):
                if not env.agents:
                    break

                actions = _actions_from_policy(policy, env, observations, step_index, infos=infos)
                observations, rewards, _, truncations, infos = env.step(actions)

                episode_reward += float(sum(rewards.values()))
                for info in infos.values():
                    queue_sum += float(info.get("mean_local_queue", 0.0))
                    queue_count += 1
                    switch_count += 1 if info.get("switched") else 0

                completed_steps += 1
                if all(truncations.values()):
                    break
        finally:
            env.close()

        episode_rewards.append(episode_reward)
        episode_queues.append(queue_sum / queue_count if queue_count else 0.0)
        episode_switches.append(switch_count)
        total_completed_steps += completed_steps

    return {
        "policy_name": _policy_name(policy),
        "episodes": episodes,
        "mean_total_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_local_queue": float(np.mean(episode_queues)) if episode_queues else 0.0,
        "total_completed_steps": int(total_completed_steps),
        "mean_switches_per_episode": float(np.mean(episode_switches)) if episode_switches else 0.0,
    }


def plot_training_histories(histories):
    """Plot training reward curves for one or more algorithms."""

    import matplotlib.pyplot as plt

    grouped = defaultdict(list)
    for record in histories:
        grouped[str(record["algorithm"])].append(record)

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm, records in grouped.items():
        records = sorted(records, key=lambda item: item["timestep"])
        timesteps = [item["timestep"] for item in records]
        rewards = [item["mean_training_reward"] for item in records]
        ax.plot(timesteps, rewards, label=algorithm.upper())

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel("Mean training reward")
    ax.set_title("Training convergence")
    if grouped:
        ax.legend()
    fig.tight_layout()
    return fig, ax
