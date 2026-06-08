"""Training and evaluation helpers for the notebook.

This module provides utilities to bridge the PettingZoo environment with 
Stable Baselines3 (SB3). It includes wrappers for vectorized environments, 
custom logging callbacks, and centralized evaluation logic that supports 
both RL models and heuristic baselines.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any
import numpy as np


def _make_vec_env(env):
    """
    Converts a PettingZoo ParallelEnv into a Stable Baselines3 VecEnv.
    
    This uses SuperSuit to:
    1. Convert the multi-agent dict-based API to a single-agent-style vector API.
    2. Concatenate agents into a single batch (parameter sharing).
    3. Ensure compatibility with SB3's expectation of a DummyVecEnv/SubprocVecEnv.
    """
    import supersuit as ss

    # Convert PettingZoo Parallel environment to a VecEnv
    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    # Wrap it so it looks like a standard SB3 VecEnv with 1 "logical" environment
    vec_env = ss.concat_vec_envs_v1(vec_env, 1, num_cpus=1, base_class="stable_baselines3")

    # Patch the environment to support the 'seed' method if missing
    target_env = vec_env.venv if hasattr(vec_env, "venv") else vec_env
    # SB3 environments often require a .seed() method which SuperSuit might not expose directly
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
    """Factory for a Stable Baselines3 callback that logs performance metrics."""
    from stable_baselines3.common.callbacks import BaseCallback
    from collections import deque

    class RewardLoggerCallback(BaseCallback):
        """Logs per-step rewards and computes moving average returns during training."""
        def __init__(self):
            super().__init__()
            # Stores metrics for later plotting
            self.history: list[dict[str, Any]] = []
            # Tracks last 100 episode returns for a smoothed performance view
            self.episode_returns = deque(maxlen=100)
            # Accumulator for returns across the vectorized environments
            self.current_returns = None
            self.completed_episodes = 0

        def _on_step(self):
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")

            if self.current_returns is None and rewards is not None:
                self.current_returns = np.zeros(len(rewards))

            # Update running returns and check for finished episodes
            if rewards is not None and dones is not None:
                self.current_returns += rewards
                for i, done in enumerate(dones):
                    if done:
                        # Episode finished: record total return and reset accumulator
                        episode_ret = float(self.current_returns[i])
                        self.episode_returns.append(episode_ret)
                        print(f"[{algorithm_name.upper()}] Step {self.num_timesteps} | Episode Return: {episode_ret:.2f}")
                        self.current_returns[i] = 0.0

            if rewards is not None:
                reward_values = [float(reward) for reward in rewards]
                mean_reward = sum(reward_values) / len(reward_values) if reward_values else 0.0
                self.history.append(
                    {
                        "algorithm": algorithm_name,
                        "timestep": int(self.num_timesteps),
                        "mean_training_reward": mean_reward,
                        "moving_avg_episode_return": float(np.mean(self.episode_returns)) if self.episode_returns else 0.0,
                        "episodes_completed": self.completed_episodes,
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
    """Sets up the environment and trains a PPO model using parameter sharing."""

    from stable_baselines3 import PPO
    from marl_tsc.traffic_env import SumoTrafficEnv

    # Used for logging and file naming
    algorithm = "ppo"

    env_options = dict(env_kwargs or {})
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )

    # Wrap the PettingZoo env for SB3
    vec_env = _make_vec_env(env)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        learning_rate=1e-4,
        n_steps=512,
        batch_size=128,
        ent_coef=0.005,
    )

    # Setup training monitoring
    reward_logger = _make_reward_logger_callback(algorithm)
    output_dir = Path(output_dir)
    model_path = output_dir / f"{algorithm}_sumo_traffic"

    try:
        model.learn(total_timesteps=total_timesteps, callback=reward_logger)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    finally:
        # Ensure SUMO processes are closed even if training crashes
        vec_env.close()

    return model, reward_logger.history, model_path.with_suffix(".zip")


def _policy_name(policy) -> str:
    """Helper to extract a string identifier for different policy types."""
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
    """
    Dispatches observation data to the provided policy and returns agent actions.
    
    Supports:
    1. Custom class instances with an .act() method (e.g., MappoModel).
    2. Stable Baselines3 models via .predict().
    3. Functional baselines (e.g., random_actions).
    """
    # Case 1: Custom MAPPO-style models
    if hasattr(policy, "act") and callable(policy.act):
        try:
            return policy.act(observations, infos=infos, deterministic=True)
        except TypeError:
            return policy.act(observations, deterministic=True)

    # Case 2: SB3 Models (Individual predictions per agent)
    if hasattr(policy, "predict"):
        actions = {}
        for agent in env.agents:
            action, _ = policy.predict(observations[agent], deterministic=True)
            actions[agent] = int(action)
        return actions

    # Case 3: Baseline functions (random, fixed-time)
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
    """
    Evaluates a policy over multiple episodes and returns performance metrics.
    
    Collects reward, queue, switching, arrivals, waiting-time, and time-loss
    metrics so policies cannot look good by optimizing only one queue average.
    """

    from marl_tsc.traffic_env import SumoTrafficEnv

    env_options = dict(env_kwargs or {})
    episode_rewards = []
    episode_queues = []
    episode_max_queues = []
    episode_switches = []
    episode_arrivals = []
    episode_waiting_times = []
    episode_time_losses = []
    episode_vehicle_counts = []
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
        max_queue = 0.0
        switch_count = 0
        arrived_vehicles = 0
        waiting_time_sum = 0.0
        time_loss_sum = 0.0
        vehicle_count_sum = 0.0
        global_metric_count = 0
        completed_steps = 0

        try:
            for step_index in range(max_steps):
                if not env.agents:
                    break

                actions = _actions_from_policy(policy, env, observations, step_index, infos=infos)
                observations, rewards, _, truncations, infos = env.step(actions)

                episode_reward += float(sum(rewards.values()))
                # Aggregate metrics from the info dict provided by SumoTrafficEnv
                for info in infos.values():
                    queue_sum += float(info.get("mean_local_queue", 0.0))
                    queue_count += 1
                    max_queue = max(max_queue, float(info.get("max_local_queue", 0.0)))
                    switch_count += 1 if info.get("switched") else 0

                first_info = next(iter(infos.values()), None)
                if first_info:
                    arrived_vehicles += int(first_info.get("arrived_vehicles", 0))
                    waiting_time_sum += float(first_info.get("mean_waiting_time", 0.0))
                    time_loss_sum += float(first_info.get("total_time_loss", 0.0))
                    vehicle_count_sum += float(first_info.get("vehicle_count", 0.0))
                    global_metric_count += 1

                completed_steps += 1
                if all(truncations.values()):
                    break
        finally:
            # Shutdown SUMO for this episode
            env.close()

        print(f"Finished Evaluation Episode {episode_index + 1}/{episodes} - Total Reward: {episode_reward:.2f}")
        episode_rewards.append(episode_reward)
        episode_queues.append(queue_sum / queue_count if queue_count else 0.0)
        episode_max_queues.append(max_queue)
        episode_switches.append(switch_count)
        episode_arrivals.append(arrived_vehicles)
        episode_waiting_times.append(waiting_time_sum / global_metric_count if global_metric_count else 0.0)
        episode_time_losses.append(time_loss_sum / global_metric_count if global_metric_count else 0.0)
        episode_vehicle_counts.append(vehicle_count_sum / global_metric_count if global_metric_count else 0.0)
        total_completed_steps += completed_steps

    return {
        "policy_name": _policy_name(policy),
        "episodes": episodes,
        "mean_total_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_local_queue": float(np.mean(episode_queues)) if episode_queues else 0.0,
        "mean_max_queue": float(np.mean(episode_max_queues)) if episode_max_queues else 0.0,
        "total_completed_steps": int(total_completed_steps),
        "mean_switches_per_episode": float(np.mean(episode_switches)) if episode_switches else 0.0,
        "mean_arrived_vehicles_per_episode": float(np.mean(episode_arrivals)) if episode_arrivals else 0.0,
        "mean_waiting_time": float(np.mean(episode_waiting_times)) if episode_waiting_times else 0.0,
        "mean_total_time_loss": float(np.mean(episode_time_losses)) if episode_time_losses else 0.0,
        "mean_vehicle_count": float(np.mean(episode_vehicle_counts)) if episode_vehicle_counts else 0.0,
    }


def plot_training_histories(histories):
    """Plot training reward curves for one or more algorithms."""

    import matplotlib.pyplot as plt

    # Organize history by algorithm for multiple lines on one plot
    grouped = defaultdict(list)
    for record in histories:
        grouped[str(record["algorithm"])].append(record)

    fig, ax = plt.subplots(figsize=(8, 5))

    for algorithm, records in grouped.items():
        records = sorted(records, key=lambda item: item["timestep"])
        episode_records = [
            item
            for item in records
            if float(item.get("moving_avg_episode_return", 0.0)) != 0.0
        ]
        if episode_records:
            timesteps = [item["timestep"] for item in episode_records]
            rewards = [item["moving_avg_episode_return"] for item in episode_records]
            ylabel = "Moving average episode return"
        else:
            timesteps = [item["timestep"] for item in records]
            rewards = [item["mean_training_reward"] for item in records]
            ylabel = "Mean training reward"
        ax.plot(timesteps, rewards, label=algorithm.upper())

    ax.set_xlabel("Training timesteps")
    ax.set_ylabel(ylabel if grouped else "Reward")
    ax.set_title("Training convergence")
    if grouped:
        ax.legend()
    fig.tight_layout()
    return fig, ax

def plot_moving_average_histories(histories, window=100):
    """Plot training reward curves smoothed by a moving average window.
    
    This handles noisy step-rewards by showing the trend over time.
    """
    import matplotlib.pyplot as plt
    import pandas as pd

    if not histories:
        return None, None

    df = pd.DataFrame(histories)
    fig, ax = plt.subplots(figsize=(10, 6))

    for algorithm in df['algorithm'].unique():
        algo_df = df[df['algorithm'] == algorithm].sort_values('timestep')
        
        # Prioritize episode return if available, else use step reward
        metric = "moving_avg_episode_return" if "moving_avg_episode_return" in algo_df.columns else "mean_training_reward"
        smoothed = algo_df[metric].rolling(window=window, min_periods=1).mean()
        
        line, = ax.plot(algo_df['timestep'], smoothed, label=f"{algorithm.upper()} (Smooth)")
        ax.plot(algo_df['timestep'], algo_df[metric], alpha=0.15, color=line.get_color())

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Performance ({window}-pt Moving Average)")
    ax.legend()
    fig.tight_layout()
    return fig, ax
