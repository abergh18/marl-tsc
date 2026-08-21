"""
training.py

Training, evaluation, and plotting helpers for the MARL traffic environment.
Provides utilities for running stable-baselines3 algorithms and evaluating
custom policies with real-world traffic constraints.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
import random
import re
from typing import Any

import numpy as np


def _make_vec_env(env):
    """Convert a PettingZoo ParallelEnv into an SB3-compatible VecEnv."""
    import supersuit as ss

    vec_env = ss.pettingzoo_env_to_vec_env_v1(env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, 1, num_cpus=1, base_class="stable_baselines3"
    )

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
    """Factory for a Stable Baselines3 callback that logs performance metrics."""
    from stable_baselines3.common.callbacks import BaseCallback

    class RewardLoggerCallback(BaseCallback):
        """Logs per-step rewards and computes moving average returns."""

        def __init__(self):
            super().__init__()
            self.history: list[dict[str, Any]] = []
            self.episode_returns = deque(maxlen=100)
            self.current_returns = None
            self.completed_episodes = 0

        def _on_step(self):
            rewards = self.locals.get("rewards")
            dones = self.locals.get("dones")

            if self.current_returns is None and rewards is not None:
                self.current_returns = np.zeros(len(rewards))

            if rewards is not None and dones is not None:
                self.current_returns += rewards
                done_indexes = [i for i, done in enumerate(dones) if done]
                if done_indexes:
                    completed_returns = [
                        float(self.current_returns[i]) for i in done_indexes
                    ]
                    episode_return = float(np.mean(completed_returns))
                    self.episode_returns.append(episode_return)
                    self.completed_episodes += 1
                    print(
                        f"[{algorithm_name.upper()}] Step {self.num_timesteps} | "
                        f"Network Episode Return: {episode_return:.2f}"
                    )
                    for i in done_indexes:
                        self.current_returns[i] = 0.0

            if rewards is not None:
                reward_values = [float(reward) for reward in rewards]
                mean_reward = (
                    sum(reward_values) / len(reward_values) if reward_values else 0.0
                )
                self.history.append(
                    {
                        "algorithm": algorithm_name,
                        "timestep": int(self.num_timesteps),
                        "mean_training_reward": mean_reward,
                        "moving_avg_episode_return": (
                            float(np.mean(self.episode_returns))
                            if self.episode_returns
                            else 0.0
                        ),
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

    from marl_tsc.traffic_env import SumoTrafficEnv
    from stable_baselines3 import PPO

    algorithm = "ppo"

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )

    vec_env = _make_vec_env(env)
    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        seed=seed,
        learning_rate=1e-4,
        n_steps=1024,
        batch_size=512,
        ent_coef=0.01,
    )

    reward_logger = _make_reward_logger_callback(algorithm)
    output_dir = Path(output_dir)
    model_path = output_dir / f"{algorithm}_sumo_traffic"

    try:
        ppo_timesteps = total_timesteps * len(traffic_light_ids) # PPO treats each individual agent descision as a timestep
        model.learn(total_timesteps=ppo_timesteps, callback=reward_logger)
        output_dir.mkdir(parents=True, exist_ok=True)
        model.save(model_path)
    finally:
        vec_env.close()

    return model, reward_logger.history, model_path.with_suffix(".zip")


def _policy_name(policy) -> str:
    """Return a readable name for a policy object or function."""
    policy_name = getattr(policy, "policy_name", None)
    if policy_name:
        return str(policy_name)

    if hasattr(policy, "predict"):
        return policy.__class__.__name__
    return getattr(policy, "__name__", policy.__class__.__name__)


def _mean_or_zero(values) -> float:
    return float(np.mean(values)) if values else 0.0


def _actions_from_policy(
    policy,
    env,
    observations,
    step_index: int,
    infos: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """Ask a MAPPO model, SB3 model, or baseline function for actions."""
    if hasattr(policy, "act") and callable(policy.act):
        try:
            raw = policy.act(observations, infos=infos, deterministic=True)
        except TypeError:
            raw = policy.act(observations, deterministic=True)
        
        cleaned = {}
        for agent, action in raw.items():
            if isinstance(action, (list, tuple, np.ndarray)):
                cleaned[agent] = int(action[0])
            else:
                cleaned[agent] = int(action)
        return cleaned

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
    """Evaluate a policy over multiple episodes.

    Collects reward, queue, switching, arrivals, waiting-time, and time-loss
    metrics so policies cannot look good by optimising only one queue average.
    """

    from marl_tsc.traffic_env import SumoTrafficEnv

    env_options = dict(env_kwargs or {})
    env_options.setdefault("global_metric_interval", 10)
    # Do not penalize phase switching during evaluation
    env_options["switch_penalty"] = 0.0
    
    episode_rewards = []
    episode_queues = []
    episode_max_queues = []
    episode_switches = []
    episode_arrivals = []
    episode_waiting_times = []
    episode_max_waiting_times = []
    episode_time_losses = []
    episode_vehicle_counts = []
    episode_details = []
    total_completed_steps = 0

    for episode_index in range(episodes):
        random.seed(seed + episode_index)
        np.random.seed(seed + episode_index)
        
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
        max_waiting_time = 0.0
        time_loss_sum = 0.0
        vehicle_count_sum = 0.0
        global_metric_count = 0
        completed_steps = 0

        try:
            for step_index in range(max_steps):
                if not env.agents:
                    break

                actions = _actions_from_policy(
                    policy, env, observations, step_index, infos=infos
                )
                
                observations, rewards, _, truncations, infos = env.step(actions)

                episode_reward += float(sum(
                    infos.get(agent, {}).get("raw_traffic_reward", r)
                    for agent, r in rewards.items()
                ))
                
                for info in infos.values():
                    queue_sum += float(info.get("mean_local_queue", 0.0))
                    queue_count += 1
                    max_queue = max(max_queue, float(info.get("max_local_queue", 0.0)))
                    max_waiting_time = max(
                        max_waiting_time,
                        float(info.get("max_waiting_time", 0.0)),
                    )
                    switch_count += 1 if info.get("switched") else 0

                first_info = next(iter(infos.values()), None)
                if first_info:
                    arrived_vehicles += int(first_info.get("arrived_vehicles", 0))
                    if first_info.get("global_metrics_updated", True):
                        waiting_time_sum += float(first_info.get("mean_waiting_time", 0.0))
                        time_loss_sum += float(first_info.get("total_time_loss", 0.0))
                        vehicle_count_sum += float(first_info.get("vehicle_count", 0.0))
                        global_metric_count += 1

                completed_steps += 1
                if all(truncations.values()):
                    break
        finally:
            env.close()

        print(
            f"Finished Evaluation Episode {episode_index + 1}/{episodes} - "
            f"Total Reward: {episode_reward:.2f}"
        )
        mean_local_queue = queue_sum / queue_count if queue_count else 0.0
        mean_waiting_time = waiting_time_sum / global_metric_count if global_metric_count else 0.0
        mean_total_time_loss = time_loss_sum / global_metric_count if global_metric_count else 0.0
        mean_vehicle_count = vehicle_count_sum / global_metric_count if global_metric_count else 0.0

        episode_rewards.append(episode_reward)
        episode_queues.append(mean_local_queue)
        episode_max_queues.append(max_queue)
        episode_switches.append(switch_count)
        episode_arrivals.append(arrived_vehicles)
        episode_waiting_times.append(mean_waiting_time)
        episode_max_waiting_times.append(max_waiting_time)
        episode_time_losses.append(mean_total_time_loss)
        episode_vehicle_counts.append(mean_vehicle_count)
        episode_details.append(
            {
                "episode": episode_index + 1,
                "total_reward": episode_reward,
                "mean_local_queue": mean_local_queue,
                "max_queue": max_queue,
                "switches": switch_count,
                "arrived_vehicles": arrived_vehicles,
                "mean_waiting_time": mean_waiting_time,
                "max_waiting_time": max_waiting_time,
                "mean_total_time_loss": mean_total_time_loss,
                "mean_vehicle_count": mean_vehicle_count,
                "completed_steps": completed_steps,
            }
        )
        total_completed_steps += completed_steps

    return {
        "policy_name": _policy_name(policy),
        "episodes": episodes,
        "mean_total_reward": _mean_or_zero(episode_rewards),
        "mean_local_queue": _mean_or_zero(episode_queues),
        "mean_max_queue": _mean_or_zero(episode_max_queues),
        "total_completed_steps": int(total_completed_steps),
        "mean_switches_per_episode": _mean_or_zero(episode_switches),
        "mean_arrived_vehicles_per_episode": _mean_or_zero(episode_arrivals),
        "mean_waiting_time": _mean_or_zero(episode_waiting_times),
        "mean_max_waiting_time": _mean_or_zero(episode_max_waiting_times),
        "mean_total_time_loss": _mean_or_zero(episode_time_losses),
        "mean_vehicle_count": _mean_or_zero(episode_vehicle_counts),
        "episode_details": episode_details,
    }


def evaluate_policies(
    config_file,
    traffic_light_ids,
    policies,
    episodes=3,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
):
    """Evaluate named policies with the same simulation settings and seeds."""

    results = {}
    for name, policy in policies.items():
        print(f"Evaluating {name}")
        results[name] = evaluate_policy(
            config_file=config_file,
            traffic_light_ids=traffic_light_ids,
            policy=policy,
            episodes=episodes,
            max_steps=max_steps,
            seed=seed,
            env_kwargs=env_kwargs,
        )
    return results


def evaluation_results_table(policy_results):
    """Return the average evaluation metrics as a pandas table."""

    import pandas as pd

    rows = [
        {
            "Policy": name,
            "Mean reward": result["mean_total_reward"],
            "Mean queue": result["mean_local_queue"],
            "Mean max queue": result["mean_max_queue"],
            "Mean wait": result["mean_waiting_time"],
            "Mean max wait": result["mean_max_waiting_time"],
            "Mean time loss": result["mean_total_time_loss"],
            "Mean arrivals": result["mean_arrived_vehicles_per_episode"],
            "Mean switches": result["mean_switches_per_episode"],
        }
        for name, result in policy_results.items()
    ]
    return pd.DataFrame(rows)


def export_policy_replay(
    config_file,
    traffic_light_ids,
    policy,
    output_dir,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
    replay_name="replay",
) -> Path:
    """Export one fixed policy rollout as SUMO files that can be opened in sumo-gui."""

    from marl_tsc.traffic_env import SumoTrafficEnv

    config_file = Path(config_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_text = config_file.read_text(encoding="utf-8")
    net_name = re.search(r'<net-file\s+value="([^"]+)"', config_text).group(1)
    route_names = re.search(r'<route-files\s+value="([^"]+)"', config_text).group(1)

    network_file = Path(net_name)
    if not network_file.is_absolute():
        network_file = config_file.parent / network_file

    route_files = []
    for route_name in route_names.split(","):
        route_file = Path(route_name.strip())
        if not route_file.is_absolute():
            route_file = config_file.parent / route_file
        route_files.append(route_file)

    replay_network_file = output_dir / f"{replay_name}.net.xml"
    replay_config_file = output_dir / f"{replay_name}.sumocfg"

    env_options = dict(env_kwargs or {})
    env_options.pop("render_mode", None)
    env = SumoTrafficEnv(
        config_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        render_mode=None,
        **env_options,
    )

    replay_phases: dict[str, list[tuple[float, str]]] = {}
    completed_steps = 0
    seconds_per_action = 0.0

    try:
        observations, infos = env.reset(seed=seed)
        traci = env._import_traci()
        agent_ids = tuple(traffic_light_ids or env.agents)
        seconds_per_action = float(env.seconds_per_action)

        for agent_id in agent_ids:
            replay_phases[agent_id] = []

        for step_index in range(max_steps):
            if not env.agents:
                break

            actions = _actions_from_policy(policy, env, observations, step_index, infos=infos)
            observations, _, _, truncations, infos = env.step(actions)
            completed_steps += 1

            for agent_id in agent_ids:
                state = traci.trafficlight.getRedYellowGreenState(agent_id)
                phases = replay_phases[agent_id]
                if phases and phases[-1][1] == state:
                    phases[-1] = (phases[-1][0] + seconds_per_action, state)
                else:
                    phases.append((seconds_per_action, state))

            if truncations and all(truncations.values()):
                break
    finally:
        env.close()

    if completed_steps == 0:
        raise RuntimeError("Policy replay export did not complete any simulation steps.")

    net_text = network_file.read_text(encoding="utf-8")
    for agent_id, phases in replay_phases.items():
        phase_lines = "\n".join(
            f'        <phase duration="{duration:g}" state="{state}"/>'
            for duration, state in phases
        )
        replay_logic = (
            f'    <tlLogic id="{agent_id}" type="static" programID="0" offset="0">\n'
            f"{phase_lines}\n"
            f"    </tlLogic>"
        )
        pattern = rf'    <tlLogic id="{re.escape(agent_id)}"[\s\S]*?    </tlLogic>'
        net_text = re.sub(pattern, replay_logic, net_text, count=1)

    replay_network_file.write_text(net_text, encoding="utf-8")

    route_value = ",".join(str(route_file) for route_file in route_files)
    replay_config_file.write_text(
        f"""<configuration>
    <input>
        <net-file value="{replay_network_file.name}"/>
        <route-files value="{route_value}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="{completed_steps * seconds_per_action:g}"/>
    </time>
</configuration>
""",
        encoding="utf-8",
    )

    return replay_config_file


def plot_training_histories(histories):
    """Plot training reward curves for one or more algorithms.

    ``histories`` may be one combined history list or a mapping from display
    names to individual history lists.
    """

    import matplotlib.pyplot as plt

    if isinstance(histories, dict):
        grouped = histories
    else:
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

    for algorithm in df["algorithm"].unique():
        algo_df = df[df["algorithm"] == algorithm].sort_values("timestep")
        metric = (
            "moving_avg_episode_return"
            if "moving_avg_episode_return" in algo_df.columns
            else "mean_training_reward"
        )
        smoothed = algo_df[metric].rolling(window=window, min_periods=1).mean()

        (line,) = ax.plot(
            algo_df["timestep"], smoothed, label=f"{algorithm.upper()} (Smooth)"
        )
        ax.plot(
            algo_df["timestep"], algo_df[metric], alpha=0.15, color=line.get_color()
        )

    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Reward")
    ax.set_title(f"Training Performance ({window}-pt Moving Average)")
    ax.legend()
    fig.tight_layout()
    return fig, ax
