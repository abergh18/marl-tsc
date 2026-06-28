"""
train_graph_mappo.py

Training entry point for Graph MAPPO.

Typical usage:

    model, history, model_path = train_graph_mappo(
        config_file=paths.config_file,
        network_file=paths.network_file,
        traffic_light_ids=traffic_light_ids,
        output_dir=OUTPUT_DIR,
        total_timesteps=100_000,
    )
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.optim

from marl_tsc.graph_based.graph_env import GraphTrafficEnv
from marl_tsc.graph_based.graph_mappo_trainer import GraphMAPPOTrainer
from marl_tsc.graph_based.models.policy_factory import (
    build_default_graph_policy,
)
from marl_tsc.graph_based.run_training import run_training


def train_graph_mappo(
    config_file,
    network_file,
    traffic_light_ids,
    output_dir,
    total_timesteps=100_000,
    rollout_steps=64,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
    clip_ratio=0.2,
    entropy_coef=0.01,
    value_coef=0.5,
    learning_rate=3e-4,
    update_epochs=3,
):
    """
    Train a graph-based MAPPO policy.

    Parameters
    ----------
    config_file : str or Path
        Path to SUMO config file.
    network_file : str or Path
        Path to SUMO network file (.net.xml).
    traffic_light_ids : list[str]
        List of traffic light agent IDs.
    output_dir : str or Path
        Directory to save the model.
    total_timesteps : int
        Total environment steps to train for.
    rollout_steps : int
        Steps per rollout batch.
    max_steps : int
        Maximum steps per episode.
    seed : int
        Random seed.
    env_kwargs : dict, optional
        Additional environment configuration.
    clip_ratio : float
        PPO clip ratio (epsilon).
    entropy_coef : float
        Entropy regularization coefficient.
    value_coef : float
        Value loss coefficient.
    learning_rate : float
        Optimizer learning rate.
    update_epochs : int
        Number of update epochs per rollout.

    Returns
    -------
    model, history, model_path
        Trained policy, training history, and saved model path.
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    env = GraphTrafficEnv(
        config_file=config_file,
        network_file=network_file,
        possible_agents=traffic_light_ids,
    )

    graph_obs, infos = env.reset()
    obs_dim = env.obs_dim
    agent_ids = env.agent_ids
    action_dim = int(env.action_spaces[agent_ids[0]].n)

    # Build policy
    policy = build_default_graph_policy(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=64,
    ).to(device)

    # Create optimizer
    optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=learning_rate,
    )

    # Create trainer
    trainer = GraphMAPPOTrainer(
        env=env,
        policy=policy,
        optimizer=optimizer,
        rollout_steps=rollout_steps,
        gae_lambda=0.95,
        gamma=0.99,
        clip_ratio=clip_ratio,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        max_grad_norm=0.5,
        update_epochs=update_epochs,
    )

    # Run training
    model, history, model_path = run_training(
        trainer=trainer,
        total_timesteps=total_timesteps,
        rollout_steps=rollout_steps,
        algorithm_name="graph_mappo",
        model_path=str(
            Path(output_dir) / "models" / "graph_mappo.pt"
        ),
        policy=policy,
    )

    return model, history, model_path
