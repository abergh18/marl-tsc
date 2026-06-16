"""
train_graph_ctde.py

High-level training entrypoint for GAT-based CTDE.

This module assembles:

    GraphTrafficEnv
    GraphPolicy
    GraphCTDETrainer

and delegates training execution to:

    run_training()

The learning algorithm itself remains inside GraphCTDETrainer.
"""

from __future__ import annotations

from pathlib import Path

import torch

from marl_tsc.graph_based.graph_env import (
    GraphTrafficEnv,
)

from marl_tsc.graph_based.graph_ctde_trainer import (
    GraphCTDETrainer,
)

from marl_tsc.graph_based.models.policy_factory import (
    build_default_graph_policy,
)

from marl_tsc.graph_based.run_training import (
    run_training,
)


def train_graph_ctde(
    config_file,
    network_file,
    traffic_light_ids,
    output_dir,
    total_timesteps,
    rollout_steps=1024,
    max_steps=1000,
    seed=42,
    learning_rate=1e-3,
    hidden_dim=64,
    #gamma=0.99,
    gae_lambda=0.95,
    env_kwargs=None,
):
    """
    Train a graph-based CTDE actor-critic.

    Returns
    -------
    policy
        Trained GraphPolicy

    history
        Training history records

    model_path
        Saved model path
    """

    env_options = dict(env_kwargs or {})

    env = GraphTrafficEnv(
        config_file=config_file,
        network_file=network_file,
        possible_agents=traffic_light_ids,
        max_steps=max_steps,
        seed=seed,
        **env_options,
    )

    try:

        #
        # Infer observation dimension
        #

        graph_obs, _ = env.reset(
            seed=seed
        )

        obs_dim = (
            graph_obs.graph.x.shape[1]
        )

        #
        # Infer action dimension
        #

        first_agent = (
            traffic_light_ids[0]
        )

        action_dim = (
            env.action_spaces[first_agent].n
        )

        #
        # Build policy
        #

        policy = (
            build_default_graph_policy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
            )
        )

        #
        # Optimizer
        #

        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )

        #
        # Trainer
        #

        trainer = GraphCTDETrainer(
            env=env,
            policy=policy,
            optimizer=optimizer,
            rollout_steps=rollout_steps,
            #gamma=gamma,
            gae_lambda=gae_lambda,
        )

        #
        # Save location
        #

        model_path = (
            Path(output_dir)
            / "graph_ctde.pt"
        )

        #
        # Train
        #

        return run_training(
            trainer=trainer,
            total_timesteps=total_timesteps,
            rollout_steps=rollout_steps,
            algorithm_name="graph_ctde",
            model_path=model_path,
        )

    finally:

        env.close()