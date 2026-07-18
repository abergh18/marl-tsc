"""
train_hetero_mappo.py  — v2

Entry point for training a heterogeneous graph MAPPO policy.

Changes from v1
---------------
- max_hops parameter added and threaded through to HeteroGraphEnv.
- CONNECTION_FEAT_DIM updated to 7 (from hetero_graph_builder v2).
- intersection_obs_dim probe tidied — done once via raw SumoTrafficEnv
  before constructing HeteroGraphEnv, avoiding double SUMO start.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.optim

from marl_tsc.graph_based.heterogene.hetero_graph_env import HeteroGraphEnv
from marl_tsc.graph_based.models.hetero_graph_mappo_policy import HeteroGraphMAPPOPolicy
from marl_tsc.graph_based.heterogene.hetero_graph_builder import CONNECTION_FEAT_DIM
from marl_tsc.graph_based.true_mappo_trainer import TrueMAPPOTrainer
from marl_tsc.graph_based.run_training import run_training
from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder
from marl_tsc.traffic_env import SumoTrafficEnv


def _build_hetero_mappo_policy(
    intersection_obs_dim: int,
    action_dim: int,
    global_state_dim: int,
    shared_dim: int = 64,
    critic_hidden_dim: int = 128,
) -> HeteroGraphMAPPOPolicy:
    encoder = GATEncoder(
        obs_dim=shared_dim,
        hidden_dim=shared_dim,
    )
    actor_head = torch.nn.Linear(shared_dim, action_dim)
    centralised_critic = CentralisedCritic(
        global_state_dim=global_state_dim,
        hidden_dim=critic_hidden_dim,
    )
    return HeteroGraphMAPPOPolicy(
        encoder=encoder,
        actor_head=actor_head,
        centralised_critic=centralised_critic,
        intersection_obs_dim=intersection_obs_dim,
        connection_feat_dim=CONNECTION_FEAT_DIM,
        shared_dim=shared_dim,
    )


def train_hetero_mappo(
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
    shared_dim=64,
    critic_hidden_dim=128,
    max_hops=3,
):
    """
    Train a heterogeneous graph MAPPO policy.

    Parameters
    ----------
    max_hops : int
        BFS depth for connection node discovery in HeteroGraphBuilder.
        Controls how many road segments are traversed when searching for
        TLS-to-TLS connections. Higher values capture more distant
        connections but increase graph complexity.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    env = None

    try:
        # ── Probe obs_dim via raw SumoTrafficEnv — avoids double SUMO start
        _probe = SumoTrafficEnv(
            config_file,
            possible_agents=traffic_light_ids,
            **env_options,
        )
        _obs, _ = _probe.reset()
        intersection_obs_dim = len(next(iter(_obs.values())))
        _probe.close()
        print(f"Observation dim: {intersection_obs_dim}")
        print(f"Connection feat dim: {CONNECTION_FEAT_DIM}")
        print(f"Max hops: {max_hops}")

        # ── Build environment ─────────────────────────────────────────────
        env = HeteroGraphEnv(
            config_file=config_file,
            network_file=network_file,
            possible_agents=traffic_light_ids,
            intersection_obs_dim=intersection_obs_dim,
            shared_dim=shared_dim,
            max_hops=max_hops,
            **env_options,
        )

        graph_obs, _     = env.reset()
        agent_ids        = env.agent_ids
        action_dim       = int(env.action_spaces[agent_ids[0]].n)
        global_state_dim = env.global_state_dim

        print(f"Agents: {len(agent_ids)}")
        print(f"Action dim: {action_dim}")
        print(f"Connection nodes: {env.topology.num_connections}")
        print(f"Total graph nodes: {env.topology.num_intersections + env.topology.num_connections}")

        # ── Policy ────────────────────────────────────────────────────────
        policy = _build_hetero_mappo_policy(
            intersection_obs_dim=intersection_obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            shared_dim=shared_dim,
            critic_hidden_dim=critic_hidden_dim,
        ).to(device)

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )

        # ── Trainer ───────────────────────────────────────────────────────
        trainer = TrueMAPPOTrainer(
            env=env,
            policy=policy,
            optimizer=optimizer,
            rollout_steps=rollout_steps,
            gae_lambda=0.95,
            gamma=0.99,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            gifting_entropy_coef=0.0,
            value_coef=value_coef,
            max_grad_norm=0.5,
            update_epochs=update_epochs,
        )

        # ── Train ─────────────────────────────────────────────────────────
        model, history, model_path = run_training(
            trainer=trainer,
            total_timesteps=total_timesteps,
            rollout_steps=rollout_steps,
            algorithm_name="hetero_mappo",
            model_path=str(
                Path(output_dir) / "models" / "hetero_mappo.pt"
            ),
        )

        return model, history, model_path

    finally:
        if env is not None:
            env.close()