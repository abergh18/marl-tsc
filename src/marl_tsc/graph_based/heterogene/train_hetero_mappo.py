"""
train_hetero_mappo.py  — v3

Entry point for training a heterogeneous graph MAPPO policy,
with optional reward sharing (zero_sum or public_goods).

Changes from v2
---------------
- reward_sharing parameter added: None | "zero_sum" | "public_goods"
- gifting_divisions parameter added.
- SumoTrafficEnv constructed first, optionally wrapped, then passed
  as sumo_env= to HeteroGraphEnv — prevents double SUMO TraCI start.
- HeteroGiftingMAPPOPolicy used when reward sharing is active.
- GiftingGraphRunner used when reward sharing is active.
- obs_dim read from sumo_env reset before wrapping.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.optim

from marl_tsc.graph_based.heterogene.hetero_graph_env import HeteroGraphEnv
from marl_tsc.graph_based.models.hetero_graph_mappo_policy import HeteroGraphMAPPOPolicy
from marl_tsc.graph_based.models.hetero_gifting_mappo_policy import HeteroGiftingMAPPOPolicy
from marl_tsc.graph_based.heterogene.hetero_graph_builder import CONNECTION_FEAT_DIM
from marl_tsc.graph_based.true_mappo_trainer import TrueMAPPOTrainer
from marl_tsc.graph_based.run_training import run_training
from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder
from marl_tsc.graph_based.gifting_graph_runner import GiftingGraphRunner
from marl_tsc.traffic_env import SumoTrafficEnv
from marl_tsc.wrappers import ZeroSumRewardWrapper, PeerRewardingWrapper


def _build_hetero_mappo_policy(
    intersection_obs_dim: int,
    action_dim: int,
    global_state_dim: int,
    shared_dim: int = 64,
    critic_hidden_dim: int = 256,
) -> HeteroGraphMAPPOPolicy:
    """Standard het MAPPO policy — no gifting branch."""
    encoder = GATEncoder(obs_dim=shared_dim, hidden_dim=shared_dim)
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


def _build_hetero_gifting_policy(
    intersection_obs_dim: int,
    action_dim: int,
    num_divisions: int,
    global_state_dim: int,
    shared_dim: int = 64,
    critic_hidden_dim: int = 256,
) -> HeteroGiftingMAPPOPolicy:
    """Het MAPPO policy with traffic + gifting actor branches."""
    encoder = GATEncoder(obs_dim=shared_dim, hidden_dim=shared_dim)
    centralised_critic = CentralisedCritic(
        global_state_dim=global_state_dim,
        hidden_dim=critic_hidden_dim,
    )
    return HeteroGiftingMAPPOPolicy(
        encoder=encoder,
        action_dim=action_dim,
        num_divisions=num_divisions,
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
    rollout_steps=128,
    max_steps=1000,
    seed=42,
    env_kwargs=None,
    clip_ratio=0.2,
    entropy_coef=0.01,
    gifting_entropy_coef=0.01,
    value_coef=0.5,
    learning_rate=1e-4,
    update_epochs=4,
    shared_dim=64,
    critic_hidden_dim=256,
    max_hops=3,
    reward_sharing=None,
    gifting_divisions=None,
):
    """
    Train a heterogeneous graph MAPPO policy.

    Parameters
    ----------
    reward_sharing : None | "zero_sum" | "public_goods"
        Reward sharing mechanic. None = standard MAPPO.
    gifting_divisions : int, optional
        Discrete gifting portions. Defaults to num_agents-1 for
        zero_sum, 10 for public_goods.
    max_hops : int
        BFS depth for connection node discovery.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    env = None

    try:
        # ── Build base SumoTrafficEnv and read obs_dim ────────────────────
        sumo_env = SumoTrafficEnv(
            config_file,
            possible_agents=traffic_light_ids,
            **env_options,
        )
        _obs, _ = sumo_env.reset()
        intersection_obs_dim = len(next(iter(_obs.values())))

        # ── Optionally wrap with reward sharing ───────────────────────────
        num_agents = len(traffic_light_ids)

        if reward_sharing == "zero_sum":
            divisions      = gifting_divisions or max(1, num_agents - 1)
            wrapped_sumo   = ZeroSumRewardWrapper(sumo_env, division=divisions)
            algorithm_name = "hetero_mappo_zero_sum"
        elif reward_sharing == "public_goods":
            divisions      = gifting_divisions or 10
            wrapped_sumo   = PeerRewardingWrapper(sumo_env, division=divisions)
            algorithm_name = "hetero_mappo_public_goods"
        else:
            wrapped_sumo   = sumo_env
            divisions      = None
            algorithm_name = "hetero_mappo"

        # ── Build HeteroGraphEnv ──────────────────────────────────────────
        env = HeteroGraphEnv(
            config_file=config_file,
            network_file=network_file,
            possible_agents=traffic_light_ids,
            intersection_obs_dim=intersection_obs_dim,
            shared_dim=shared_dim,
            max_hops=max_hops,
            sumo_env=wrapped_sumo,
        )

        graph_obs, _     = env.reset()
        agent_ids        = env.agent_ids
        global_state_dim = env.global_state_dim

        # action_dim from traffic portion only
        raw_space  = env.action_spaces[agent_ids[0]]
        action_dim = int(
            raw_space.nvec[0]
            if hasattr(raw_space, 'nvec')
            else raw_space.n
        )

        print(f"Algorithm       : {algorithm_name}")
        print(f"Observation dim : {intersection_obs_dim}")
        print(f"Action dim      : {action_dim}")
        print(f"Connection nodes: {env.topology.num_connections}")
        print(f"Total nodes     : {env.topology.num_intersections + env.topology.num_connections}")
        print(f"Max hops        : {max_hops}")
        print(f"Reward sharing  : {reward_sharing or 'none'}")
        if divisions:
            print(f"Gifting divs    : {divisions}")

        # ── Policy ────────────────────────────────────────────────────────
        if reward_sharing is not None:
            policy = _build_hetero_gifting_policy(
                intersection_obs_dim=intersection_obs_dim,
                action_dim=action_dim,
                num_divisions=divisions,
                global_state_dim=global_state_dim,
                shared_dim=shared_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(device)
        else:
            policy = _build_hetero_mappo_policy(
                intersection_obs_dim=intersection_obs_dim,
                action_dim=action_dim,
                global_state_dim=global_state_dim,
                shared_dim=shared_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(device)

        # ── Optimiser ─────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

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
            gifting_entropy_coef=gifting_entropy_coef,
            value_coef=value_coef,
            max_grad_norm=0.5,
            update_epochs=update_epochs,
        )

        # ── Swap runner for gifting path ───────────────────────────────────
        if reward_sharing is not None:
            trainer.runner = GiftingGraphRunner(env=env, policy=policy)

        # ── Train ─────────────────────────────────────────────────────────
        model, history, model_path = run_training(
            trainer=trainer,
            total_timesteps=total_timesteps,
            rollout_steps=rollout_steps,
            algorithm_name=algorithm_name,
            model_path=str(
                Path(output_dir) / "models" / f"{algorithm_name}.pt"
            ),
        )

        return model, history, model_path

    finally:
        if env is not None:
            env.close()