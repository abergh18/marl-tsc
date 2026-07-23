"""
train_true_mappo.py

Training entry point for true Graph MAPPO with optional reward sharing.

Typical usage (no gifting):

    model, history, model_path = train_true_mappo(
        config_file=paths.config_file,
        network_file=paths.network_file,
        traffic_light_ids=traffic_light_ids,
        output_dir=OUTPUT_DIR,
        total_timesteps=100_000,
    )

Typical usage (zero-sum gifting):

    model, history, model_path = train_true_mappo(
        config_file=paths.config_file,
        network_file=paths.network_file,
        traffic_light_ids=traffic_light_ids,
        output_dir=OUTPUT_DIR,
        total_timesteps=100_000,
        reward_sharing="zero_sum",
    )

reward_sharing options
----------------------
None           — no gifting, standard true MAPPO (default)
"zero_sum"     — zero-sum peer reward sharing via ZeroSumRewardWrapper
"public_goods" — public goods peer reward sharing via PeerRewardingWrapper
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.optim

from marl_tsc.graph_based.graph_env import GraphTrafficEnv
from marl_tsc.graph_based.true_mappo_trainer import TrueMAPPOTrainer
from marl_tsc.graph_based.run_training import run_training
from marl_tsc.graph_based.models.policy_factory import (
    build_default_graph_mappo_policy,
)

def train_true_mappo(
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
    gifting_entropy_coef=0.01,
    value_coef=0.5,
    learning_rate=3e-4,
    update_epochs=3,
    hidden_dim=64,
    critic_hidden_dim=128,
    reward_sharing=None,
    gifting_divisions=None,
    graph_builder=None,       # NEW — optional MultiHopGraphBuilder
):
    """
    Train a true graph-based MAPPO policy with optional reward sharing.

    Parameters
    ----------
    graph_builder : optional
        Any object with a .build() method returning GraphTopology.
        Defaults to None — GraphTrafficEnv uses GraphBuilder internally.
        Pass a MultiHopGraphBuilder for richer connectivity on OSM networks.
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    env = None

    try:
        if reward_sharing is not None:
            from marl_tsc.traffic_env import SumoTrafficEnv
            from marl_tsc.graph_based.gifting_graph_runner import GiftingGraphRunner
            from marl_tsc.graph_based.models.reward_sharing_mappo_policy import GiftingMAPPOPolicy
            from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
            from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder

            sumo_env = SumoTrafficEnv(
                config_file,
                possible_agents=traffic_light_ids,
            )

            num_agents    = len(traffic_light_ids)
            num_divisions = gifting_divisions or max(1, num_agents - 1)

            if reward_sharing == "zero_sum":
                from marl_tsc.wrappers import ZeroSumRewardWrapper
                wrapped_sumo   = ZeroSumRewardWrapper(sumo_env, division=num_divisions)
                algorithm_name = "true_mappo_zero_sum"
                model_filename = "true_mappo_zero_sum.pt"

            elif reward_sharing == "public_goods":
                from marl_tsc.wrappers import PeerRewardingWrapper
                wrapped_sumo   = PeerRewardingWrapper(sumo_env, division=num_divisions)
                algorithm_name = "true_mappo_public_goods"
                model_filename = "true_mappo_public_goods.pt"

            else:
                raise ValueError(
                    f"Unknown reward_sharing type: {reward_sharing!r}. "
                    f"Currently supported: 'zero_sum', 'public_goods'."
                )

            env = GraphTrafficEnv(
                config_file=config_file,
                network_file=network_file,
                possible_agents=traffic_light_ids,
                sumo_env=wrapped_sumo,
                graph_builder=graph_builder,    # NEW
            )

            graph_obs, infos  = env.reset()
            obs_dim           = graph_obs.graph.x.shape[1]
            agent_ids         = env.agent_ids
            action_dim        = int(env.action_spaces[agent_ids[0]].nvec[0])
            global_state_dim  = env.global_state_dim

            encoder = GATEncoder(obs_dim=obs_dim, hidden_dim=hidden_dim)
            centralised_critic = CentralisedCritic(
                global_state_dim=global_state_dim,
                hidden_dim=critic_hidden_dim,
            )
            policy = GiftingMAPPOPolicy(
                encoder=encoder,
                action_dim=action_dim,
                num_divisions=num_divisions,
                centralised_critic=centralised_critic,
                hidden_dim=hidden_dim,
            ).to(device)

            runner_class = GiftingGraphRunner

        else:
            env = GraphTrafficEnv(
                config_file=config_file,
                network_file=network_file,
                possible_agents=traffic_light_ids,
                graph_builder=graph_builder,    # NEW
            )

            graph_obs, infos  = env.reset()
            obs_dim           = graph_obs.graph.x.shape[1]
            agent_ids         = env.agent_ids
            action_dim        = int(env.action_spaces[agent_ids[0]].n)
            global_state_dim  = env.global_state_dim

            policy = build_default_graph_mappo_policy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                global_state_dim=global_state_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(device)

            runner_class   = None
            algorithm_name = "true_mappo"
            model_filename = "true_mappo.pt"

        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)

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

        if runner_class is not None:
            trainer.runner = runner_class(env=env, policy=policy)

        model, history, model_path = run_training(
            trainer=trainer,
            total_timesteps=total_timesteps,
            rollout_steps=rollout_steps,
            algorithm_name=algorithm_name,
            model_path=str(
                Path(output_dir) / "models" / model_filename
            ),
        )

        return model, history, model_path

    finally:
        if env is not None:
            env.close()