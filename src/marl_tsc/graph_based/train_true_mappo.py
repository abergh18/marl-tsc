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
None        — no gifting, standard true MAPPO (default)
"zero_sum"  — zero-sum peer reward sharing via ZeroSumRewardWrapper
              Additional reward sharing types can be added here as
              further wrappers are implemented.
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
):
    """
    Train a true graph-based MAPPO policy with optional reward sharing.

    Parameters
    ----------
    config_file : str or Path
    network_file : str or Path
    traffic_light_ids : list[str]
    output_dir : str or Path
    total_timesteps : int
    rollout_steps : int
    max_steps : int
    seed : int
    env_kwargs : dict, optional
    clip_ratio : float
    entropy_coef : float
        Entropy coefficient for the traffic actor head.
    gifting_entropy_coef : float
        Entropy coefficient for the gifting head. Only used when
        reward_sharing is not None.
    value_coef : float
    learning_rate : float
    update_epochs : int
    hidden_dim : int
    critic_hidden_dim : int
    reward_sharing : str or None
        Reward sharing mechanism to use. Options:
            None        — standard true MAPPO, no gifting
            "zero_sum"  — zero-sum peer reward sharing
    gifting_divisions : int or None
        Number of discrete gifting portions. Defaults to
        num_agents - 1 when reward_sharing is not None.

    Returns
    -------
    model, history, model_path
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    # ── Build base environment ────────────────────────────────────────────────
    env = GraphTrafficEnv(
        config_file=config_file,
        network_file=network_file,
        possible_agents=traffic_light_ids,
    )

    try:
        graph_obs, infos = env.reset()
        obs_dim = env.obs_dim
        agent_ids = env.agent_ids
        action_dim = int(env.action_spaces[agent_ids[0]].n)
        global_state_dim = env.global_state_dim
        num_agents = len(agent_ids)

        # ── Apply reward sharing wrapper ──────────────────────────────────────
        if reward_sharing is not None:
            num_divisions = gifting_divisions or max(1, num_agents - 1)

            if reward_sharing == "zero_sum":
                from marl_tsc.graph_based.gifting_graph_runner import GiftingGraphRunner
                from marl_tsc.models.reward_sharing_mappo_policy import GiftingMAPPOPolicy
                from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
                from marl_tsc.graph_based.models.graph_encoder import GATEncoder
    
                env = ZeroSumRewardWrapper(env, division=num_divisions)
                graph_obs, infos = env.reset()
    
                encoder = GATEncoder(
                    obs_dim=obs_dim,
                    hidden_dim=hidden_dim,
                )
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
                algorithm_name = "true_mappo_zero_sum"
                model_filename = "true_mappo_zero_sum.pt"

            elif reward_sharing == "public_goods":
              from marl_tsc.wrappers import PeerRewardingWrapper
              from marl_tsc.graph_based.gifting_graph_runner import GiftingGraphRunner
              from marl_tsc.graph_based.reward_sharing_mappo_policy import GiftingMAPPOPolicy
              from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
              from marl_tsc.graph_based.models.graph_encoder import GATEncoder

              env = PeerRewardingWrapper(env, division=num_divisions)
              graph_obs, infos = env.reset()

              encoder = GATEncoder(
                  obs_dim=obs_dim,
                  hidden_dim=hidden_dim,
              )
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
              algorithm_name = "true_mappo_public_goods"
              model_filename = "true_mappo_public_goods.pt"
            
            else:
                raise ValueError(
                    f"Unknown reward_sharing type: {reward_sharing!r}. "
                    f"Currently supported: 'public_goods', 'zero_sum'."
                )

        else:
            # Standard true MAPPO — no gifting
            policy = build_default_graph_mappo_policy(
                obs_dim=obs_dim,
                action_dim=action_dim,
                hidden_dim=hidden_dim,
                global_state_dim=global_state_dim,
                critic_hidden_dim=critic_hidden_dim,
            ).to(device)

            runner_class = None
            algorithm_name = "true_mappo"
            model_filename = "true_mappo.pt"

        # ── Optimizer ─────────────────────────────────────────────────────────
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )

        # ── Trainer ───────────────────────────────────────────────────────────
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

        # Swap in gifting runner if needed
        if runner_class is not None:
            trainer.runner = runner_class(env=env, policy=policy)

        # ── Train ─────────────────────────────────────────────────────────────
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
        env.close()