"""
train_hetero_mappo.py

Entry point for training a heterogeneous graph MAPPO policy.

Mirrors train_true_mappo.py exactly, with three substitutions:

    GraphTrafficEnv          -> HeteroGraphEnv
    GraphMAPPOPolicy         -> HeteroGraphMAPPOPolicy
    build_default_graph_mappo_policy -> build_default_hetero_mappo_policy

The reward sharing path is intentionally omitted for now — gifting on
het networks is a separate step once the base het MAPPO is verified.

Roadmap position: step 1 (het network support).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.optim

from marl_tsc.graph_based.hetero_graph_env import HeteroGraphEnv
from marl_tsc.graph_based.true_mappo_trainer import TrueMAPPOTrainer
from marl_tsc.graph_based.run_training import run_training
from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic
from marl_tsc.graph_based.models.hetero_graph_mappo_policy import HeteroGraphMAPPOPolicy
from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder


def _build_hetero_mappo_policy(
    intersection_obs_dim: int,
    action_dim: int,
    global_state_dim: int,
    shared_dim: int = 64,
    critic_hidden_dim: int = 128,
) -> HeteroGraphMAPPOPolicy:
    """
    Construct a HeteroGraphMAPPOPolicy with default actor/critic heads.

    The encoder is initialised with obs_dim=shared_dim because the
    projection layers inside the policy map both node types to shared_dim
    before the GAT encoder runs.
    """
    encoder = GATEncoder(
        obs_dim=shared_dim,         # encoder sees projected features
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
):
    """
    Train a heterogeneous graph MAPPO policy on a SUMO network.

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
    value_coef : float
    learning_rate : float
    update_epochs : int
    shared_dim : int
        Shared embedding dimension for both node types.  Used to size
        the projection layers and the GAT encoder input.
    critic_hidden_dim : int

    Returns
    -------
    model, history, model_path
    """

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env_options = dict(env_kwargs or {})
    env_options.setdefault("collect_global_metrics", False)

    env = None

    try:
        # ── Build environment ─────────────────────────────────────────────
        # HeteroGraphEnv needs intersection_obs_dim to size the projection
        # layers in HeteroGraphBuilder.  We do a temporary reset to read it,
        # then construct the full env properly.
        #
        # To avoid starting SUMO twice, we pass intersection_obs_dim
        # explicitly from env_kwargs if the caller knows it, otherwise we
        # fall back to a probe reset.  The cleaner long-term solution is to
        # read obs_dim from the SumoTrafficEnv before wrapping.

        # First pass: build env without knowing obs_dim yet.
        # We use a sentinel shared_dim=1 just to get obs_dim, then rebuild.
        # This is the same pattern as GraphTrafficEnv.obs_dim.

        _probe_env = HeteroGraphEnv(
            config_file=config_file,
            network_file=network_file,
            possible_agents=traffic_light_ids,
            intersection_obs_dim=1,     # placeholder — not used for obs_dim probe
            shared_dim=1,
            **env_options,
        )
        _probe_obs, _ = _probe_env.env.reset()
        intersection_obs_dim = len(next(iter(_probe_obs.values())))
        _probe_env.close()

        # ── Build proper environment with correct obs_dim ─────────────────
        env = HeteroGraphEnv(
            config_file=config_file,
            network_file=network_file,
            possible_agents=traffic_light_ids,
            intersection_obs_dim=intersection_obs_dim,
            shared_dim=shared_dim,
            **env_options,
        )

        graph_obs, _ = env.reset()
        agent_ids        = env.agent_ids
        action_dim       = int(env.action_spaces[agent_ids[0]].n)
        global_state_dim = env.global_state_dim

        # ── Policy ────────────────────────────────────────────────────────
        policy = _build_hetero_mappo_policy(
            intersection_obs_dim=intersection_obs_dim,
            action_dim=action_dim,
            global_state_dim=global_state_dim,
            shared_dim=shared_dim,
            critic_hidden_dim=critic_hidden_dim,
        ).to(device)

        # ── Optimiser ─────────────────────────────────────────────────────
        # policy.parameters() covers projection layers, encoder, actor head,
        # and centralised critic in one call.
        optimizer = torch.optim.Adam(
            policy.parameters(),
            lr=learning_rate,
        )

        # ── Trainer ───────────────────────────────────────────────────────
        # TrueMAPPOTrainer is unchanged — it works with any policy that
        # returns MAPPOPolicyOutput.
        trainer = TrueMAPPOTrainer(
            env=env,
            policy=policy,
            optimizer=optimizer,
            rollout_steps=rollout_steps,
            gae_lambda=0.95,
            gamma=0.99,
            clip_ratio=clip_ratio,
            entropy_coef=entropy_coef,
            gifting_entropy_coef=0.0,   # no gifting in this entry point
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
