"""
policy_factory.py

Convenience factory for constructing the default graph policy stack.

Why this exists
---------------
Most experiments use the same architecture:

    GATEncoder
        ↓
    ActorHead
        ↓
    CriticHead
        ↓
    GraphPolicy

This helper centralises model construction so training scripts can create
a policy with a single function call.
"""

from __future__ import annotations
from typing import Union

from marl_tsc.graph_based.encoders.gat_encoder import GATEncoder
from marl_tsc.graph_based.models.actor import ActorHead
from marl_tsc.graph_based.models.critic import CriticHead
from marl_tsc.graph_based.models.graph_policy import GraphPolicy
from marl_tsc.graph_based.models.true_mappo_policy import (
    CentralisedCritic,
    GraphMAPPOPolicy,
)


def build_default_graph_policy(
    obs_dim: int,
    action_dims: Union[int, list[int]],
    hidden_dim: int = 64,
):
    encoder = GATEncoder(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )

    actor_head = ActorHead(
        embedding_dim=hidden_dim,
        action_dims=action_dims,
    )

    critic_head = CriticHead(
        embedding_dim=hidden_dim,
    )

    return GraphPolicy(
        encoder=encoder,
        actor_head=actor_head,
        critic_head=critic_head,
    )


def build_default_graph_mappo_policy(
    obs_dim: int,
    action_dims: Union[int, list[int]],
    global_state_dim: int,
    hidden_dim: int = 64,
    critic_hidden_dim: int = 128,
):
    """
    Build the default GraphMAPPOPolicy.

    Actor and encoder are identical to build_default_graph_policy —
    same architecture, same hidden_dim. Only the critic differs:
    CentralisedCritic replaces the per-agent critic_head.

    Parameters
    ----------
    obs_dim : int
        Per-agent observation dimension.
    action_dims : int or list[int]
        Number of discrete actions per agent, or a list of dimensions
        for multi-discrete environments.
    global_state_dim : int
        Dimension of the global state vector (num_agents * obs_dim).
        Obtained from env.global_state_dim.
    hidden_dim : int
        Hidden dimension for encoder and actor head.
    critic_hidden_dim : int
        Hidden dimension for the centralised critic.
        Defaults to 128 (wider than actor is standard practice).

    Returns
    -------
    GraphMAPPOPolicy
    """
    encoder = GATEncoder(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )

    actor_head = ActorHead(
        embedding_dim=hidden_dim,
        action_dims=action_dims,
    )

    centralised_critic = CentralisedCritic(
        global_state_dim=global_state_dim,
        hidden_dim=critic_hidden_dim,
    )

    return GraphMAPPOPolicy(
        encoder=encoder,
        actor_head=actor_head,
        centralised_critic=centralised_critic,
    )