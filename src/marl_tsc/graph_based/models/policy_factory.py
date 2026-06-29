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

from marl_tsc.graph_based.encoders.gat_encoder import (
    GATEncoder,
)

from marl_tsc.graph_based.models.actor import (
    ActorHead,
)

from marl_tsc.graph_based.models.critic import (
    CriticHead,
)

from marl_tsc.graph_based.models.graph_policy import (
    GraphPolicy,
)

def build_default_graph_policy(
    obs_dim,
    action_dim,
    hidden_dim=64,
):
    encoder = GATEncoder(
        obs_dim=obs_dim,
        hidden_dim=hidden_dim,
    )

    actor_head = ActorHead(
        embedding_dim=hidden_dim,
        action_dim=action_dim,
    )

    critic_head = CriticHead(
        embedding_dim=hidden_dim,
    )

    return GraphPolicy(
        encoder=encoder,
        actor_head=actor_head,
        critic_head=critic_head,
    )