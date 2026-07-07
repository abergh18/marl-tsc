"""
graph_mappo_policy.py

MAPPO-specific policy: graph-based actor + centralised critic.

Why a separate file
-------------------
The existing GraphPolicy and PolicyOutput are used by all other algorithms.
Rather than modifying shared code, this file defines parallel classes that
are only imported by GraphMAPPOTrainer. All other trainers continue using
GraphPolicy and PolicyOutput unchanged.

Architecture
------------
Actor  : GAT encoder -> actor_head, consumes local graph observations.
         Identical pathway to GraphPolicy — no change.

Critic : CentralisedCritic MLP, consumes global_state (all agent obs
         concatenated into one flat vector). Produces a single scalar V(s_t)
         broadcast to all agents during advantage computation.

This matches the MAPPO paper (Yu et al., 2021):
    "each agent shares the same policy parameters but has access to a
     centralised value function conditioned on the global state."
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


# ── Output dataclass ─────────────────────────────────────────────────────────
# Separate from PolicyOutput so existing algorithms are unaffected.

@dataclass
class MAPPOPolicyOutput:
    logits: Tensor          # (num_agents, action_dim)  — actor output
    global_value: Tensor    # (1,)                      — centralised V(s)
    encoder_output: Tensor  # (num_agents, hidden_dim)  — for inspection/logging


# ── Centralised critic ───────────────────────────────────────────────────────

class CentralisedCritic(nn.Module):
    """
    MLP critic conditioned on the full global state.

    Input  : global_state  (num_agents * obs_dim,)
    Output : V(s)          (1,)

    Two hidden layers with ReLU. Hidden dim is independent of the actor's
    hidden dim — critics often benefit from being wider than actors.
    """

    def __init__(self, global_state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(global_state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, global_state: Tensor) -> Tensor:
        return self.net(global_state)   # (1,)


# ── MAPPO policy ─────────────────────────────────────────────────────────────

class GraphMAPPOPolicy(nn.Module):
    """
    Graph-based MAPPO policy.

    Parameters
    ----------
    encoder : nn.Module
        Graph encoder (e.g. GAT). Takes a PyG Data object, returns node
        embeddings of shape (num_agents, hidden_dim).
    actor_head : nn.Module
        Maps encoder output (num_agents, hidden_dim) -> logits
        (num_agents, action_dim).
    centralised_critic : CentralisedCritic
        Maps global_state (num_agents * obs_dim,) -> V(s) (1,).
    """

    def __init__(
        self,
        encoder: nn.Module,
        actor_head: nn.Module,
        centralised_critic: CentralisedCritic,
    ):
        super().__init__()
        self.encoder = encoder
        self.actor_head = actor_head
        self.centralised_critic = centralised_critic

    def forward(self, graph_obs) -> MAPPOPolicyOutput:
        """
        Parameters
        ----------
        graph_obs : GraphObservation
            Must have:
                .graph        — PyG Data with node features x (num_agents, obs_dim)
                .global_state — flat Tensor (num_agents * obs_dim,)

        Returns
        -------
        MAPPOPolicyOutput
        """
        # Actor: local graph embeddings, identical to GraphPolicy
        encoder_output = self.encoder(graph_obs.graph)          # (num_agents, hidden_dim)
        logits = self.actor_head(encoder_output)                 # (num_agents, action_dim)

        # Critic: centralised, sees full global state
        global_value = self.centralised_critic(
            graph_obs.global_state                               # (num_agents * obs_dim,)
        )                                                        # (1,)

        return MAPPOPolicyOutput(
            logits=logits,
            global_value=global_value,
            encoder_output=encoder_output,
        )
