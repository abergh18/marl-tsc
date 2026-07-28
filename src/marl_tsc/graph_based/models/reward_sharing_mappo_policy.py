"""
reward_sharing_mappo_policy.py

True MAPPO policy with centralised critic and branched gifting actor.

Architecture
------------
Encoder  : GATEncoder — shared base, analogous to the flat MAPPO's
           shared MLP base. Produces node_embeddings (num_agents, hidden_dim).

Branches : nn.ModuleList with two heads on top of node_embeddings:
               branches[0] — traffic logits  (num_agents, action_dim)
               branches[1] — gifting logits  (num_agents, num_divisions + 1)

           Mirrors the branched Actor in the collaborator's flat MAPPO
           implementation for consistency across the codebase.

Critic   : CentralisedCritic — maps global_state -> V(s) (1,).
           Unchanged from GraphMAPPOPolicy.

Why a separate file
-------------------
Keeps gifting experiments isolated from true_mappo_policy.py so
non-gifting runs are completely unaffected.

Attribution
-----------
Branched actor design consistent with collaborator's flat MAPPO
implementation. Zero-sum gifting mechanic adapted from:

    Lupu, A. & Precup, D. (2020). Gifting in Multi-Agent Reinforcement
    Learning. Proceedings of AAMAS 2020.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from marl_tsc.graph_based.models.true_mappo_policy import CentralisedCritic


# ── Output dataclass ─────────────────────────────────────────────────────────

@dataclass
class GiftingMAPPOPolicyOutput:
    logits: Tensor              # (num_agents, action_dim)       traffic
    gifting_logits: Tensor      # (num_agents, num_divisions+1)  gifting
    global_value: Tensor        # (1,)                           V(s)
    encoder_output: object      # EncoderOutput                  for inspection
    value: Tensor = None        # (num_agents,)                  GAE compat


# ── Policy ───────────────────────────────────────────────────────────────────

class GiftingMAPPOPolicy(nn.Module):
    """
    Graph-based true MAPPO policy with branched gifting actor.

    Parameters
    ----------
    encoder : nn.Module
        GATEncoder. Shared base — equivalent to the flat MLP base in
        the collaborator's Actor. Returns EncoderOutput with
        .node_embeddings (num_agents, hidden_dim).
    action_dim : int
        Number of discrete traffic actions.
    num_divisions : int
        Number of discrete gifting portions. Gifting action space
        has num_divisions + 1 choices (0..num_divisions).
    centralised_critic : CentralisedCritic
        Maps global_state (num_agents * obs_dim,) -> V(s) (1,).
    hidden_dim : int
        Dimension of encoder node embeddings. Branch inputs.
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        num_divisions: int,
        centralised_critic: CentralisedCritic,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.centralised_critic = centralised_critic
        self.action_dims = [action_dim, num_divisions + 1]

        # Branched heads on top of shared encoder output —
        # mirrors collaborator's nn.ModuleList(branches) pattern.
        self.branches = nn.ModuleList([
            nn.Linear(hidden_dim, action_dim),           # traffic
            nn.Linear(hidden_dim, num_divisions + 1),    # gifting
        ])

    def forward(self, graph_obs) -> GiftingMAPPOPolicyOutput:
        """
        Parameters
        ----------
        graph_obs : GraphObservation
            Must have:
                .graph        — PyG Data (node features for GAT)
                .global_state — flat Tensor (num_agents * obs_dim,)

        Returns
        -------
        GiftingMAPPOPolicyOutput
        """
        # Shared encoder — one pass feeds both branches and critic
        encoder_output = self.encoder(graph_obs)              # EncoderOutput
        node_emb = encoder_output.node_embeddings             # (num_agents, hidden_dim)

        # Branched heads
        logits = self.branches[0](node_emb)                   # (num_agents, action_dim)
        gifting_logits = self.branches[1](node_emb)           # (num_agents, div+1)

        # Centralised critic
        global_value = self.centralised_critic(
            graph_obs.global_state                            # (num_agents * obs_dim,)
        )                                                     # (1,)

        num_agents = node_emb.shape[0]

        return GiftingMAPPOPolicyOutput(
            logits=logits,
            gifting_logits=gifting_logits,
            global_value=global_value,
            encoder_output=encoder_output,
            value=global_value.expand(num_agents),            # (num_agents,) for GAE
        )