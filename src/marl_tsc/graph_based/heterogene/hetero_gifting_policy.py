"""
hetero_gifting_mappo_policy.py

MAPPO policy with gifting branch for heterogeneous graphs.

Extends HeteroGraphMAPPOPolicy by adding a second actor branch for
the gifting action, mirroring GiftingMAPPOPolicy but with the
het-specific projection layers (intersection_proj, connection_proj)
included as nn.Module submodules.

Architecture
------------
    intersection obs  ──► intersection_proj ──┐
                                               ├──► cat ──► GATEncoder ──► mask ──► traffic_head ──► traffic logits
    connection feats  ──► connection_proj   ──┘                      │
                                                                      └──► gifting_head ──► gifting logits
    global_state ─────────────────────────────────────────────────────────► critic ──► V(s)

The gifting branch operates on agent embeddings after the agent mask
is applied — connection nodes never directly influence gifting decisions.
This mirrors the flat GiftingMAPPOPolicy design for consistency.

Parameters
----------
encoder : GATEncoder (or any BaseGraphEncoder)
    Initialised with obs_dim=shared_dim.
action_dim : int
    Number of discrete traffic phase actions.
num_divisions : int
    Number of discrete gifting portions. Gifting action space = num_divisions + 1.
centralised_critic : CentralisedCritic
    Maps global_state (N * obs_dim,) -> V(s).
intersection_obs_dim : int
    Raw per-agent SUMO observation dimension.
connection_feat_dim : int
    Static connection node feature dimension (default CONNECTION_FEAT_DIM=7).
shared_dim : int
    Shared embedding dimension. Must match encoder obs_dim.
"""

from __future__ import annotations
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .true_mappo_policy import CentralisedCritic, MAPPOPolicyOutput
from .hetero_graph_builder import CONNECTION_FEAT_DIM


@dataclass
class HeteroGiftingPolicyOutput:
    """
    Output from HeteroGiftingMAPPOPolicy.

    Mirrors GiftingMAPPOPolicyOutput for compatibility with
    GiftingGraphRunner and TrueMAPPOTrainer gifting loss path.
    """
    logits:          Tensor   # (N, action_dim)   — traffic actions
    gifting_logits:  Tensor   # (N, num_divisions+1) — gifting actions
    global_value:    Tensor   # (1,)
    value:           Tensor   # (N,)
    encoder_output:  Tensor   # (N, hidden_dim) — agent embeddings after mask


class HeteroGiftingMAPPOPolicy(nn.Module):
    """
    Het MAPPO policy with gifting branch.

    All trainable components — projection layers, encoder, traffic head,
    gifting head, critic — are submodules covered by policy.parameters().
    """

    def __init__(
        self,
        encoder: nn.Module,
        action_dim: int,
        num_divisions: int,
        centralised_critic: CentralisedCritic,
        intersection_obs_dim: int,
        connection_feat_dim: int = CONNECTION_FEAT_DIM,
        shared_dim: int = 64,
    ):
        super().__init__()

        # ── Projection layers ─────────────────────────────────────────────
        self.intersection_proj = nn.Linear(intersection_obs_dim, shared_dim)
        self.connection_proj   = nn.Linear(connection_feat_dim, shared_dim)

        # ── Core components ───────────────────────────────────────────────
        self.encoder            = encoder
        self.centralised_critic = centralised_critic

        # ── Actor branches ────────────────────────────────────────────────
        # Branch 0: traffic phase selection
        # Branch 1: gifting fraction selection
        # Stored as nn.ModuleList so both are covered by parameters()
        self.branches = nn.ModuleList([
            nn.Linear(shared_dim, action_dim),
            nn.Linear(shared_dim, num_divisions + 1),
        ])

        self.shared_dim    = shared_dim
        self.num_divisions = num_divisions

    def forward(self, graph_obs) -> HeteroGiftingPolicyOutput:
        """
        Parameters
        ----------
        graph_obs : GraphObservation
            .graph.x             (N, intersection_obs_dim)
            .graph.connection_x  (C, connection_feat_dim)
            .graph.edge_index    (2, E)
            .graph.agent_mask    (N+C,) bool
            .global_state        (N * intersection_obs_dim,)
        """
        graph = graph_obs.graph

        # ── 1. Project both node types ────────────────────────────────────
        intersection_emb = self.intersection_proj(graph.x)           # (N, shared_dim)
        connection_emb   = self.connection_proj(graph.connection_x)  # (C, shared_dim)

        projected_graph = _ProjectedGraph(
            x=torch.cat([intersection_emb, connection_emb], dim=0),
            edge_index=graph.edge_index,
        )
        projected_obs = _ProjectedObservation(
            graph=projected_graph,
            global_state=graph_obs.global_state,
        )

        # ── 2. Encode ─────────────────────────────────────────────────────
        encoder_output = self.encoder(projected_obs)   # (N+C, hidden_dim)

        # ── 3. Mask to agent nodes ────────────────────────────────────────
        agent_embeddings = encoder_output.node_embeddings[
            graph.agent_mask
        ]                                              # (N, hidden_dim)

        # ── 4. Actor branches ─────────────────────────────────────────────
        traffic_logits  = self.branches[0](agent_embeddings)  # (N, action_dim)
        gifting_logits  = self.branches[1](agent_embeddings)  # (N, num_divisions+1)

        # ── 5. Centralised critic ─────────────────────────────────────────
        global_value = self.centralised_critic(graph_obs.global_state)  # (1,)
        num_agents   = agent_embeddings.shape[0]

        return HeteroGiftingPolicyOutput(
            logits=traffic_logits,
            gifting_logits=gifting_logits,
            global_value=global_value,
            value=global_value.expand(num_agents),
            encoder_output=agent_embeddings,
        )


# ── Lightweight wrappers ──────────────────────────────────────────────────────

class _ProjectedGraph:
    __slots__ = ("x", "edge_index")
    def __init__(self, x: Tensor, edge_index: Tensor):
        self.x          = x
        self.edge_index = edge_index


class _ProjectedObservation:
    __slots__ = ("graph", "global_state")
    def __init__(self, graph: _ProjectedGraph, global_state: Tensor):
        self.graph        = graph
        self.global_state = global_state