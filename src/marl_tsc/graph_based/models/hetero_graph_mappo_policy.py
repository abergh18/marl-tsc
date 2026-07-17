"""
hetero_graph_mappo_policy.py

MAPPO policy for heterogeneous graphs that include both intersection
(agent) nodes and connection nodes.

The policy is a drop-in replacement for GraphMAPPOPolicy for use with
HeteroGraphBuilder output.  It differs in three ways:

1.  Two learned projection layers (intersection_proj, connection_proj)
    map both node types to a shared embedding dimension before the GAT
    encoder runs.  All nn.Module parameters — projections, encoder,
    actor head, critic — live here and are covered by one
    policy.parameters() call.

2.  An agent_mask is applied after the encoder to slice agent-only
    embeddings before the actor head.  Connection nodes participate in
    GAT message passing but never reach the action or value heads.

3.  The centralised critic is unchanged — it still receives the flat
    global_state tensor (num_agents * obs_dim,) built from intersection
    observations only.

Architecture
------------
    intersection obs  ──► intersection_proj ──┐
                                               ├──► cat ──► GATEncoder ──► mask ──► actor_head ──► logits
    connection feats  ──► connection_proj   ──┘                      │
                                                                      └──► (discarded)
    global_state ─────────────────────────────────────────────────────────► critic ──► V(s)

Parameters
----------
encoder : GATEncoder (or any BaseGraphEncoder)
    Must be initialised with obs_dim=shared_dim, not the raw
    intersection observation dimension.
actor_head : nn.Module
    Maps (num_agents, hidden_dim) -> (num_agents, action_dim).
centralised_critic : CentralisedCritic
    Maps (num_agents * obs_dim,) -> (1,).  Unchanged from GraphMAPPOPolicy.
intersection_obs_dim : int
    Dimension of the raw per-agent SUMO observation vector.
connection_feat_dim : int
    Dimension of the static connection node feature vector.
    Defaults to CONNECTION_FEAT_DIM from hetero_graph_builder (9).
shared_dim : int
    Dimension of the shared embedding space fed to the GAT encoder.
    Must match the obs_dim the encoder was initialised with.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from .true_mappo_policy import CentralisedCritic, MAPPOPolicyOutput
from ..heterogene.hetero_graph_builder import CONNECTION_FEAT_DIM


class HeteroGraphMAPPOPolicy(nn.Module):
    """
    MAPPO policy for heterogeneous graphs.

    All trainable components are submodules of this class so a single
    optimiser.step() covers projections, encoder, actor, and critic.
    """

    def __init__(
        self,
        encoder: nn.Module,
        actor_head: nn.Module,
        centralised_critic: CentralisedCritic,
        intersection_obs_dim: int,
        connection_feat_dim: int = CONNECTION_FEAT_DIM,
        shared_dim: int = 64,
    ):
        super().__init__()

        # ── Projection layers ────────────────────────────────────────────
        # Both map their respective node types to shared_dim so the GAT
        # encoder sees a uniform feature dimension across all nodes.
        self.intersection_proj = nn.Linear(intersection_obs_dim, shared_dim)
        self.connection_proj   = nn.Linear(connection_feat_dim, shared_dim)

        # ── Core components (same roles as GraphMAPPOPolicy) ─────────────
        self.encoder             = encoder
        self.actor_head          = actor_head
        self.centralised_critic  = centralised_critic

        self.shared_dim = shared_dim

    # ── Forward ──────────────────────────────────────────────────────────

    def forward(self, graph_obs) -> MAPPOPolicyOutput:
        """
        Parameters
        ----------
        graph_obs : GraphObservation
            Must have:
                .graph             — PyG Data with:
                                         .x           raw intersection obs
                                                       (N, intersection_obs_dim)
                                         .connection_x static connection feats
                                                       (C, connection_feat_dim)
                                         .edge_index   combined [2, E]
                                         .agent_mask   bool [N+C], True=agent
                .global_state      — flat Tensor (N * intersection_obs_dim,)

        Returns
        -------
        MAPPOPolicyOutput
            Same interface as GraphMAPPOPolicy output.
        """
        graph = graph_obs.graph

        # ── 1. Project both node types to shared_dim ─────────────────────
        intersection_emb = self.intersection_proj(graph.x)          # (N, shared_dim)
        connection_emb   = self.connection_proj(graph.connection_x)  # (C, shared_dim)

        # Replace graph.x with the combined projected features so the
        # encoder sees a single homogeneous feature matrix.
        # We avoid mutating graph_obs in place — build a lightweight
        # wrapper the encoder can consume.
        projected_graph = _ProjectedGraph(
            x=torch.cat([intersection_emb, connection_emb], dim=0),  # (N+C, shared_dim)
            edge_index=graph.edge_index,
        )
        projected_obs = _ProjectedObservation(
            graph=projected_graph,
            global_state=graph_obs.global_state,
        )

        # ── 2. Encode ────────────────────────────────────────────────────
        encoder_output = self.encoder(projected_obs)    # node_embeddings: (N+C, hidden_dim)

        # ── 3. Mask to agent nodes only ──────────────────────────────────
        agent_embeddings = encoder_output.node_embeddings[
            graph.agent_mask
        ]                                               # (N, hidden_dim)

        # ── 4. Actor head ────────────────────────────────────────────────
        logits = self.actor_head(agent_embeddings)      # (N, action_dim)

        # ── 5. Centralised critic ────────────────────────────────────────
        # Global state is intersection-only — unchanged from GraphMAPPOPolicy.
        global_value = self.centralised_critic(
            graph_obs.global_state                      # (N * obs_dim,)
        )                                               # (1,)

        num_agents = agent_embeddings.shape[0]

        return MAPPOPolicyOutput(
            logits=logits,
            global_value=global_value,
            encoder_output=agent_embeddings,            # (N, hidden_dim) — agents only
            value=global_value.expand(num_agents),
        )


# ── Lightweight wrappers ──────────────────────────────────────────────────────
# These avoid mutating GraphObservation in place while keeping the interface
# the encoder expects.

class _ProjectedGraph:
    """Minimal graph container for the projected features."""
    __slots__ = ("x", "edge_index")

    def __init__(self, x: Tensor, edge_index: Tensor):
        self.x          = x
        self.edge_index = edge_index


class _ProjectedObservation:
    """Minimal observation container wrapping the projected graph."""
    __slots__ = ("graph", "global_state")

    def __init__(self, graph: _ProjectedGraph, global_state: Tensor):
        self.graph        = graph
        self.global_state = global_state
