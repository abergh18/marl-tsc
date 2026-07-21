"""
hetero_graph_policy_adapter.py

Policy adapter for heterogeneous graph policies, compatible with
evaluate_policy() in training.py.

Mirrors GraphPolicyAdapter but builds a het-aware GraphObservation
with connection_x and agent_mask attached to the Data object.

Usage
-----
    from marl_tsc.graph_based.heterogene.hetero_graph_policy_adapter import HeteroGraphPolicyAdapter

    het_adapter = HeteroGraphPolicyAdapter(
        graph_policy=het_model,
        topology=env.topology,
        policy_name="hetero_mappo",
    )
    results = evaluate_policy(
        config_file=paths.config_file,
        traffic_light_ids=traffic_light_ids,
        policy=het_adapter,
        ...
    )
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical
from torch_geometric.data import Data

from marl_tsc.graph_based.graph_types import GraphObservation


class HeteroGraphPolicyAdapter:
    """
    Wraps a HeteroGraphMAPPOPolicy or HeteroGiftingMAPPOPolicy for use
    with the standard evaluate_policy() function.

    Parameters
    ----------
    graph_policy : HeteroGraphMAPPOPolicy | HeteroGiftingMAPPOPolicy
        Trained het policy.
    topology : HeteroGraphTopology
        Built by HeteroGraphBuilder — carries connection_features,
        agent_mask, edge_index, and agent_ids.
    policy_name : str, optional
        Display name for logging.
    """

    def __init__(self, graph_policy, topology, policy_name=None):
        self.policy      = graph_policy
        self.topology    = topology
        self.policy_name = policy_name

    @torch.no_grad()
    def act(self, observations, infos=None, deterministic=True):
        device = next(self.policy.parameters()).device

        # ── Intersection node features ────────────────────────────────────
        x = torch.tensor(
            [observations[agent] for agent in self.topology.agent_ids],
            dtype=torch.float32,
            device=device,
        )

        # ── Static het tensors on correct device ──────────────────────────
        connection_x = self.topology.connection_features.to(device)
        edge_index   = self.topology.edge_index.to(device)
        agent_mask   = self.topology.agent_mask.to(device)

        # ── Build het-aware graph observation ─────────────────────────────
        graph = Data(x=x, edge_index=edge_index)
        graph.connection_x = connection_x
        graph.agent_mask   = agent_mask

        graph_obs = GraphObservation(
            graph=graph,
            agent_ids=self.topology.agent_ids,
            global_state=x.flatten(),
        )

        output = self.policy(graph_obs)

        # ── Select traffic action only ────────────────────────────────────
        # For gifting policies output.logits is the traffic branch (branches[0])
        if deterministic:
            actions = output.logits.argmax(dim=-1)
        else:
            dist    = Categorical(logits=output.logits)
            actions = dist.sample()

        return {
            agent_id: int(action)
            for agent_id, action in zip(
                self.topology.agent_ids,
                actions.cpu(),
            )
        }