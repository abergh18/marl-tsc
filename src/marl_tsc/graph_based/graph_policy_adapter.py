from __future__ import annotations

import torch

from marl_tsc.graph_based.graph_builder import GraphBuilder
from marl_tsc.graph_based.graph_types import GraphObservation

from torch_geometric.data import Data


class GraphPolicyAdapter:

    def __init__(
        self,
        graph_policy,
        topology,
    ):
        self.policy = graph_policy
        self.topology = topology

    @torch.no_grad()
    def act(
        self,
        observations,
        infos=None,
        deterministic=True,
    ):

        x = torch.tensor(
            [
                observations[agent]
                for agent in self.topology.agent_ids
            ],
            dtype=torch.float32,
        )

        graph_obs = GraphObservation(
            graph=Data(
                x=x,
                edge_index=self.topology.edge_index,
            ),
            agent_ids=self.topology.agent_ids,
        )

        output = self.policy(
            graph_obs
        )

        if deterministic:

            actions = (
                output.logits.argmax(
                    dim=-1
                )
            )

        else:

            dist = Categorical(
                logits=output.logits
            )

            actions = dist.sample()

        return {
            agent_id: int(action)
            for agent_id, action in zip(
                self.topology.agent_ids,
                actions,
            )
        }