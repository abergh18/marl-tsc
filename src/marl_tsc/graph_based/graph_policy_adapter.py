from __future__ import annotations

import torch

from marl_tsc.graph_based.graph_builder import GraphBuilder
from marl_tsc.graph_based.graph_types import GraphObservation

from torch_geometric.data import Data
from torch.distributions import Categorical


class GraphPolicyAdapter:

    def __init__(
        self,
        graph_policy,
        topology,
        policy_name = None,
    ):
        self.policy = graph_policy
        self.topology = topology
        self.policy_name = policy_name

    @torch.no_grad()
    def act(
        self,
        observations,
        infos=None,
        deterministic=True,
    ):
        # Identify the policy's device (e.g., cuda:0)
        device = next(self.policy.parameters()).device

        # Create the node features tensor directly on the target device
        x = torch.tensor(
            [
                observations[agent]
                for agent in self.topology.agent_ids
            ],
            dtype=torch.float32,
            device=device,
        )

        # Move the topology edge_index to the target device as well
        edge_index = self.topology.edge_index.to(device)

        graph_obs = GraphObservation(
            graph=Data(
                x=x,
                edge_index=edge_index,
            ),
            agent_ids=self.topology.agent_ids,
            global_state=x.flatten(),
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

        # Bring actions back to CPU if necessary for zip/int casting (safer for loop extraction)
        actions_cpu = actions.cpu()

        return {
            agent_id: int(action)
            for agent_id, action in zip(
                self.topology.agent_ids,
                actions_cpu,
            )
        }