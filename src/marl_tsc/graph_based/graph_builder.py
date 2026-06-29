from dataclasses import dataclass
from pathlib import Path

import torch
from sumolib.net import readNet


@dataclass
class GraphTopology:
    agent_ids: list[str]
    edge_index: torch.Tensor
    node_index: dict[str, int]


class GraphBuilder:

    def __init__(self, network_file):
        self.network_file = Path(network_file)
        self.net = readNet(str(self.network_file))

    def build(self):

        agent_ids = sorted(
            tls.getID()
            for tls in self.net.getTrafficLights()
        )

        node_index = {
            agent_id: idx
            for idx, agent_id in enumerate(agent_ids)
        }

        edges = set()

        for tls in self.net.getTrafficLights():

            source_id = tls.getID()

            for edge in tls.getEdges():

                for outgoing_edge in edge.getOutgoing().keys():

                    neighbour_tls = outgoing_edge.getTLS()

                    if neighbour_tls is None:
                        continue

                    target_id = neighbour_tls.getID()

                    if source_id == target_id:
                        continue

                    edges.add(
                        (
                            node_index[source_id],
                            node_index[target_id],
                        )
                    )

        undirected_edges = set()

        for src, dst in edges:
            undirected_edges.add((src, dst))
            undirected_edges.add((dst, src))

        edge_index = torch.tensor(
            list(undirected_edges),
            dtype=torch.long,
        ).t().contiguous()

        return GraphTopology(
            agent_ids=agent_ids,
            edge_index=edge_index,
            node_index=node_index,
        )