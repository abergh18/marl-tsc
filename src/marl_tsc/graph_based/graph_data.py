from __future__ import annotations

import numpy as np
import torch

from torch_geometric.data import Data


class GraphObservationBuilder:

    def __init__(self, topology):
        self.topology = topology

    def build(self, observations: dict[str, np.ndarray]) -> Data:
        """
        Convert SumoTrafficEnv observations into a PyG graph.

        Parameters
        ----------
        observations:
            {
                agent_id: observation_vector
            }

        Returns
        -------
        torch_geometric.data.Data
        """

        x = torch.tensor(
            np.stack(
                [
                    observations[agent_id]
                    for agent_id in self.topology.agent_ids
                ]
            ),
            dtype=torch.float32,
        )

        return Data(
            x=x,
            edge_index=self.topology.edge_index,
        )