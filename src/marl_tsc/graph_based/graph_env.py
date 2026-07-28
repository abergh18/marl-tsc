from __future__ import annotations

import numpy as np
import torch

from torch_geometric.data import Data

from marl_tsc.traffic_env import SumoTrafficEnv

from .graph_builder import GraphBuilder
from .graph_types import GraphObservation


class GraphTrafficEnv:
    """
    Graph-based wrapper around SumoTrafficEnv.

    Returns PyTorch Geometric Data objects instead of
    observation dictionaries.

    Changes from original
    ---------------------
    - _to_graph() now computes global_state = x.flatten() and attaches it
      to GraphObservation. This is used by the centralised critic in true
      MAPPO. All other algorithms receive it as None by default and ignore it.
    - global_state_dim property added for policy construction.
    """

    def __init__(
        self,
        config_file,
        network_file,
        possible_agents,
        sumo_env=None,
        **env_kwargs,
    ):
        if sumo_env is not None:
            self.env = sumo_env
        else:
            self.env = SumoTrafficEnv(
                config_file,
                possible_agents=possible_agents,
                **env_kwargs,
            )

        self.topology = GraphBuilder(
            network_file
        ).build()

        self.agent_ids = self.topology.agent_ids

    def _to_graph(self, observations):

        x = torch.tensor(
            np.stack(
                [
                    observations[agent]
                    for agent in self.agent_ids
                ]
            ),
            dtype=torch.float32,
        )                                       # (num_agents, obs_dim)

        global_state = x.flatten()              # (num_agents * obs_dim,)

        return GraphObservation(
            graph=Data(
                x=x,
                edge_index=self.topology.edge_index,
            ),
            agent_ids=self.agent_ids,
            global_state=global_state,
        )

    def reset(self, *args, **kwargs):

        observations, infos = self.env.reset(
            *args,
            **kwargs,
        )

        graph = self._to_graph(observations)

        return graph, infos

    def step(self, actions):

        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        graph = self._to_graph(observations)

        return (
            graph,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def close(self):
        self.env.close()

    @property
    def action_spaces(self):
        return {
            agent: self.env.action_space(agent)
            for agent in self.agent_ids
        }

    @property
    def observation_dim(self):
        graph_obs, _ = self.reset()
        return graph_obs.graph.x.shape[1]

    @property
    def obs_dim(self):
        return self.reset()[0].graph.x.shape[1]

    @property
    def global_state_dim(self):
        """Dimension of the centralised global state vector."""
        return self.obs_dim * len(self.agent_ids)