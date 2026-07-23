# graph_env.py

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

    Parameters
    ----------
    config_file : str
    network_file : str
    possible_agents : list[str]
    graph_builder : optional
        Any object with a .build() method returning GraphTopology.
        Defaults to GraphBuilder(network_file) for backwards compatibility.
        Pass a MultiHopGraphBuilder instance for richer connectivity.
    sumo_env : optional
        Pre-constructed SumoTrafficEnv.
    **env_kwargs
        Passed to SumoTrafficEnv if sumo_env is None.
    """

    def __init__(
        self,
        config_file,
        network_file,
        possible_agents,
        graph_builder=None,
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

        # Use provided builder or default to original GraphBuilder
        builder = graph_builder or GraphBuilder(network_file)
        self.topology = builder.build()
        self.agent_ids = self.topology.agent_ids

    def _to_graph(self, observations):
        x = torch.tensor(
            np.stack([
                observations[agent]
                for agent in self.agent_ids
            ]),
            dtype=torch.float32,
        )
        global_state = x.flatten()

        return GraphObservation(
            graph=Data(
                x=x,
                edge_index=self.topology.edge_index,
            ),
            agent_ids=self.agent_ids,
            global_state=global_state,
        )

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        return self._to_graph(observations), infos

    def step(self, actions):
        observations, rewards, terminations, truncations, infos = (
            self.env.step(actions)
        )
        return self._to_graph(observations), rewards, terminations, truncations, infos

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
        return len(self.agent_ids) * self.obs_dim