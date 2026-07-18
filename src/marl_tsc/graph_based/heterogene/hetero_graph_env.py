"""
hetero_graph_env.py  — v2

Graph-based environment wrapper using HeteroGraphBuilder v2.

Changes from v1
---------------
- max_hops parameter added and threaded through to HeteroGraphBuilder.
- proximity_matrix exposed as property for topology-aware reward sharing.
- connection_feat_dim updated to reflect new 7-feature vector.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..graph_env import GraphObservation
from .hetero_graph_builder import HeteroGraphBuilder
from ...traffic_env import SumoTrafficEnv


class HeteroGraphEnv:
    """
    Wrapper around SumoTrafficEnv returning heterogeneous graph observations.

    Parameters
    ----------
    config_file : str
    network_file : str
    possible_agents : list[str]
    intersection_obs_dim : int
    shared_dim : int
    max_hops : int
        BFS depth for connection node discovery. Default 3.
    sumo_env : optional
        Pre-constructed SumoTrafficEnv.
    **env_kwargs
        Passed to SumoTrafficEnv if sumo_env is None.
    """

    def __init__(
        self,
        config_file: str,
        network_file: str,
        possible_agents: list,
        intersection_obs_dim: int,
        shared_dim: int = 64,
        max_hops: int = 3,
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

        self.builder = HeteroGraphBuilder(
            network_file=network_file,
            intersection_obs_dim=intersection_obs_dim,
            shared_dim=shared_dim,
            max_hops=max_hops,
        )
        self.topology  = self.builder.build()
        self.agent_ids = self.topology.agent_ids

        # Cache static tensors
        self._connection_x = self.topology.connection_features
        self._agent_mask   = self.topology.agent_mask
        self._edge_index   = self.topology.edge_index

    # ── Graph construction ────────────────────────────────────────────────

    def _to_graph(self, observations: dict) -> GraphObservation:
        x = torch.tensor(
            np.stack([
                observations[agent]
                for agent in self.agent_ids
            ]),
            dtype=torch.float32,
        )

        global_state = x.flatten()

        graph = Data(
            x=x,
            edge_index=self._edge_index,
        )
        graph.connection_x = self._connection_x
        graph.agent_mask   = self._agent_mask

        return GraphObservation(
            graph=graph,
            agent_ids=self.agent_ids,
            global_state=global_state,
        )

    # ── PettingZoo-style interface ────────────────────────────────────────

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

    # ── Properties ───────────────────────────────────────────────────────

    @property
    def action_spaces(self):
        return {
            agent: self.env.action_space(agent)
            for agent in self.agent_ids
        }

    @property
    def obs_dim(self) -> int:
        graph_obs, _ = self.reset()
        return graph_obs.graph.x.shape[1]

    @property
    def global_state_dim(self) -> int:
        return len(self.agent_ids) * self.obs_dim

    @property
    def connection_feat_dim(self) -> int:
        return self._connection_x.shape[1]

    @property
    def proximity_matrix(self) -> torch.Tensor:
        """
        N x N proximity matrix for topology-aware reward sharing.
        proximity[i][j] = exp(-path_length / scale), 0 if unconnected.
        """
        return self.topology.proximity_matrix