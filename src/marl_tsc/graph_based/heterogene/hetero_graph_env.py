"""
hetero_graph_env.py

Graph-based environment wrapper that uses HeteroGraphBuilder to produce
observation graphs containing both intersection (agent) nodes and
connection nodes.

Differences from GraphTrafficEnv
---------------------------------
1.  Uses HeteroGraphBuilder instead of GraphBuilder.
2.  _to_graph() attaches connection_x (static, from topology) and
    agent_mask to the PyG Data object at each step.
3.  obs_dim refers to the raw intersection observation dimension only.
    The connection feature dimension is available via connection_feat_dim.
4.  shared_dim must be passed at construction so HeteroGraphBuilder can
    size its projection layers — these are exposed via policy_modules so
    the caller can include them in the policy's parameter group.

Everything else — reset(), step(), close(), action_spaces,
global_state_dim — is identical to GraphTrafficEnv.
"""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from ..graph_env import GraphObservation          # reuse existing dataclass
from .hetero_graph_builder import HeteroGraphBuilder
from ...traffic_env import SumoTrafficEnv            # adjust import path as needed


class HeteroGraphEnv:
    """
    Wrapper around SumoTrafficEnv that returns heterogeneous graph
    observations compatible with HeteroGraphMAPPOPolicy.

    Parameters
    ----------
    config_file : str
        Path to the SUMO .sumocfg file.
    network_file : str
        Path to the SUMO .net.xml file.
    possible_agents : list[str]
        TLS IDs to control.
    intersection_obs_dim : int
        Dimension of the per-agent SUMO observation vector.  Required to
        size the projection layers in HeteroGraphBuilder.
    shared_dim : int
        Shared embedding dimension for both node types.  Must match the
        obs_dim the GAT encoder is initialised with.
    sumo_env : optional
        Pre-constructed (and optionally pre-wrapped) SumoTrafficEnv.
        If provided, config_file is ignored for env construction.
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
        sumo_env=None,
        **env_kwargs,
    ):
        # Underlying SUMO environment
        if sumo_env is not None:
            self.env = sumo_env
        else:
            self.env = SumoTrafficEnv(
                config_file,
                possible_agents=possible_agents,
                **env_kwargs,
            )

        # Heterogeneous graph builder
        self.builder = HeteroGraphBuilder(
            network_file=network_file,
            intersection_obs_dim=intersection_obs_dim,
            shared_dim=shared_dim,
        )
        self.topology  = self.builder.build()
        self.agent_ids = self.topology.agent_ids

        # Cache static tensors — these don't change between steps
        self._connection_x = self.topology.connection_features  # (C, conn_feat_dim)
        self._agent_mask   = self.topology.agent_mask           # (N+C,) bool
        self._edge_index   = self.topology.edge_index           # (2, E)

    # ---- Graph construction ----------------------------------------------

    def _to_graph(self, observations: dict) -> GraphObservation:
        """
        Build a GraphObservation from the current SUMO observation dict.

        The Data object carries:
            .x            raw intersection obs       (N, intersection_obs_dim)
            .connection_x static connection feats    (C, connection_feat_dim)
            .edge_index   combined het edge index    (2, E)
            .agent_mask   bool mask (N+C,)  True = intersection/agent node

        Projection from raw features to shared_dim happens inside
        HeteroGraphMAPPOPolicy.forward() so that projection layers are
        trained end-to-end with the rest of the policy.
        """
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
            # Set custom attributes explicitly after construction
            graph.connection_x = self._connection_x
            graph.agent_mask   = self._agent_mask

            return GraphObservation(
                graph=graph,
                agent_ids=self.agent_ids,
                global_state=global_state,
            )

    # ---- PettingZoo-style interface --------------------------------------

    def reset(self, *args, **kwargs):
        observations, infos = self.env.reset(*args, **kwargs)
        graph_obs = self._to_graph(observations)
        return graph_obs, infos

    def step(self, actions):
        (
            observations,
            rewards,
            terminations,
            truncations,
            infos,
        ) = self.env.step(actions)

        graph_obs = self._to_graph(observations)

        return (
            graph_obs,
            rewards,
            terminations,
            truncations,
            infos,
        )

    def close(self):
        self.env.close()

    # ---- Properties ------------------------------------------------------

    @property
    def action_spaces(self):
        return {
            agent: self.env.action_space(agent)
            for agent in self.agent_ids
        }

    @property
    def obs_dim(self) -> int:
        """Raw intersection observation dimension."""
        graph_obs, _ = self.reset()
        return graph_obs.graph.x.shape[1]

    @property
    def global_state_dim(self) -> int:
        """Dimension of the centralised global state vector."""
        return len(self.agent_ids) * self.obs_dim

    @property
    def connection_feat_dim(self) -> int:
        """Dimension of the static connection node feature vector."""
        return self._connection_x.shape[1]
