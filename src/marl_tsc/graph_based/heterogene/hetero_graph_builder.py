"""
hetero_graph_builder.py  — v2

Drop-in companion to GraphBuilder that enriches the intersection graph
with connection nodes representing road segments between intersections.

Changes from v1
---------------
- Connection node traversal uses multi-hop BFS (up to max_hops) instead
  of single-hop getOutgoing(). This captures tertiary, unclassified, and
  residential connections between TLS junctions, not just direct primary
  road links.
- Connection node feature vector updated to combined B+C scheme (7 features)
  capturing both bottleneck constraints and aggregate path characteristics.
- max_hops exposed as constructor parameter.

Node layout in the output Data object
--------------------------------------
Indices 0 .. N-1          : intersection (agent) nodes
Indices N .. N+C-1        : connection nodes

Both node types are projected to shared_dim so the existing homogeneous
GAT encoder can be used unchanged.

Connection node features (static, built once from .net.xml)
------------------------------------------------------------
0  first_priority    road priority of first edge leaving source TLS  (normalised)
1  min_lanes         bottleneck lane count along path                (normalised)
2  total_length      sum of all hop lengths                          (normalised)
3  min_speed         bottleneck speed limit                          (normalised)
4  mean_speed        average speed limit across hops                 (normalised)
5  num_hops          number of road segments in path                 (normalised by max_hops)
6  any_signalised    binary: any intermediate junction is TLS-controlled
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sumolib.net import readNet
from torch_geometric.data import Data


# ── Constants ─────────────────────────────────────────────────────────────────

CONNECTION_FEAT_DIM = 7

_MAX_PRIORITY  = 12.0
_MAX_LENGTH    = 1500.0   # increased for multi-hop paths
_MAX_SPEED     = 33.33
_MAX_LANES     = 4.0


# ── Dataclass ─────────────────────────────────────────────────────────────────

@dataclass
class HeteroGraphTopology:
    """
    Static topology built once from the network file.

    Parameters
    ----------
    agent_ids
        Ordered list of TLS intersection IDs.  Node indices 0..N-1.
    node_index
        Maps agent_id -> index in [0, N).
    connection_features
        Float tensor [C, CONNECTION_FEAT_DIM] for connection nodes N..N+C-1.
    connection_meta
        List of (from_tls_id, to_tls_id, edge_id, num_hops) for each
        connection node, in the same order as connection_features rows.
    edge_index
        Combined edge_index [2, E] covering both directions of
        intersection -> connection -> intersection paths.
    num_intersections
        N — number of agent/intersection nodes.
    num_connections
        C — number of connection nodes.
    agent_mask
        Boolean tensor [N+C] — True for intersection nodes.
    proximity_matrix
        Float tensor [N, N] — normalised proximity weights between agents,
        computed from path lengths. Useful for topology-aware reward sharing.
    """
    agent_ids:           list
    node_index:          dict
    connection_features: torch.Tensor
    connection_meta:     list
    edge_index:          torch.Tensor
    num_intersections:   int
    num_connections:     int
    agent_mask:          torch.Tensor
    proximity_matrix:    torch.Tensor


# ── Builder ───────────────────────────────────────────────────────────────────

class HeteroGraphBuilder:
    """
    Builds an enriched graph inserting connection nodes between intersections,
    compatible with the existing homogeneous GAT encoder via learned projection.

    Parameters
    ----------
    network_file : str
        Path to the SUMO .net.xml file.
    intersection_obs_dim : int
        Dimension of the per-agent SUMO observation vector.
    shared_dim : int
        Shared embedding dimension. Both node types are projected here.
    max_hops : int
        Maximum BFS depth when searching for TLS-to-TLS connections.
        max_hops=1 finds only direct single-edge connections.
        max_hops=3 (default) finds connections via up to 3 road segments.
    """

    def __init__(
        self,
        network_file: str,
        intersection_obs_dim: int,
        shared_dim: int = 64,
        max_hops: int = 3,
    ):
        self.network_file         = Path(network_file)
        self.intersection_obs_dim = intersection_obs_dim
        self.shared_dim           = shared_dim
        self.max_hops             = max_hops

        self.net = readNet(str(self.network_file))

        # Projection layers — included in HeteroGraphMAPPOPolicy.parameters()
        self.intersection_proj = nn.Linear(intersection_obs_dim, shared_dim)
        self.connection_proj   = nn.Linear(CONNECTION_FEAT_DIM, shared_dim)

        self.topology = self._build_topology()

    # ── Public API ────────────────────────────────────────────────────────

    def build(self) -> HeteroGraphTopology:
        return self.topology

    def to_graph(
        self,
        observations: dict,
        topology: Optional[HeteroGraphTopology] = None,
    ) -> Data:
        """
        Build a PyG Data object from current SUMO observations.

        Returns
        -------
        Data with:
            .x            raw intersection obs  (N, intersection_obs_dim)
            .connection_x static connection feats (C, CONNECTION_FEAT_DIM)
            .edge_index   combined het edge index (2, E)
            .agent_mask   bool [N+C], True = intersection node
        """
        if topology is None:
            topology = self.topology

        x = torch.tensor(
            np.stack([
                observations[agent_id]
                for agent_id in topology.agent_ids
            ]),
            dtype=torch.float32,
        )

        graph = Data(
            x=x,
            edge_index=topology.edge_index,
        )
        graph.connection_x = topology.connection_features
        graph.agent_mask   = topology.agent_mask

        return graph

    # ── Internal topology construction ────────────────────────────────────

    def _build_topology(self) -> HeteroGraphTopology:
        tls_list  = list(self.net.getTrafficLights())
        agent_ids = sorted(tls.getID() for tls in tls_list)
        node_index = {aid: idx for idx, aid in enumerate(agent_ids)}
        N = len(agent_ids)

        # ── Find connections via BFS ───────────────────────────────────────
        raw_connections = self._find_connections(agent_ids)
        C = len(raw_connections)

        # ── Build connection feature matrix ───────────────────────────────
        connection_features_list = []
        connection_meta          = []
        edges_src = []   # intersection index
        edges_dst = []   # connection node index (offset by N)

        for c_idx, conn in enumerate(raw_connections):
            feat = self._connection_features(conn)
            connection_features_list.append(feat)
            connection_meta.append((
                conn["from_tls"],
                conn["to_tls"],
                conn["edge_id"],
                conn["num_hops"],
            ))
            edges_src.append(node_index[conn["from_tls"]])
            edges_dst.append(c_idx + N)

        # ── Build connection -> destination intersection edges ─────────────
        conn_to_int_src = []
        conn_to_int_dst = []
        for c_idx, conn in enumerate(raw_connections):
            conn_to_int_src.append(c_idx + N)
            conn_to_int_dst.append(node_index[conn["to_tls"]])

        # Both directions for undirected-equivalent message passing
        all_src = (
            edges_src           +   # intersection -> connection
            [d for d in edges_dst] +   # connection -> intersection (reverse)
            conn_to_int_src     +   # connection -> intersection
            conn_to_int_dst         # intersection -> connection (reverse)
        )
        all_dst = (
            edges_dst           +
            edges_src           +
            conn_to_int_dst     +
            conn_to_int_src
        )

        edge_index = torch.tensor(
            [all_src, all_dst], dtype=torch.long
        ).contiguous()

        connection_features = torch.tensor(
            np.stack(connection_features_list),
            dtype=torch.float32,
        ) if connection_features_list else torch.zeros(0, CONNECTION_FEAT_DIM)

        # ── Agent mask ────────────────────────────────────────────────────
        agent_mask = torch.zeros(N + C, dtype=torch.bool)
        agent_mask[:N] = True

        # ── Proximity matrix ──────────────────────────────────────────────
        proximity_matrix = self._build_proximity_matrix(
            agent_ids, node_index, raw_connections, N
        )

        return HeteroGraphTopology(
            agent_ids=agent_ids,
            node_index=node_index,
            connection_features=connection_features,
            connection_meta=connection_meta,
            edge_index=edge_index,
            num_intersections=N,
            num_connections=C,
            agent_mask=agent_mask,
            proximity_matrix=proximity_matrix,
        )

    def _find_connections(self, agent_ids: list) -> list:
        """
        BFS from each TLS junction to find all reachable TLS junctions
        within max_hops road segments.

        Returns list of connection dicts with aggregated path features.
        """
        node_index = set(agent_ids)
        connections = []
        seen_pairs  = set()

        for start_tls_id in agent_ids:
            start_node = self._get_tls_node(start_tls_id)
            if start_node is None:
                continue

            # BFS: (current_node, hops, path_data)
            queue   = [(start_node, 0, None)]
            visited = {start_node.getID()}

            while queue:
                current_node, hops, path_data = queue.pop(0)
                if hops >= self.max_hops:
                    continue

                for out_edge in current_node.getOutgoing():
                    to_node    = out_edge.getToNode()
                    to_node_id = to_node.getID()

                    if to_node_id in visited:
                        continue
                    visited.add(to_node_id)

                    # Accumulate path data
                    if path_data is None:
                        new_path = {
                            "edge_id":        out_edge.getID(),
                            "first_priority": out_edge.getPriority(),
                            "min_lanes":      out_edge.getLaneNumber(),
                            "total_length":   out_edge.getLength(),
                            "min_speed":      out_edge.getSpeed(),
                            "speed_sum":      out_edge.getSpeed(),
                            "num_hops":       1,
                            "any_signalised": out_edge.getTLS() is not None,
                        }
                    else:
                        new_path = {
                            "edge_id":        path_data["edge_id"],
                            "first_priority": path_data["first_priority"],
                            "min_lanes":      min(path_data["min_lanes"],
                                                  out_edge.getLaneNumber()),
                            "total_length":   path_data["total_length"] +
                                              out_edge.getLength(),
                            "min_speed":      min(path_data["min_speed"],
                                                  out_edge.getSpeed()),
                            "speed_sum":      path_data["speed_sum"] +
                                              out_edge.getSpeed(),
                            "num_hops":       path_data["num_hops"] + 1,
                            "any_signalised": path_data["any_signalised"] or
                                              out_edge.getTLS() is not None,
                        }

                    to_tls_id = to_node.getTLSID()
                    if (to_tls_id and to_tls_id in node_index
                            and to_tls_id != start_tls_id):
                        pair = (start_tls_id, to_tls_id)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            connections.append({
                                "from_tls":       start_tls_id,
                                "to_tls":         to_tls_id,
                                "mean_speed":     new_path["speed_sum"] /
                                                  new_path["num_hops"],
                                **{k: v for k, v in new_path.items()
                                   if k != "speed_sum"},
                            })
                    else:
                        queue.append((to_node, hops + 1, new_path))

        return connections

    def _connection_features(self, conn: dict) -> np.ndarray:
        """
        Build normalised 7-feature vector for a connection node.

        Features
        --------
        0  first_priority   priority of first edge / _MAX_PRIORITY
        1  min_lanes        bottleneck lane count / _MAX_LANES
        2  total_length     total path length / _MAX_LENGTH
        3  min_speed        bottleneck speed / _MAX_SPEED
        4  mean_speed       average speed / _MAX_SPEED
        5  num_hops         hops / max_hops
        6  any_signalised   binary
        """
        return np.array([
            conn["first_priority"] / _MAX_PRIORITY,
            conn["min_lanes"]      / _MAX_LANES,
            min(conn["total_length"], _MAX_LENGTH) / _MAX_LENGTH,
            conn["min_speed"]      / _MAX_SPEED,
            conn["mean_speed"]     / _MAX_SPEED,
            conn["num_hops"]       / self.max_hops,
            float(conn["any_signalised"]),
        ], dtype=np.float32)

    def _build_proximity_matrix(
        self,
        agent_ids: list,
        node_index: dict,
        connections: list,
        N: int,
    ) -> torch.Tensor:
        """
        Build an N x N proximity matrix using exponential decay on
        total path length.  proximity[i][j] = exp(-total_length / scale).

        Diagonal is 0 (no self-gifting).
        Unconnected pairs get proximity 0.
        """
        scale = 200.0   # decay scale in metres — tunable
        matrix = torch.zeros(N, N)

        for conn in connections:
            i = node_index[conn["from_tls"]]
            j = node_index[conn["to_tls"]]
            w = np.exp(-conn["total_length"] / scale)
            matrix[i][j] = w
            matrix[j][i] = w   # symmetric

        return matrix

    def _get_tls_node(self, tls_id: str):
        """
        Get the sumolib Node for a TLS ID, handling joinedS composite IDs.
        """
        if self.net.hasNode(tls_id):
            return self.net.getNode(tls_id)
        parts = tls_id.replace("joinedS_", "").split("_cluster_")
        for part in parts:
            for candidate in [part] + part.split("_"):
                if self.net.hasNode(candidate):
                    return self.net.getNode(candidate)
        return None