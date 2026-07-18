"""
hetero_graph_builder.py  — v3

Drop-in companion to GraphBuilder that enriches the intersection graph
with connection nodes representing road segments between intersections.

Changes from v2
---------------
- Normalisation constants are now computed from the actual network file
  at init time rather than hardcoded. Uses 95th percentile for length
  and speed (robust to outlier edges), max for priority and lanes.
  This ensures feature values stay in [0, 1] regardless of which network
  is used — critical for transfer between cities.
- _compute_norm_stats() added as a new method.
- _MAX_* module-level constants removed.

Node layout in the output Data object
--------------------------------------
Indices 0 .. N-1          : intersection (agent) nodes
Indices N .. N+C-1        : connection nodes

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
    norm_stats
        Dict of normalisation constants derived from this network.
        Stored for inspection and logging.
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
    norm_stats:          dict


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

        # Compute normalisation stats from this network before building topology
        self._norm = self._compute_norm_stats()

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

    # ── Normalisation ─────────────────────────────────────────────────────

    def _compute_norm_stats(self) -> dict:
        """
        Compute normalisation constants from the actual network file.

        Uses max for priority and lanes (small integer ranges, no outlier risk).
        Uses 95th percentile for length and speed (robust to outlier edges
        such as motorway slips or internal connector edges).

        All values floored at 1.0 to avoid division by zero on degenerate
        networks.
        """
        priorities = []
        lanes      = []
        lengths    = []
        speeds     = []

        for edge in self.net.getEdges():
            priorities.append(edge.getPriority())
            lanes.append(edge.getLaneNumber())
            lengths.append(edge.getLength())
            speeds.append(edge.getSpeed())

        stats = {
            "max_priority": float(max(priorities)) if priorities else 1.0,
            "max_lanes":    float(max(lanes))      if lanes      else 1.0,
            "max_length":   float(np.percentile(lengths, 95)) if lengths else 1.0,
            "max_speed":    float(np.percentile(speeds,  95)) if speeds  else 1.0,
        }

        # Floor at 1.0 to prevent division by zero
        return {k: max(v, 1.0) for k, v in stats.items()}

    # ── Internal topology construction ────────────────────────────────────

    def _build_topology(self) -> HeteroGraphTopology:
        tls_list   = list(self.net.getTrafficLights())
        agent_ids  = sorted(tls.getID() for tls in tls_list)
        node_index = {aid: idx for idx, aid in enumerate(agent_ids)}
        N = len(agent_ids)

        # ── Find connections via BFS ───────────────────────────────────────
        raw_connections = self._find_connections(agent_ids)
        C = len(raw_connections)

        # ── Build connection feature matrix ───────────────────────────────
        connection_features_list = []
        connection_meta          = []
        edges_src = []
        edges_dst = []

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
            edges_src           +
            [d for d in edges_dst] +
            conn_to_int_src     +
            conn_to_int_dst
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
            agent_ids, node_index, raw_connections
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
            norm_stats=self._norm,
        )

    def _find_connections(self, agent_ids: list) -> list:
        """
        BFS from each TLS junction to find all reachable TLS junctions
        within max_hops road segments, accumulating path features.
        """
        node_index  = set(agent_ids)
        connections = []
        seen_pairs  = set()

        for start_tls_id in agent_ids:
            start_node = self._get_tls_node(start_tls_id)
            if start_node is None:
                continue

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
                                "from_tls":   start_tls_id,
                                "to_tls":     to_tls_id,
                                "mean_speed": new_path["speed_sum"] /
                                              new_path["num_hops"],
                                **{k: v for k, v in new_path.items()
                                   if k != "speed_sum"},
                            })
                    else:
                        queue.append((to_node, hops + 1, new_path))

        return connections

    def _connection_features(self, conn: dict) -> np.ndarray:
        """
        Build normalised 7-feature vector using network-derived stats.
        All values clipped to [0, 1].
        """
        norm = self._norm
        return np.clip(np.array([
            conn["first_priority"] / norm["max_priority"],
            conn["min_lanes"]      / norm["max_lanes"],
            conn["total_length"]   / norm["max_length"],
            conn["min_speed"]      / norm["max_speed"],
            conn["mean_speed"]     / norm["max_speed"],
            conn["num_hops"]       / self.max_hops,
            float(conn["any_signalised"]),
        ], dtype=np.float32), 0.0, 1.0)

    def _build_proximity_matrix(
        self,
        agent_ids: list,
        node_index: dict,
        connections: list,
    ) -> torch.Tensor:
        """
        N x N proximity matrix using exponential decay on total path length.
        proximity[i][j] = exp(-total_length / scale).
        Diagonal is 0. Unconnected pairs are 0.
        Scale is set to the median connected path length so decay is
        network-relative rather than hardcoded.
        """
        N = len(agent_ids)
        matrix = torch.zeros(N, N)

        if not connections:
            return matrix

        lengths = [c["total_length"] for c in connections]
        scale   = float(np.median(lengths)) or 200.0

        for conn in connections:
            i = node_index[conn["from_tls"]]
            j = node_index[conn["to_tls"]]
            w = float(np.exp(-conn["total_length"] / scale))
            matrix[i][j] = w
            matrix[j][i] = w

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