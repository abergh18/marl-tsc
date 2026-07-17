"""
hetero_graph_builder.py

A drop-in companion to GraphBuilder that enriches the intersection graph
with connection nodes representing the road segments between intersections.

Node layout in the output Data object
--------------------------------------
Indices 0 .. N-1          : intersection (agent) nodes
Indices N .. N+C-1        : connection nodes

Both node types are projected to a shared embedding dimension so the
existing homogeneous GAT encoder can be used unchanged.

Edge layout
-----------
intersection_i --> connection_k --> intersection_j

replaces the direct

intersection_i --> intersection_j

of the original GraphBuilder.  Both directions are included so the graph
remains undirected from the GAT's perspective.

Connection node features (static, built once from .net.xml)
------------------------------------------------------------
0  priority          normalised road priority  (residential=3 .. primary=12)
1  num_lanes         number of lanes on the connecting edge
2  length            road segment length  (metres, normalised)
3  speed_limit       speed limit  (m/s, normalised)
4  dir_straight      binary: connection direction is straight
5  dir_left          binary: left turn
6  dir_right         binary: right turn
7  dir_turn          binary: U-turn
8  is_signalised     binary: connection is controlled by a TLS

Intersection node features
--------------------------
Passed in at runtime from SUMO observations (same as existing GraphBuilder).
A learned linear projection maps them to CONNECTION_DIM before concatenation
so both node types share the same feature dimension fed to the GAT.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sumolib.net import readNet
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONNECTION_FEAT_DIM = 9          # dimension of raw connection node features
_MAX_PRIORITY      = 12.0        # highway.primary priority in SUMO
_MAX_LENGTH        = 500.0       # normalisation cap for road length (m)
_MAX_SPEED         = 33.33       # normalisation cap ~120 km/h (m/s)
_MAX_LANES         = 4.0         # normalisation cap for lane count

_DIR_MAP = {
    "s": (1, 0, 0, 0),   # straight
    "l": (0, 1, 0, 0),   # left
    "L": (0, 1, 0, 0),   # partial left — treated as left
    "r": (0, 0, 1, 0),   # right
    "R": (0, 0, 1, 0),   # partial right — treated as right
    "t": (0, 0, 0, 1),   # U-turn
}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class HeteroGraphTopology:
    """
    Static topology built once from the network file.

    Parameters
    ----------
    agent_ids
        Ordered list of TLS intersection IDs.  Matches node indices 0..N-1.
    node_index
        Maps agent_id -> index in [0, N).
    connection_features
        Float tensor of shape [C, CONNECTION_FEAT_DIM] for the C connection
        nodes (indices N..N+C-1 in the combined graph).
    connection_meta
        List of (from_tls_id, to_tls_id, edge_id) for each connection node,
        in the same order as connection_features rows.  Useful for the
        proximity matrix and for logging.
    edge_index
        Combined edge_index [2, E] covering both directions of
        intersection -> connection -> intersection paths.
    num_intersections
        N — number of agent/intersection nodes.
    num_connections
        C — number of connection nodes.
    agent_mask
        Boolean tensor of shape [N+C] — True for intersection nodes,
        False for connection nodes.  Used by the policy to select only
        agent embeddings for action/value heads.
    """
    agent_ids:           list
    node_index:          dict
    connection_features: torch.Tensor
    connection_meta:     list
    edge_index:          torch.Tensor
    num_intersections:   int
    num_connections:     int
    agent_mask:          torch.Tensor


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

class HeteroGraphBuilder:
    """
    Builds an enriched graph that inserts connection nodes between
    intersections while remaining compatible with the existing homogeneous
    GAT encoder via a learned projection layer.

    Parameters
    ----------
    network_file
        Path to the SUMO .net.xml file.
    intersection_obs_dim
        Dimension of the per-agent SUMO observation vector (runtime).
        Required so the projection layer can be sized correctly.
    shared_dim
        Dimension of the shared embedding space fed to the GAT.
        Both intersection and connection nodes are projected to this size.
    """

    def __init__(
        self,
        network_file: str,
        intersection_obs_dim: int,
        shared_dim: int = 64,
    ):
        self.network_file        = Path(network_file)
        self.intersection_obs_dim = intersection_obs_dim
        self.shared_dim          = shared_dim

        self.net = readNet(str(self.network_file))

        # Projection layers — these are plain nn.Linear modules.
        # They should be included in the policy's parameter group so they
        # are trained end-to-end with the GAT encoder.
        self.intersection_proj = nn.Linear(intersection_obs_dim, shared_dim)
        self.connection_proj   = nn.Linear(CONNECTION_FEAT_DIM, shared_dim)

        # Build static topology once
        self.topology = self._build_topology()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(self) -> HeteroGraphTopology:
        """Return the pre-built static topology."""
        return self.topology

    def to_graph(
        self,
        observations: dict,
        topology: Optional[HeteroGraphTopology] = None,
    ) -> Data:
        """
        Build a PyG Data object compatible with the existing GAT encoder.

        Parameters
        ----------
        observations
            Dict mapping agent_id -> np.ndarray of SUMO observation features.
        topology
            Pre-built HeteroGraphTopology.  Defaults to self.topology.

        Returns
        -------
        torch_geometric.data.Data
            .x           : [N+C, shared_dim]  projected node features
            .edge_index  : [2, E]             combined edge index
            .agent_mask  : [N+C]              True for intersection nodes
        """
        if topology is None:
            topology = self.topology

        # --- Intersection node features (runtime) -------------------------
        intersection_x = torch.tensor(
            np.stack([
                observations[agent_id]
                for agent_id in topology.agent_ids
            ]),
            dtype=torch.float32,
        )                                                    # [N, obs_dim]
        intersection_emb = self.intersection_proj(
            intersection_x
        )                                                    # [N, shared_dim]

        # --- Connection node features (static) ----------------------------
        connection_emb = self.connection_proj(
            topology.connection_features
        )                                                    # [C, shared_dim]

        # --- Combine ------------------------------------------------------
        x = torch.cat([intersection_emb, connection_emb], dim=0)  # [N+C, shared_dim]

        return Data(
            x=x,
            edge_index=topology.edge_index,
            agent_mask=topology.agent_mask,
        )

    # ------------------------------------------------------------------
    # Internal topology construction
    # ------------------------------------------------------------------

    def _build_topology(self) -> HeteroGraphTopology:
        # ---- Intersection nodes -----------------------------------------
        agent_ids = sorted(
            tls.getID()
            for tls in self.net.getTrafficLights()
        )
        node_index = {
            agent_id: idx
            for idx, agent_id in enumerate(agent_ids)
        }
        N = len(agent_ids)

        # ---- Connection nodes -------------------------------------------
        # One connection node per (from_tls, to_tls, edge) triple.
        # Multiple edges between the same pair of intersections each get
        # their own connection node — this preserves lane-level detail.
        connection_features_list = []
        connection_meta          = []   # (from_tls_id, to_tls_id, edge_id)
        conn_node_idx            = {}   # (from_tls_id, to_tls_id, edge_id) -> int

        edges_src = []   # intersection index
        edges_dst = []   # connection node index (offset by N after loop)

        for tls in self.net.getTrafficLights():
            from_id = tls.getID()
            for edge in tls.getEdges():
                for outgoing_edge, _ in edge.getOutgoing().items():
                    neighbour_tls = outgoing_edge.getTLS()
                    if neighbour_tls is None:
                        continue
                    to_id = neighbour_tls.getID()
                    if from_id == to_id:
                        continue

                    key = (from_id, to_id, outgoing_edge.getID())
                    if key in conn_node_idx:
                        continue

                    c_idx = len(connection_meta)
                    conn_node_idx[key] = c_idx
                    connection_meta.append(key)

                    feat = self._connection_features(outgoing_edge)
                    connection_features_list.append(feat)

                    # intersection -> connection (directed)
                    edges_src.append(node_index[from_id])
                    edges_dst.append(c_idx)   # offset applied below

        C = len(connection_meta)

        # Offset connection node indices by N
        edges_dst_offset = [idx + N for idx in edges_dst]

        # Build connection -> destination intersection edges
        # For each connection node, find the to_tls and add reverse edge
        conn_to_intersection_src = []
        conn_to_intersection_dst = []
        for c_idx, (from_id, to_id, _) in enumerate(connection_meta):
            conn_to_intersection_src.append(c_idx + N)
            conn_to_intersection_dst.append(node_index[to_id])

        # Combine all edges (both directions for undirected behaviour)
        all_src = (
            edges_src                   +   # intersection -> connection
            edges_dst_offset            +   # connection -> intersection (reverse of above)
            conn_to_intersection_src    +   # connection -> intersection
            conn_to_intersection_dst        # intersection -> connection (reverse of above)
        )
        all_dst = (
            edges_dst_offset            +
            edges_src                   +
            conn_to_intersection_dst    +
            conn_to_intersection_src
        )

        edge_index = torch.tensor(
            [all_src, all_dst],
            dtype=torch.long,
        ).contiguous()

        # ---- Node features -----------------------------------------------
        connection_features = torch.tensor(
            np.stack(connection_features_list),
            dtype=torch.float32,
        ) if connection_features_list else torch.zeros(0, CONNECTION_FEAT_DIM)

        # ---- Agent mask --------------------------------------------------
        agent_mask = torch.zeros(N + C, dtype=torch.bool)
        agent_mask[:N] = True

        return HeteroGraphTopology(
            agent_ids=agent_ids,
            node_index=node_index,
            connection_features=connection_features,
            connection_meta=connection_meta,
            edge_index=edge_index,
            num_intersections=N,
            num_connections=C,
            agent_mask=agent_mask,
        )

    def _connection_features(self, edge) -> np.ndarray:
        """
        Extract and normalise static features for a single connection edge.
        """
        priority    = edge.getPriority() / _MAX_PRIORITY
        num_lanes   = edge.getLaneNumber() / _MAX_LANES
        length      = min(edge.getLength(), _MAX_LENGTH) / _MAX_LENGTH
        speed_limit = min(edge.getSpeed(), _MAX_SPEED) / _MAX_SPEED

        # Direction — take the most common direction across all connections
        # on this edge (most edges have one dominant direction)
        directions = [
            conn.getDirection()
            for conn in edge.getConnections()
        ]
        dir_counts = {}
        for d in directions:
            dir_counts[d] = dir_counts.get(d, 0) + 1
        dominant_dir = max(dir_counts, key=dir_counts.get) if dir_counts else "s"
        dir_straight, dir_left, dir_right, dir_turn = _DIR_MAP.get(
            dominant_dir, (1, 0, 0, 0)
        )

        # Signalisation
        is_signalised = float(edge.getTLS() is not None)

        return np.array([
            priority,
            num_lanes,
            length,
            speed_limit,
            float(dir_straight),
            float(dir_left),
            float(dir_right),
            float(dir_turn),
            is_signalised,
        ], dtype=np.float32)
