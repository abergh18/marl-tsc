"""
multi_hop_graph_builder.py

Drop-in replacement for GraphBuilder that uses BFS to find TLS-to-TLS
connections via intermediate non-TLS junctions.

Outputs standard GraphTopology — compatible with GraphTrafficEnv and
train_true_mappo without any changes to downstream code.

Why this exists
---------------
GraphBuilder uses getOutgoing() which only finds direct TLS-to-TLS edges.
On real OSM networks most intersections connect via intermediate non-TLS
junctions, leaving most agents isolated (degree 0) and making GAT message
passing useless.

MultiHopGraphBuilder uses BFS up to max_hops to find paths between TLS
junctions through intermediate nodes, producing a much richer edge_index.

The output is identical in structure to GraphBuilder.build() — same
GraphTopology dataclass, same fields — so all downstream code is unaffected.
"""

from __future__ import annotations

from pathlib import Path

import torch
from sumolib.net import readNet

from .graph_builder import GraphTopology


class MultiHopGraphBuilder:
    """
    GraphBuilder replacement using BFS for TLS-to-TLS connection discovery.

    Parameters
    ----------
    network_file : str
        Path to the SUMO .net.xml file.
    max_hops : int
        Maximum BFS depth when searching for TLS-to-TLS connections.
        max_hops=1 is equivalent to GraphBuilder (direct edges only).
        max_hops=3 (default) finds connections via up to 3 road segments.
    """

    def __init__(self, network_file: str, max_hops: int = 3):
        self.network_file = Path(network_file)
        self.max_hops     = max_hops
        self.net          = readNet(str(self.network_file))

    def build(self) -> GraphTopology:
        """
        Build GraphTopology with multi-hop edges.

        Returns
        -------
        GraphTopology
            Same interface as GraphBuilder.build(). Compatible with
            GraphTrafficEnv, GraphRunner, and all downstream code.
        """
        agent_ids  = sorted(
            tls.getID() for tls in self.net.getTrafficLights()
        )
        node_index = {aid: idx for idx, aid in enumerate(agent_ids)}

        # BFS to find all TLS-to-TLS connections
        connections = self._find_connections(agent_ids)

        # Build undirected edge set
        edges = set()
        for conn in connections:
            i = node_index[conn["from_tls"]]
            j = node_index[conn["to_tls"]]
            edges.add((i, j))
            edges.add((j, i))

        if edges:
            edge_index = torch.tensor(
                list(edges), dtype=torch.long
            ).t().contiguous()
        else:
            edge_index = torch.zeros(2, 0, dtype=torch.long)

        return GraphTopology(
            agent_ids=agent_ids,
            edge_index=edge_index,
            node_index=node_index,
        )

    def _find_connections(self, agent_ids: list) -> list:
        """
        BFS from each TLS junction to find all reachable TLS junctions
        within max_hops road segments.

        Returns list of dicts with from_tls and to_tls keys.
        """
        node_index  = set(agent_ids)
        connections = []
        seen_pairs  = set()

        for start_tls_id in agent_ids:
            start_node = self._get_tls_node(start_tls_id)
            if start_node is None:
                continue

            queue   = [(start_node, 0)]
            visited = {start_node.getID()}

            while queue:
                current_node, hops = queue.pop(0)
                if hops >= self.max_hops:
                    continue

                for out_edge in current_node.getOutgoing():
                    to_node    = out_edge.getToNode()
                    to_node_id = to_node.getID()

                    if to_node_id in visited:
                        continue
                    visited.add(to_node_id)

                    to_tls_id = to_node.getTLSID()
                    if (to_tls_id
                            and to_tls_id in node_index
                            and to_tls_id != start_tls_id):
                        pair = (start_tls_id, to_tls_id)
                        if pair not in seen_pairs:
                            seen_pairs.add(pair)
                            connections.append({
                                "from_tls": start_tls_id,
                                "to_tls":   to_tls_id,
                            })
                    else:
                        queue.append((to_node, hops + 1))

        return connections

    def _get_tls_node(self, tls_id: str):
        """
        Get sumolib Node for a TLS ID, handling joinedS composite IDs.
        """
        if self.net.hasNode(tls_id):
            return self.net.getNode(tls_id)
        parts = tls_id.replace("joinedS_", "").split("_cluster_")
        for part in parts:
            for candidate in [part] + part.split("_"):
                if self.net.hasNode(candidate):
                    return self.net.getNode(candidate)
        return None