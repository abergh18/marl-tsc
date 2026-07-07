"""
graph_types.py

This module defines project-level graph abstractions used throughout the
graph-based MARL framework.

Why this exists
---------------
The current implementation uses torch_geometric and homogeneous graphs
represented by `torch_geometric.data.Data`. This is sufficient for the
initial graph-attention experiments, where traffic-light controllers are
represented as graph nodes connected according to the road network.

However, future work may require richer graph representations, including:

- Heterogeneous graphs with multiple node types
  (traffic lights, roads, vehicles, weather, events, etc.).

- Hypergraphs where relationships involve more than two entities.

- Multi-relational graphs with different edge types.

- Learned or dynamically changing graph structures.

If graph objects are passed directly throughout the codebase as PyG
`Data` objects, future changes to the graph representation would require
modifications across encoders, policies, training code, and utilities.

Instead, the framework passes GraphObservation objects. This creates a
stable interface between:

    Environment
        ↓
    Graph Representation
        ↓
    Encoder
        ↓
    Policy

Encoders can then decide how to interpret the underlying graph object,
whether it is a PyG Data object, HeteroData object, hypergraph
representation, or another structure entirely.

This abstraction is intended to minimise future refactoring as more
advanced graph-based approaches are explored.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class GraphObservation:
    """
    Container for graph observations produced by the environment.

    Parameters
    ----------
    graph
        Underlying graph representation. Currently expected to be a
        torch_geometric.data.Data object.

    agent_ids
        Ordered list of controllable agents represented within the graph.

    global_state
        Flat tensor of shape (num_agents * obs_dim,) containing all agent
        observations concatenated. Used by the centralised critic in true
        MAPPO. None for algorithms that do not require global state.

    metadata
        Optional auxiliary information associated with the graph.
        Future implementations may store node-type mappings,
        hyperedge information, attention masks, etc.
    """

    graph: object
    agent_ids: list[str]
    global_state: torch.Tensor | None = None
    metadata: dict = field(default_factory=dict)