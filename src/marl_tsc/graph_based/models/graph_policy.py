# graph_policy.py
"""
graph_policy.py

This module defines the high-level policy wrapper used by graph-based
traffic-signal-control agents.

Why this exists
---------------
The framework separates graph representation, feature extraction, and
decision making into independent components:

    GraphObservation
            ↓
        Encoder
            ↓
    EncoderOutput
        ↙       ↘
 Policy Head   Critic Head

GraphPolicy is responsible for composing these components into a single
forward pass. It does not implement graph learning itself, nor does it
contain any optimisation or training logic.

This separation is intentional. The intention is to explore a
range of graph-based approaches,
potentially including:

- Graph Attention Networks (GATs)
- Graph Transformers
- Hypergraph Neural Networks
- Heterogeneous Graph Networks
- Other experimental graph architectures

By isolating the encoder behind a common interface, new graph models can
be introduced without modifying policy heads, critic heads, or training
code. A GraphPolicy constructed with a GAT encoder should be
interchangeable with one using a hypergraph or heterogeneous encoder,
provided they return a compatible EncoderOutput.

This module therefore acts as the integration point between graph
representations and reinforcement-learning components while preserving
the flexibility required for future experimentation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch.nn as nn


@dataclass
class PolicyOutput:
    logits: object
    value: object
    encoder_output: object


class GraphPolicy(nn.Module):

    def __init__(
        self,
        encoder,
        actor_head,
        critic_head,
    ):
        super().__init__()

        self.encoder = encoder
        self.actor_head = actor_head
        self.critic_head = critic_head

    def forward(self, graph):

        encoder_output = self.encoder(graph)

        logits = self.actor_head(
            encoder_output
        )

        value = self.critic_head(
            encoder_output
        )

        return PolicyOutput(
            logits=logits,
            value=value,
            encoder_output=encoder_output,
        )