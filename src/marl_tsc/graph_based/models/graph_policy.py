# graph_policy.py

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