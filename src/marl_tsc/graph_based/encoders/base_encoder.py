from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderOutput:
    """
    Generic graph encoder output.

    Future encoders can add richer metadata while
    preserving the same interface to actor/critic code.
    """

    node_embeddings: torch.Tensor
    graph_embedding: torch.Tensor

    metadata: dict | None = None


class BaseGraphEncoder(nn.Module, ABC):
    """
    Abstract base class for graph encoders.

    All graph encoders should return an EncoderOutput.
    """

    @abstractmethod
    def forward(self, graph) -> EncoderOutput:
        raise NotImplementedError