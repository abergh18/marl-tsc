import torch
#import torch.nn as nn

from torch_geometric.nn import GATConv
from .base_encoder import (
    BaseGraphEncoder,
    EncoderOutput,
)

class GATEncoder(BaseGraphEncoder):

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int = 64,
        heads: int = 4,
    ):
        super().__init__()

        self.gat1 = GATConv(
            obs_dim,
            hidden_dim,
            heads=heads,
            concat=True,
        )

        self.gat2 = GATConv(
            hidden_dim * heads,
            hidden_dim,
            heads=1,
            concat=False,
        )

    def forward(self, graph):

        x = self.gat1(
            graph.x,
            graph.edge_index,
        )

        x = torch.relu(x)

        x = self.gat2(
            x,
            graph.edge_index,
        )

        x = torch.relu(x)

        graph_embedding = x.mean(dim=0)

        return EncoderOutput(
            node_embeddings=x,
            graph_embedding=graph_embedding,
        )