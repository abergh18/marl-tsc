import torch.nn as nn


class CriticHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        self.value = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, encoder_output):

        return self.value(
            encoder_output.graph_embedding
        )