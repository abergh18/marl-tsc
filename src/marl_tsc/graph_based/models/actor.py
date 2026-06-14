import torch.nn as nn


class ActorHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
        action_dim: int,
    ):
        super().__init__()

        self.policy = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, encoder_output):

        return self.policy(
            encoder_output.node_embeddings
        )