import torch
import torch.nn as nn


class CriticHead(nn.Module):
    """
    Per-agent critic head for CTDE (Centralised Training with Decentralised Execution).

    Outputs one value per node/agent directly from a concatenation of
    that node's own embedding and the global graph embedding. This keeps 
    the critic, the GAE bootstrap, and the actor's advantages all in the 
    same per-agent shape throughout, whilst still centralising the value 
    function by sharing the global graph state.
    """

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        # Combine node embedding + graph embedding
        input_dim = embedding_dim * 2

        self.value = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        encoder_output,
    ):
        node_embeddings = encoder_output.node_embeddings  # (N, D)
        graph_embedding = encoder_output.graph_embedding  # (D,)

        # Repeat graph embedding for every node
        graph_embedding = graph_embedding.unsqueeze(0).expand(
            node_embeddings.size(0),
            -1,
        )  # (N, D)

        # Concatenate local and global information
        critic_input = torch.cat(
            [
                node_embeddings,
                graph_embedding,
            ],
            dim=1,
        )  # (N, 2D)

        return self.value(critic_input)