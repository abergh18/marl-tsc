import torch.nn as nn
import torch


class CriticHead(nn.Module):

    def __init__(
        self,
        embedding_dim: int,
    ):
        super().__init__()

        #
        # node embedding + graph embedding
        #
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

      node_embeddings = (
          encoder_output.node_embeddings
      )          # (N, D)

      graph_embedding = (
          encoder_output.graph_embedding
      )          # (D,)

      #
      # Repeat graph embedding for every node
      #
      graph_embedding = graph_embedding.unsqueeze(0).expand(
          node_embeddings.size(0),
          -1,
      )          # (N, D)

      #
      # Concatenate local and global information
      #
      critic_input = torch.cat(
          [
              node_embeddings,
              graph_embedding,
          ],
          dim=1,
      )          # (N, 2D)

      return self.value(
          critic_input
      )
    '''
    def forward(self, encoder_output):
      return self.value(
          encoder_output.node_embeddings
      )'''
