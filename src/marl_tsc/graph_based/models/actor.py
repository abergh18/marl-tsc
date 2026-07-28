import torch.nn as nn
from typing import Union


class ActorHead(nn.Module):
    """
    An Actor head designed to output logits for single or multiple discrete actions.
    """

    def __init__(
        self,
        embedding_dim: int,
        action_dims: Union[int, list[int]],
    ):
        """
        Initialise the actor with a shared hidden layer and separate branches 
        for each action.

        Args:
            embedding_dim: The size of the input node embeddings.
            action_dims: A list containing the sizes of each action space 
                         (e.g., [8, 11] for 8 phases and 11 sharing options),
                         or a single integer for standard environments.
        """
        super().__init__()
        
        # Ensure action_dims is always a list to prevent crashes
        if isinstance(action_dims, int):
            action_dims = [action_dims]

        # The group's shared hidden layer to increase the network's capacity
        self.shared_net = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
        )

        # Your separate linear layers (branches) for each discrete action
        self.branches = nn.ModuleList([
            nn.Linear(64, dim) for dim in action_dims
        ])

    def forward(self, encoder_output):
        """
        Calculate the logits for all action branches.
        """
        # Extract the node embeddings
        x = encoder_output.node_embeddings
        
        # Pass the embeddings through the shared hidden layer
        x = self.shared_net(x)

        # Pass the processed embeddings through each branch independently
        logits_list = [branch(x) for branch in self.branches]

        return logits_list