import torch.nn as nn


class ActorHead(nn.Module):
    """
    An Actor head designed to output logits for multiple discrete actions.
    """

    def __init__(self, embedding_dim, action_dims):
        """
        Initialise the actor with separate branches for each action.

        Args:
            embedding_dim: The size of the input node embeddings.
            action_dims: A list containing the sizes of each action space 
                         (e.g., [8, 11] for 8 phases and 11 sharing options).
        """
        super().__init__()
        
        # Create a separate linear layer (branch) for each discrete action
        self.branches = nn.ModuleList([
            nn.Linear(embedding_dim, dim) for dim in action_dims
        ])

    def forward(self, encoder_output):
        """
        Calculate the logits for all action branches.
        """
        x = encoder_output.node_embeddings
        
        # Pass the embeddings through each branch independently
        logits_list = [branch(x) for branch in self.branches]
        
        return logits_list
