import torch.nn as nn


class CriticHead(nn.Module):
    """
    Per-agent critic head for CTDE.

    CHANGED: previously this pooled all node embeddings into a single
    graph-level vector (mean-pool) and produced one scalar value for
    the whole team. That created a shape mismatch once advantages
    became per-agent: per-agent GAE recursion was bootstrapping off a
    single shared value baseline that was never trained to predict
    per-agent returns, and the mismatch compounded across the rollout
    until training diverged.

    Now the critic outputs one value per node/agent directly from
    that node's own embedding -- mirroring ActorHead's per-node
    output. This keeps the critic, the GAE bootstrap, and the actor's
    advantages all in the same per-agent shape throughout.

    NOTE: this is no longer a "centralised" value function in the
    strict sense of pooling the whole graph into one number. Each
    node's value still benefits from the GAT's message passing (so
    it indirectly reflects neighbouring agents' states), but the
    output itself is per-agent. This is the standard way CTDE critics
    are implemented when you want per-agent advantages -- the
    "centralisation" comes from the shared GAT encoder seeing the
    whole graph, not from pooling to a single scalar.
    """

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
        # CHANGED: use node_embeddings (per-node), not graph_embedding
        # (pooled). Output shape: (num_nodes, 1) -> squeeze handled by
        # caller, matching ActorHead's convention.
        return self.value(
            encoder_output.node_embeddings
        )
