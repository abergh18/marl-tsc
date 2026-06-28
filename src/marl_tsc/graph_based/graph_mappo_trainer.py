"""
graph_mappo_trainer.py

MAPPO (Multi-Agent PPO) trainer using graph-based policies.

Why this exists
---------------
PPO is one of the most stable and widely-used policy gradient algorithms.
This trainer brings PPO's clipping and entropy regularization to the
graph framework, enabling comparison with CTDE and other methods.

Unlike the standard MAPPO (which uses flat MLP agents and a centralized
critic over concatenated observations), GraphMAPPOTrainer uses:

- Graph-encoded observations for each agent
- Per-agent actor heads (local policy)
- Per-agent critic heads (local value functions)
- PPO's clipping mechanism for policy stability
- Entropy regularization for exploration

This is a natural hybrid: graph-based perception with PPO's training
stability.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_trainer import BaseGraphTrainer


class GraphMAPPOTrainer(BaseGraphTrainer):
    """
    Graph-based MAPPO trainer.

    Combines:
    - Graph encoder (e.g., GAT) for feature extraction
    - Per-agent actor and critic heads
    - PPO's clipped policy gradient for stability
    - Entropy regularization for exploration
    """

    def __init__(
        self,
        env,
        policy,
        optimizer,
        rollout_steps=64,
        gae_lambda=0.95,
        gamma=0.99,
        clip_ratio=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        max_grad_norm=0.5,
        update_epochs=3,
    ):
        """
        Parameters
        ----------
        env : GraphTrafficEnv
            The environment.
        policy : GraphPolicy
            The policy (encoder + actor/critic heads).
        optimizer : torch.optim.Optimizer
            Optimizer for all parameters.
        rollout_steps : int
            Number of steps per rollout.
        gae_lambda : float
            GAE lambda parameter.
        gamma : float
            Discount factor.
        clip_ratio : float
            PPO clip ratio (epsilon).
        entropy_coef : float
            Entropy regularization coefficient.
        value_coef : float
            Value loss coefficient.
        max_grad_norm : float
            Gradient clipping norm.
        update_epochs : int
            Number of update epochs per rollout.
        """
        super().__init__(
            env=env,
            policy=policy,
            rollout_steps=rollout_steps,
            gae_lambda=gae_lambda,
        )

        self.optimizer = optimizer
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs

    def update(self, rollout_batch, advantage_batch):
        """
        PPO update step.

        Performs multiple epochs of policy and value updates using the
        PPO clipping mechanism.

        Parameters
        ----------
        rollout_batch : RolloutBatch
            Collected transitions.
        advantage_batch : AdvantageBatch
            Computed advantages and returns.

        Returns
        -------
        dict
            Training statistics.
        """

        advantages = advantage_batch.advantages
        returns = advantage_batch.returns

        # Normalize advantages for stable training
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        actor_losses = []
        value_losses = []
        entropy_losses = []
        policy_clip_fracs = []

        for epoch in range(self.update_epochs):

            for t, graph_obs in enumerate(rollout_batch.observations):

                # Forward pass through policy
                output = self.policy(graph_obs)

                # Extract data for this timestep
                actions = rollout_batch.actions[t]  # (num_agents,)
                old_log_probs = rollout_batch.log_probs[t]  # (num_agents,)
                logits = output.logits  # (num_agents, action_dim)
                values = output.value.squeeze(-1)  # (num_agents,)

                # Policy update
                dist = Categorical(logits=logits)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                # PPO clipping
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages[t]
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_ratio,
                        1.0 + self.clip_ratio,
                    )
                    * advantages[t]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value update
                value_loss = F.mse_loss(values, returns[t])

                # Total loss
                total_loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                # Record statistics
                with torch.no_grad():
                    clip_frac = (
                        (
                            torch.abs(ratio - 1.0)
                            > self.clip_ratio
                        )
                        .float()
                        .mean()
                    )

                actor_losses.append(policy_loss.detach())
                value_losses.append(value_loss.detach())
                entropy_losses.append(entropy.detach())
                policy_clip_fracs.append(clip_frac.detach())

        # Aggregate statistics
        result = {
            "actor_loss": float(
                torch.stack(actor_losses).mean()
            ),
            "value_loss": float(
                torch.stack(value_losses).mean()
            ),
            "entropy_loss": float(
                torch.stack(entropy_losses).mean()
            ),
            "policy_clip_fraction": float(
                torch.stack(policy_clip_fracs).mean()
            ),
            "mean_training_reward": float(
                rollout_batch.rewards.mean()
            ),
            "rollout_length": len(rollout_batch.observations),
            "update_epochs": self.update_epochs,
        }

        return result
