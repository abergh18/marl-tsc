"""
graph_ctde_trainer.py

Centralized Training, Decentralized Execution (CTDE) trainer for
graph-based traffic signal control.

This implementation intentionally uses a simple actor-critic update
rather than PPO. The goal is to provide a clear and extensible baseline
for graph-based MARL experiments.

Architecture:

    GraphObservation
            ↓
        GATEncoder
            ↓
      Actor Head
      Critic Head
            ↓
       Actor-Critic
            ↓
      Gradient Update

The critic learns a global value estimate for the traffic network while
the actor learns decentralized traffic-light policies.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_trainer import BaseGraphTrainer

class GraphCTDETrainer(BaseGraphTrainer):

  def __init__(
      self,
      env,
      policy,
      optimizer,
      rollout_steps=64,
      #gamma=0.99,
      gae_lambda=0.95,
  ):

    super().__init__(
        env=env,
        policy=policy,
        optimizer=optimizer,
        rollout_steps=rollout_steps,
        #gamma=gamma,
        gae_lambda=gae_lambda,
    )
  
  def update(
    self,
    rollout_batch,
    advantage_batch
    ):

      advantages = advantage_batch.advantages
      returns = advantage_batch.returns

      advantages = (
          advantages
          - advantages.mean()
      ) / (
          advantages.std()
          + 1e-8
      )

      # -----------------------------
      # Replay observations through
      # current network
      # -----------------------------

      actor_losses = []
      critic_losses = []

      for t, graph_obs in enumerate(
          rollout_batch.observations
      ):

          output = self.policy(
              graph_obs
          )

          dist = Categorical(
              logits=output.logits
          )

          actions = rollout_batch.actions[t]

          log_probs = dist.log_prob(
              actions
          )

          #
          # Actor
          #

          actor_loss = -(
              log_probs.sum()
              * advantages[t].detach()
          )

          #
          # Critic
          #

          critic_loss = F.mse_loss(
              output.value.squeeze(),
              returns[t].detach(),
          )

          actor_losses.append(
              actor_loss
          )

          critic_losses.append(
              critic_loss
          )

      actor_loss = torch.stack(
          actor_losses
      ).mean()

      critic_loss = torch.stack(
          critic_losses
      ).mean()

      total_loss = (
          actor_loss
          + critic_loss
      )
      mean_reward = float(
      rollout_batch.rewards.mean()
      )

      self.optimizer.zero_grad()

      total_loss.backward()

      self.optimizer.step()

      result = {
          "actor_loss": float(
              actor_loss.detach()
          ),
          "critic_loss": float(
              critic_loss.detach()
          ),
          "total_loss": float(
              total_loss.detach()
          ),
          "rollout_length": len(
              rollout_batch.observations
          ),
          "mean_training_reward": float(
              rollout_batch.rewards.mean()
          ),
      }

      print("UPDATE RETURN:", result)

      return result