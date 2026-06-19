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

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from .base_trainer import BaseGraphTrainer

class RunningMeanStd:

    def __init__(self):
        self.mean = 0.0
        self.var = 1.0
        self.count = 1e-4

    def update(self, x):

        x = np.asarray(x)

        batch_mean = x.mean()
        batch_var = x.var()
        batch_count = len(x)

        delta = batch_mean - self.mean

        total_count = self.count + batch_count

        new_mean = (
            self.mean
            + delta * batch_count / total_count
        )

        m_a = self.var * self.count
        m_b = batch_var * batch_count

        m2 = (
            m_a
            + m_b
            + delta**2
            * self.count
            * batch_count
            / total_count
        )

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    @property
    def std(self):
        return np.sqrt(self.var)


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
    self.value_normalizer = RunningMeanStd()
   
  def update(
    self,
    rollout_batch,
    advantage_batch
    ):

      advantages = advantage_batch.advantages
      returns = advantage_batch.returns
      
      self.value_normalizer.update(
      returns.cpu().numpy()
      )
      print(
        f"ValueNorm mean={self.value_normalizer.mean:.3f} "
        f"std={self.value_normalizer.std:.3f}"
      )

      normalized_returns = (
          returns
          - self.value_normalizer.mean
      ) / (
          self.value_normalizer.std
          + 1e-8
      )
      '''    
      print(
        f"Returns: mean={returns.mean():.4f} "
        f"std={returns.std():.4f} "
        f"min={returns.min():.4f} "
        f"max={returns.max():.4f}"
      ) 

      print(
        f"Advantages: mean={advantages.mean():.4f} "
        f"std={advantages.std():.4f} "
        f"min={advantages.min():.4f} "
        f"max={advantages.max():.4f}"
      )'''

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
          print(
              f"value={output.value.mean().item():.3f} "
              f"target={normalized_returns[t].item():.3f}"
          )

          #
          # Critic
          #

          critic_loss = F.mse_loss(
              output.value.squeeze(),
              normalized_returns[t].detach(),
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