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

from .graph_runner import GraphRunner
from .graph_rollout import GraphRollout
from .advantage_estimator import AdvantageEstimator


class GraphCTDETrainer:

    def __init__(
        self,
        env,
        policy,
        optimizer,
        rollout_steps: int = 64,
    ):
        self.env = env
        self.policy = policy
        self.optimizer = optimizer
        self.rollout_steps = rollout_steps

    def train_step(self):

        # -----------------------------
        # Collect rollout
        # -----------------------------

        runner = GraphRunner(
            env=self.env,
            policy=self.policy,
        )

        rollout = runner.collect_rollout(
            num_steps=self.rollout_steps,
        )

        batch = GraphRollout.from_transitions(
            rollout,
            self.env.agent_ids,
        )

        adv_batch = (
            AdvantageEstimator
            .compute_gae(batch)
        )

        advantages = adv_batch.advantages
        returns = adv_batch.returns

        # -----------------------------
        # Replay observations through
        # current network
        # -----------------------------

        actor_losses = []
        critic_losses = []

        for t, graph_obs in enumerate(
            batch.observations
        ):

            output = self.policy(
                graph_obs
            )

            dist = Categorical(
                logits=output.logits
            )

            actions = batch.actions[t]

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

        self.optimizer.zero_grad()

        total_loss.backward()

        self.optimizer.step()

        return {
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
                rollout
            ),
        }