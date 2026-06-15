"""
advantage_estimator.py

Generalized Advantage Estimation (GAE) utilities.

Why this exists
---------------
Policy-gradient methods such as PPO do not train directly from rewards.
Instead they compute:

    Returns
    Advantages

Returns estimate the total future reward from a state.

Advantages estimate whether an action produced a better or worse outcome
than the critic expected.

This module converts a RolloutBatch into tensors suitable for PPO-style
optimisation while remaining independent of the training loop itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdvantageBatch:

    advantages: torch.Tensor

    returns: torch.Tensor


class AdvantageEstimator:

    @staticmethod
    def compute_gae(
        rollout_batch,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> AdvantageBatch:

        rewards = rollout_batch.rewards.mean(
            dim=1
        )

        values = rollout_batch.values

        dones = rollout_batch.dones

        advantages = torch.zeros_like(
            rewards
        )

        gae = 0.0

        next_value = 0.0

        for t in reversed(
            range(len(rewards))
        ):

            mask = 1.0 - dones[t]

            delta = (
                rewards[t]
                + gamma * next_value * mask
                - values[t]
            )

            gae = (
                delta
                + gamma
                * gae_lambda
                * mask
                * gae
            )

            advantages[t] = gae

            next_value = values[t]

        returns = (
            advantages
            + values
        )

        return AdvantageBatch(
            advantages=advantages,
            returns=returns,
        )