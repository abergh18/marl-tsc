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

CHANGED (truncation bootstrap fix)
------------------------------------
Previous versions seeded the backward GAE recursion with
`next_value = 0`, which is only correct if the *last* transition in
the rollout is a true terminal state (episode actually ended there).

In this environment, episodes run for ~600 steps but rollouts are only
64 steps -- `dones` is essentially always 0 across an entire rollout.
Seeding `next_value = 0` at the end of every rollout was implicitly
telling GAE "the episode ends here with zero future value", which is
false: the episode is merely truncated mid-flight. Since rewards are
strictly non-positive in this env, that false "zero future value"
assumption combined with 64 steps of discounted negative reward
produced a structural negative bias that compounded every single
rollout, with no mechanism to correct itself -- this was the actual
cause of the monotonic ValueNorm drift and critic divergence seen in
training.

The fix: accept the critic's own value estimate for the observation
*immediately following* the last collected transition, and use that
as the bootstrap seed instead of zero. The caller (collect_batch) is
responsible for fetching this value from the policy before truncating
the rollout. When a transition genuinely IS a terminal state, `dones`
correctly zeroes out the bootstrap term in the recursion regardless of
what bootstrap_value is, so passing a real value here is always safe.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class AdvantageBatch:
    advantages: torch.Tensor  # (T, N)
    returns: torch.Tensor     # (T, N)


class AdvantageEstimator:

    @staticmethod
    def compute_gae(
        rollout_batch,
        bootstrap_value: torch.Tensor,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> AdvantageBatch:
        """
        Parameters
        ----------
        bootstrap_value : torch.Tensor, shape (N,)
            The critic's value estimate for the state immediately
            following the last transition in this rollout. Pass a
            zero tensor only if you are certain the rollout's final
            transition is a genuine terminal state (dones[-1] == 1).
            Otherwise this MUST be a real value estimate, or the
            recursion will systematically underestimate returns for
            rollouts that end mid-episode (truncation, not
            termination).
        """

        rewards = rollout_batch.rewards  # (T, N)
        values = rollout_batch.values    # (T, N) -- per-node critic outputs
        dones = rollout_batch.dones      # (T,)

        num_agents = rewards.shape[1]

        advantages = torch.zeros_like(rewards)  # (T, N)

        gae = torch.zeros(num_agents)  # (N,)

        # CHANGED: seed with the real bootstrap value, not zero.
        # If dones[-1] == 1 (genuine terminal state), the mask in the
        # first loop iteration below will zero out this term anyway,
        # so passing a real value is always safe regardless of
        # whether the last step happened to be terminal or truncated.
        next_value = bootstrap_value  # (N,)
        amp = 5.0
        for t in reversed(range(len(rewards))):
            mask = 1.0 - dones[t]  # scalar, broadcasts against (N,)
          
            delta = (
                amp * rewards[t]                   # (N,)
                + gamma * next_value * mask   # (N,)
                - values[t]                   # (N,)
            )

            gae = delta + gamma * gae_lambda * mask * gae  # (N,)

            advantages[t] = gae

            next_value = values[t]  # (N,)

        returns = advantages + values  # (T, N)

        return AdvantageBatch(
            advantages=advantages,
            returns=returns,
        )
