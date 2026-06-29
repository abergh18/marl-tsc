"""
graph_rollout.py

Utilities for converting environment transitions into rollout batches
suitable for optimisation.

Why this exists
---------------
GraphRunner collects transitions as Python objects. This is useful for
debugging and inspection, but learning algorithms typically operate on
batched tensors.

This module converts a list of Transition objects into a RolloutBatch,
providing a consistent interface for PPO-style optimisation, return
calculation, advantage estimation, and future training algorithms.

The rollout representation is intentionally independent of any specific
learning algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class RolloutBatch:

    observations: list

    actions: torch.Tensor

    log_probs:torch.Tensor

    logits: torch.Tensor

    values: torch.Tensor

    rewards: torch.Tensor

    dones: torch.Tensor


class GraphRollout:

    @staticmethod
    def from_transitions(
        transitions,
        agent_ids,
    ) -> RolloutBatch:

        observations = []

        actions = []

        log_probs = []

        logits = []

        values = []

        rewards = []

        dones = []

        for transition in transitions:

            observations.append(
                transition.observation
            )

            actions.append(
                [
                    transition.action_dict[agent]
                    for agent in agent_ids
                ]
            )

            rewards.append(
                [
                    transition.reward_dict[agent]
                    for agent in agent_ids
                ]
            )

            log_probs.append(
              transition.log_prob
            )

            logits.append(
                transition.logits
            )

            values.append(
                transition.value
            )

            dones.append(
                float(transition.done)
            )

        return RolloutBatch(
            observations=observations,

            actions=torch.tensor(
                actions,
                dtype=torch.long,
            ),

            log_probs=torch.stack(
              log_probs
            ),

            logits=torch.stack(
                logits
            ),

            values=torch.stack(
                values
            ).squeeze(-1),

            rewards=torch.tensor(
                rewards,
                dtype=torch.float32,
            ),

            dones=torch.tensor(
                dones,
                dtype=torch.float32,
            ),
        )