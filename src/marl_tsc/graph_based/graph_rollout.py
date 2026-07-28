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

Changes from original
---------------------
- Added gifting_actions, gifting_log_probs, gifting_stats fields to
  RolloutBatch. These are None when built from standard Transition
  objects (non-gifting runs) and populated when built from
  GiftingTransition objects. Non-gifting trainers ignore them entirely.
- Added empty transitions guard.
- values reshape made explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutBatch:
    observations: list
    actions: torch.Tensor
    log_probs: torch.Tensor
    logits: torch.Tensor
    values: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor

    # Gifting fields — None for non-gifting runs
    gifting_actions: Optional[torch.Tensor] = None      # (T, num_agents)
    gifting_log_probs: Optional[torch.Tensor] = None    # (T, num_agents)
    gifting_stats: Optional[list] = None                # list of per-step dicts


class GraphRollout:

    @staticmethod
    def from_transitions(
        transitions,
        agent_ids,
    ) -> RolloutBatch:

        if not transitions:
            raise ValueError(
                "Cannot build RolloutBatch from empty transitions list."
            )

        observations = []
        actions = []
        log_probs = []
        logits = []
        values = []
        rewards = []
        dones = []

        # Gifting — detected from first transition
        is_gifting = hasattr(transitions[0], "gifting_log_prob") and \
                     transitions[0].gifting_log_prob is not None
        
        gifting_actions = [] if is_gifting else None
        gifting_log_probs = [] if is_gifting else None
        gifting_stats = [] if is_gifting else None

        for transition in transitions:
            observations.append(transition.observation)

            # We retain the logic to pass the entire action dict so that 
            # multi-discrete actions are preserved for the trainer
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

            log_probs.append(transition.log_prob)
            logits.append(transition.logits)
            values.append(transition.value)
            dones.append(float(transition.done))

            # Safely capture the group's tracking metrics if they exist
            if is_gifting:
                if hasattr(transition, "gifting_action_dict"):
                    gifting_actions.append(
                        [
                            transition.gifting_action_dict[agent]
                            for agent in agent_ids
                        ]
                    )
                gifting_log_probs.append(transition.gifting_log_prob)
                
                if hasattr(transition, "gifting_stats"):
                    gifting_stats.append(transition.gifting_stats)

        return RolloutBatch(
            observations=observations,
            actions=torch.tensor(
                actions,
                dtype=torch.long,
            ),
            log_probs=torch.stack(log_probs),
            logits=torch.stack(logits),
            values=torch.stack(values).reshape(len(transitions), -1),
            rewards=torch.tensor(
                rewards,
                dtype=torch.float32,
            ),
            dones=torch.tensor(
                dones,
                dtype=torch.float32,
            ),
            gifting_actions=torch.tensor(
                gifting_actions,
                dtype=torch.long,
            ) if (is_gifting and len(gifting_actions) > 0) else None,
            gifting_log_probs=torch.stack(
                gifting_log_probs
            ) if is_gifting else None,
            gifting_stats=gifting_stats if is_gifting else None,
        )