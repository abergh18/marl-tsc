"""
graph_rollout.py  — v2

Changes from v1
---------------
- per_agent_gifting field added to RolloutBatch — list of per-step
  dicts mapping agent_id to gift detail. None for non-gifting runs.
- Populated from GiftingTransition.per_agent_gifting in from_transitions().
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutBatch:
    observations:      list
    actions:           torch.Tensor
    log_probs:         torch.Tensor
    logits:            torch.Tensor
    values:            torch.Tensor
    rewards:           torch.Tensor
    dones:             torch.Tensor

    # Gifting fields — None for non-gifting runs
    gifting_actions:    Optional[torch.Tensor] = None
    gifting_log_probs:  Optional[torch.Tensor] = None
    gifting_stats:      Optional[list]         = None
    per_agent_gifting:  Optional[list]         = None   # NEW — list of per-step dicts


class GraphRollout:

    @staticmethod
    def from_transitions(transitions, agent_ids) -> RolloutBatch:

        if not transitions:
            raise ValueError(
                "Cannot build RolloutBatch from empty transitions list."
            )

        observations  = []
        actions       = []
        log_probs     = []
        logits        = []
        values        = []
        rewards       = []
        dones         = []

        is_gifting = (
            hasattr(transitions[0], "gifting_log_prob")
            and transitions[0].gifting_log_prob is not None
        )

        gifting_actions   = [] if is_gifting else None
        gifting_log_probs = [] if is_gifting else None
        gifting_stats     = [] if is_gifting else None
        per_agent_gifting = [] if is_gifting else None

        for transition in transitions:

            observations.append(transition.observation)

            actions.append([
                transition.action_dict[agent][0]
                if isinstance(transition.action_dict[agent], list)
                else transition.action_dict[agent]
                for agent in agent_ids
            ])

            rewards.append([
                transition.reward_dict[agent]
                for agent in agent_ids
            ])

            log_probs.append(transition.log_prob)
            logits.append(transition.logits)
            values.append(transition.value)
            dones.append(float(transition.done))

            if is_gifting:
                gifting_actions.append([
                    transition.gifting_action_dict[agent]
                    for agent in agent_ids
                ])
                gifting_log_probs.append(transition.gifting_log_prob)
                gifting_stats.append(transition.gifting_stats)
                per_agent_gifting.append(
                    getattr(transition, "per_agent_gifting", {})
                )

        return RolloutBatch(
            observations=observations,
            actions=torch.tensor(actions, dtype=torch.long),
            log_probs=torch.stack(log_probs),
            logits=torch.stack(logits),
            values=torch.stack(values).reshape(len(transitions), -1),
            rewards=torch.tensor(rewards, dtype=torch.float32),
            dones=torch.tensor(dones, dtype=torch.float32),
            gifting_actions=torch.tensor(
                gifting_actions, dtype=torch.long
            ) if is_gifting else None,
            gifting_log_probs=torch.stack(
                gifting_log_probs
            ) if is_gifting else None,
            gifting_stats=gifting_stats if is_gifting else None,
            per_agent_gifting=per_agent_gifting if is_gifting else None,
        )