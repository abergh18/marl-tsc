"""
gifting_graph_runner.py  — v2

Changes from v1
---------------
- GiftingTransition now carries per_agent_gifting: dict mapping agent_id
  to its gifting action, gift_fraction, gift_amount, and raw_traffic_reward.
  This enables per-agent gifting logging in run_training and history.
- gifting_stats retains aggregate stats for backwards compatibility.
- public_goods contributions also captured per agent via infos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import numpy as np
from torch.distributions import Categorical

from marl_tsc.graph_based.graph_runner import GraphRunner, Transition


# ── Extended transition dataclass ─────────────────────────────────────────────

@dataclass
class GiftingTransition(Transition):
    """
    Extends Transition with gifting-specific fields.

    gifting_action_dict : dict[str, int]
        Discrete gifting action chosen by each agent this step.
    gifting_log_prob : torch.Tensor  (num_agents,)
        Log probability of each agent's gifting action.
    gifting_stats : dict
        Aggregate step-level gifting statistics.
    per_agent_gifting : dict[str, dict]
        Per-agent gifting detail:
            gift_action    : int   — discrete gifting action chosen
            gift_fraction  : float — fraction of reward gifted
            gift_amount    : float — absolute amount gifted
            raw_reward     : float — traffic reward before redistribution
    """
    gifting_action_dict: dict  = field(default_factory=dict)
    gifting_log_prob:    torch.Tensor = None
    gifting_stats:       dict  = field(default_factory=dict)
    per_agent_gifting:   dict  = field(default_factory=dict)


# ── Runner ────────────────────────────────────────────────────────────────────

class GiftingGraphRunner(GraphRunner):
    """
    GraphRunner subclass for reward sharing experiments.

    Assumes:
    - env is wrapped with PeerRewardingWrapper or ZeroSumRewardWrapper
    - policy is GiftingMAPPOPolicy (has .branches[0] and .branches[1])
    """

    def collect_rollout(self, num_steps: int, seed: int | None = None):
        from torch_geometric.data import Data
        from marl_tsc.graph_based.graph_types import GraphObservation

        rollout = []

        if self._current_obs is None:
            graph_obs, infos = self.env.reset(seed=seed)
            self._current_obs   = graph_obs
            self._current_infos = infos

        graph_obs = self._current_obs
        infos     = self._current_infos

        for _ in range(num_steps):

            graph_obs_device = self._move_graph_obs_to_device(graph_obs)
            policy_output    = self.policy(graph_obs_device)

            # ── Traffic action ────────────────────────────────────────────────
            traffic_action_dim = policy_output.logits.shape[-1]

            masks = np.stack([
                infos[agent]["action_mask"][:traffic_action_dim]
                for agent in graph_obs.agent_ids
            ])
            mask_tensor = (
                torch.from_numpy(masks).bool().to(policy_output.logits.device)
            )
            masked_logits    = policy_output.logits.masked_fill(~mask_tensor, -1e9)
            dist_traffic     = Categorical(logits=masked_logits)
            traffic_actions  = dist_traffic.sample()
            traffic_log_probs = dist_traffic.log_prob(traffic_actions)

            # ── Gifting action ────────────────────────────────────────────────
            dist_gifting      = Categorical(logits=policy_output.gifting_logits)
            gifting_actions   = dist_gifting.sample()
            gifting_log_probs = dist_gifting.log_prob(gifting_actions)

            # ── Build action dicts ────────────────────────────────────────────
            action_dict = {
                agent_id: [int(t), int(g)]
                for agent_id, t, g in zip(
                    graph_obs.agent_ids, traffic_actions, gifting_actions
                )
            }
            gifting_action_dict = {
                agent_id: int(g)
                for agent_id, g in zip(graph_obs.agent_ids, gifting_actions)
            }

            # ── Environment step ──────────────────────────────────────────────
            next_graph_obs, rewards, terminations, truncations, infos = (
                self.env.step(action_dict)
            )

            done = any(terminations.values()) or any(truncations.values())

            if done:
                graph_obs, infos = self.env.reset()
                if len(rollout) == 0:
                    continue
                break

            # ── Aggregate gifting stats ───────────────────────────────────────
            first_agent   = graph_obs.agent_ids[0]
            gifting_stats = {
                "mean_gift_fraction": infos[first_agent].get("mean_gift_fraction", 0.0),
                "gift_rate":          infos[first_agent].get("gift_rate",          0.0),
                "mean_gift_amount":   infos[first_agent].get("mean_gift_amount",   0.0),
            }

            # ── Per-agent gifting detail ──────────────────────────────────────
            per_agent_gifting = {}
            for agent_id in graph_obs.agent_ids:
                agent_info = infos.get(agent_id, {})
                per_agent_gifting[agent_id] = {
                    "gift_action":   gifting_action_dict[agent_id],
                    "gift_fraction": agent_info.get("gift_fraction",  0.0),
                    "gift_amount":   agent_info.get("gift_amount",    0.0),
                    "raw_reward":    agent_info.get("raw_traffic_reward",
                                                   rewards.get(agent_id, 0.0)),
                }

            # ── Store transition ──────────────────────────────────────────────
            rollout.append(GiftingTransition(
                observation=graph_obs_device,
                action_dict=action_dict,
                log_prob=traffic_log_probs.detach(),
                logits=policy_output.logits.detach(),
                value=policy_output.value.detach(),
                reward_dict=rewards,
                action_masks=mask_tensor.cpu(),
                done=done,
                gifting_action_dict=gifting_action_dict,
                gifting_log_prob=gifting_log_probs.detach(),
                gifting_stats=gifting_stats,
                per_agent_gifting=per_agent_gifting,
            ))

            graph_obs = next_graph_obs

        self._current_obs   = graph_obs
        self._current_infos = infos
        self.last_observation = self._move_graph_obs_to_device(graph_obs)

        return rollout