"""
gifting_graph_runner.py

Gifting-aware rollout collector for graph-based MARL.

Why a subclass
--------------
GiftingGraphRunner extends GraphRunner without modifying it. Non-gifting
experiments continue to use GraphRunner directly — same code, no hidden
conditionals, clean comparison integrity.

What changes
------------
- Samples a gifting action from GiftingMAPPOPolicy.gifting_head alongside
  the traffic action each step.
- Applies zero-sum reward redistribution after env.step() using the
  gifting actions, before storing the transition.
- Stores gifting actions and log probs in GiftingTransition for separate
  PPO loss computation in TrueMAPPOTrainer.
- Extracts and logs gifting stats from infos into the transition for
  downstream history logging.
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
        Step-level gifting statistics from infos:
        mean_gift_fraction, gift_rate, mean_gift_amount.
    """
    gifting_action_dict: dict = field(default_factory=dict)
    gifting_log_prob: torch.Tensor = None
    gifting_stats: dict = field(default_factory=dict)


# ── Runner ────────────────────────────────────────────────────────────────────

class GiftingGraphRunner(GraphRunner):
    """
    GraphRunner subclass for zero-sum gifting experiments.

    Assumes:
    - env is wrapped with ZeroSumRewardWrapper (MultiDiscrete action space)
    - policy is GiftingMAPPOPolicy (has .gifting_head)

    Parameters
    ----------
    env : ZeroSumRewardWrapper
        Environment with extended MultiDiscrete action space.
    policy : GiftingMAPPOPolicy
        Policy with traffic actor, centralised critic, and gifting head.
    """

    def collect_rollout(
        self,
        num_steps: int,
        seed: int | None = None,
    ):
        from torch_geometric.data import Data
        from marl_tsc.graph_based.graph_types import GraphObservation

        rollout = []

        if self._current_obs is None:
            graph_obs, infos = self.env.reset(seed=seed)
            self._current_obs = graph_obs
            self._current_infos = infos

        graph_obs = self._current_obs
        infos = self._current_infos

        for _ in range(num_steps):

            graph_obs_device = self._move_graph_obs_to_device(graph_obs)

            policy_output = self.policy(graph_obs_device)

            # ── Traffic action ────────────────────────────────────────────────
            # Action mask covers only traffic dimensions — gifting is always
            # legal so the gifting portion of the mask is all-ones.
            # We slice just the traffic part for masking logits.
            traffic_action_dim = policy_output.logits.shape[-1]

            masks = np.stack(
                [
                    infos[agent]["action_mask"][:traffic_action_dim]
                    for agent in graph_obs.agent_ids
                ]
            )

            mask_tensor = (
                torch.from_numpy(masks)
                .bool()
                .to(policy_output.logits.device)
            )

            masked_logits = policy_output.logits.masked_fill(
                ~mask_tensor,
                -1e9,
            )

            dist_traffic = Categorical(logits=masked_logits)
            traffic_actions = dist_traffic.sample()          # (num_agents,)
            traffic_log_probs = dist_traffic.log_prob(
                traffic_actions
            )                                                # (num_agents,)

            # ── Gifting action ────────────────────────────────────────────────
            dist_gifting = Categorical(
                logits=policy_output.gifting_logits          # (num_agents, div+1)
            )
            gifting_actions = dist_gifting.sample()          # (num_agents,)
            gifting_log_probs = dist_gifting.log_prob(
                gifting_actions
            )                                                # (num_agents,)

            # ── Build action dicts ────────────────────────────────────────────
            # ZeroSumRewardWrapper expects [traffic_action, gifting_action]
            action_dict = {
                agent_id: [int(t), int(g)]
                for agent_id, t, g in zip(
                    graph_obs.agent_ids,
                    traffic_actions,
                    gifting_actions,
                )
            }

            gifting_action_dict = {
                agent_id: int(g)
                for agent_id, g in zip(
                    graph_obs.agent_ids,
                    gifting_actions,
                )
            }

            # ── Environment step ──────────────────────────────────────────────
            (
                next_graph_obs,
                rewards,          # already redistributed by ZeroSumRewardWrapper
                terminations,
                truncations,
                infos,
            ) = self.env.step(action_dict)

            done = (
                any(terminations.values())
                or any(truncations.values())
            )

            # ── Extract gifting stats from infos ──────────────────────────────
            # ZeroSumRewardWrapper attaches these at each step.
            first_agent = graph_obs.agent_ids[0]
            gifting_stats = {
                "mean_gift_fraction": infos[first_agent].get("mean_gift_fraction", 0.0),
                "gift_rate": infos[first_agent].get("gift_rate", 0.0),
                "mean_gift_amount": infos[first_agent].get("mean_gift_amount", 0.0),
            }

            # ── Store transition ──────────────────────────────────────────────
            # traffic log_prob stored in the standard field so existing
            # RolloutBatch and GAE code works unchanged.
            rollout.append(
                GiftingTransition(
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
                )
            )

            graph_obs = next_graph_obs

            if done:
                graph_obs, infos = self.env.reset()
                break

        self._current_obs = graph_obs
        self._current_infos = infos
        self.last_observation = self._move_graph_obs_to_device(graph_obs)

        return rollout