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
- Samples a gifting action from GiftingMAPPOPolicy.branches[1] alongside
  the traffic action each step.
- Applies reward sharing via the wrapper (PeerRewardingWrapper or
  ZeroSumRewardWrapper) which handles redistribution internally.
- Stores gifting actions and log probs in GiftingTransition for separate
  PPO loss computation in TrueMAPPOTrainer.
- Extracts and logs gifting stats from infos into the transition for
  downstream history logging.
- Done check happens before appending transition to avoid empty reward
  dicts from terminal steps.
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
    GraphRunner subclass for reward sharing experiments.

    Assumes:
    - env is wrapped with PeerRewardingWrapper or ZeroSumRewardWrapper
      (MultiDiscrete action space)
    - policy is GiftingMAPPOPolicy (has .branches[0] for traffic,
      .branches[1] for gifting)

    Parameters
    ----------
    env : PeerRewardingWrapper or ZeroSumRewardWrapper
        Environment with extended MultiDiscrete action space.
    policy : GiftingMAPPOPolicy
        Policy with shared encoder and branched actor heads.
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
            # Slice just the traffic portion of the mask
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
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(action_dict)

            done = (
                any(terminations.values())
                or any(truncations.values())
            )

            # ── Done check before appending ───────────────────────────────────
            # If the episode ended, reset and break before storing the
            # terminal transition. The terminal reward dict may be empty
            # or incomplete after SumoTrafficEnv clears self.agents.
            if done:
                graph_obs, infos = self.env.reset()
                if len(rollout) == 0:
                    # Episode ended before any transitions collected
                    # Reset and continue rather than returning empty rollout
                    continue
                break

            # ── Extract gifting stats from infos ──────────────────────────────
            first_agent = graph_obs.agent_ids[0]
            gifting_stats = {
                "mean_gift_fraction": infos[first_agent].get("mean_gift_fraction", 0.0),
                "gift_rate": infos[first_agent].get("gift_rate", 0.0),
                "mean_gift_amount": infos[first_agent].get("mean_gift_amount", 0.0),
            }

            # ── Store transition ──────────────────────────────────────────────
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

        self._current_obs = graph_obs
        self._current_infos = infos
        self.last_observation = self._move_graph_obs_to_device(graph_obs)

        return rollout