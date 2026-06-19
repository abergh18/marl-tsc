"""
graph_runner.py

Collects graph-based experience from the environment.

This module is intentionally independent of any specific reinforcement
learning algorithm. Its responsibility is to:

    observe
        ↓
    encode
        ↓
    act
        ↓
    step environment
        ↓
    store transition

The resulting rollout can later be consumed by PPO, MAPPO-style
optimisation, imitation learning, offline RL, or other algorithms.

Keeping rollout collection separate from optimisation makes it easier to
experiment with different graph encoders and learning algorithms without
modifying environment interaction code.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np
from torch.distributions import Categorical


@dataclass
class Transition:

    observation: object

    action_dict: dict

    log_prob: torch.Tensor

    logits: torch.Tensor

    value: torch.Tensor

    reward_dict: dict

    action_masks: dict

    done: bool


class GraphRunner:

    def __init__(
        self,
        env,
        policy,
    ):
        self.env = env
        self.policy = policy

    def collect_rollout(
        self,
        num_steps: int,
        seed: int | None = None,
    ):

        rollout = []

        graph_obs, infos = self.env.reset(
            seed=seed
        )

        for _ in range(num_steps):
            policy_output = self.policy(
                graph_obs
            )

          
            masks = np.stack(
                [
                    infos[agent]["action_mask"]
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

            dist = Categorical(
                logits=masked_logits
            )

            actions = dist.sample()

            log_probs = dist.log_prob(
                actions
            )

            action_dict = {
                agent_id: int(action)
                for agent_id, action in zip(
                    graph_obs.agent_ids,
                    actions,
                )
            }

            (
                next_graph_obs,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(
                action_dict
            )

            done = (
                any(terminations.values())
                or
                any(truncations.values())
            )

            rollout.append(
                Transition(
                    observation=graph_obs,
                    action_dict=action_dict,
                    log_prob=log_probs.detach(),
                    logits=policy_output.logits.detach(),
                    value=policy_output.value.detach(),
                    reward_dict=rewards,
                    action_masks=mask_tensor.cpu(),
                    done=done,
                )
            )

            graph_obs = next_graph_obs

            if done:
                break

        return rollout