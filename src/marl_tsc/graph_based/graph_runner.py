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

CHANGED (persist environment state across rollouts)
------------------------------------------------------
Previously collect_rollout() called self.env.reset() at the START of
every call. Episodes in this env run ~600 steps; rollouts are only 64
steps. Resetting every call meant every single rollout restarted the
episode from step 0 -- the agent never experienced anything past
roughly the first 10% of an episode's traffic dynamics during
training, and every rollout's value/return statistics came from the
same early-episode regime instead of a representative sample across
the full episode.

Now the environment's observation is persisted on self between calls.
collect_rollout() only resets when the PREVIOUS rollout actually ended
in a terminal state (done=True) -- otherwise it continues stepping
from wherever the last rollout left off. This also enables the GAE
truncation-bootstrap fix in base_trainer.py, which needs access to the
observation immediately following the last collected transition.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.distributions import Categorical


@dataclass
class Transition:
    observation: object
    action_dict: dict
    log_prob: torch.Tensor
    logits: list[torch.Tensor]       # Now a list holding [traffic, sharing]
    value: torch.Tensor
    reward_dict: dict
    action_masks: list[torch.Tensor] # Now a list holding [traffic, sharing]
    done: bool


class GraphRunner:

    def __init__(
        self,
        env,
        policy,
    ):
        self.env = env
        self.policy = policy

        # Persisted across collect_rollout() calls. None until the first call,
        # at which point a fresh reset happens. After that, this carries over 
        # between rollouts unless the previous rollout ended in a terminal state.
        self._current_obs = None
        self._current_infos = None

        # Exposed so base_trainer.collect_batch() can fetch a bootstrap value 
        # for GAE from whatever observation comes right after the rollout.
        self.last_observation = None

    def collect_rollout(
        self,
        num_steps: int,
        seed: int | None = None,
    ) -> list[Transition]:

        rollout = []

        # Only reset if this is the very first call, or if the previous 
        # rollout actually ended in a terminal state. Otherwise, continue 
        # from wherever the last rollout left off.
        if self._current_obs is None:
            graph_obs, infos = self.env.reset(seed=seed)
            self._current_obs = graph_obs
            self._current_infos = infos

        graph_obs = self._current_obs
        infos = self._current_infos

        for _ in range(num_steps):
            policy_output = self.policy(graph_obs)

            # 1. Unpack the two branches of logits
            traffic_logits, sharing_logits = policy_output.logits

            # 2. Get the combined flat masks from the environment
            flat_masks = np.stack([
                infos[agent]["action_mask"]
                for agent in graph_obs.agent_ids
            ])

            flat_mask_tensor = (
                torch.from_numpy(flat_masks)
                .bool()
                .to(traffic_logits.device)
            )

            # 3. Split the flat mask back into traffic and sharing parts
            traffic_dim = traffic_logits.shape[-1]
            traffic_mask = flat_mask_tensor[:, :traffic_dim]
            sharing_mask = flat_mask_tensor[:, traffic_dim:]

            # 4. Apply the correct masks to prevent illegal actions
            masked_traffic_logits = traffic_logits.masked_fill(
                ~traffic_mask,
                -1e9,
            )
            masked_sharing_logits = sharing_logits.masked_fill(
                ~sharing_mask,
                -1e9,
            )

            # 5. Create distributions for both action types
            dist_traffic = Categorical(logits=masked_traffic_logits)
            dist_sharing = Categorical(logits=masked_sharing_logits)

            # Sample actions from both distributions
            traffic_actions = dist_traffic.sample()
            sharing_actions = dist_sharing.sample()

            # 6. Calculate the joint log probability 
            # (summing them combines the mathematical probability)
            log_probs = (
                dist_traffic.log_prob(traffic_actions)
                + dist_sharing.log_prob(sharing_actions)
            )

            # 7. Create the two-part action dictionary for the environment
            action_dict = {
                agent_id: [int(t_act), int(s_act)]
                for agent_id, t_act, s_act in zip(
                    graph_obs.agent_ids,
                    traffic_actions,
                    sharing_actions,
                )
            }

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

            # 8. Store both sets of logits and masks in the Transition
            rollout.append(
                Transition(
                    observation=graph_obs,
                    action_dict=action_dict,
                    log_prob=log_probs.detach(),
                    logits=[
                        masked_traffic_logits.detach(),
                        masked_sharing_logits.detach()
                    ],
                    value=policy_output.value.detach(),
                    reward_dict=rewards,
                    action_masks=[traffic_mask.cpu(), sharing_mask.cpu()],
                    done=done,
                )
            )

            graph_obs = next_graph_obs

            if done:
                # Episode genuinely ended -- reset now so the NEXT call 
                # starts fresh, but still finish out this rollout cleanly.
                graph_obs, infos = self.env.reset()
                break

        # Persist state for the next call, and expose the post-rollout 
        # observation for the GAE bootstrap fetch.
        self._current_obs = graph_obs
        self._current_infos = infos
        self.last_observation = graph_obs

        return rollout
