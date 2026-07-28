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

import torch
import numpy as np
from torch.distributions import Categorical
from torch_geometric.data import Data


@dataclass
class Transition:
    observation: object
    action_dict: dict
    log_prob: torch.Tensor
    logits: object        # Can be torch.Tensor or list[torch.Tensor]
    value: torch.Tensor
    reward_dict: dict
    action_masks: object  # Can be torch.Tensor or list[torch.Tensor]
    done: bool


class GraphRunner:

    def __init__(
        self,
        env,
        policy,
    ):
        self.env = env
        self.policy = policy

        # CHANGED: persisted across collect_rollout() calls. None
        # until the first call, at which point a fresh reset happens.
        # After that, this carries over between rollouts unless the
        # previous rollout ended in a genuine terminal state.
        self._current_obs = None
        self._current_infos = None

        # CHANGED: exposed so base_trainer.collect_batch() can fetch
        # a bootstrap value for GAE from whatever observation comes
        # right after the most recently collected rollout.
        self.last_observation = None

    def _move_graph_obs_to_device(self, graph_obs):
        device = next(self.policy.parameters()).device

        graph = Data(
            x=graph_obs.graph.x.to(device),
            edge_index=graph_obs.graph.edge_index.to(device),
        )

        # Carry through het graph attributes if present
        if hasattr(graph_obs.graph, 'connection_x'):
            graph.connection_x = graph_obs.graph.connection_x.to(device)
        if hasattr(graph_obs.graph, 'agent_mask'):
            graph.agent_mask = graph_obs.graph.agent_mask.to(device)

        from marl_tsc.graph_based.graph_types import GraphObservation

        return GraphObservation(
            graph=graph,
            agent_ids=graph_obs.agent_ids,
            global_state=graph_obs.global_state.to(device) if graph_obs.global_state is not None else None,
            metadata=graph_obs.metadata,
        )

    def collect_rollout(
        self,
        num_steps: int,
        seed: int | None = None,
    ) -> list[Transition]:

        rollout = []

        # CHANGED: only reset if this is the very first call, or if
        # the previous rollout actually ended in a terminal state.
        # Otherwise continue from wherever the last rollout left off.
        if self._current_obs is None:
            graph_obs, infos = self.env.reset(seed=seed)
            self._current_obs = graph_obs
            self._current_infos = infos

        graph_obs = self._current_obs
        infos = self._current_infos

        for _ in range(num_steps):
            
            # Move graph observation to policy device
            graph_obs_device = self._move_graph_obs_to_device(graph_obs)

            policy_output = self.policy(graph_obs_device)

            # Get the combined flat masks from the environment
            masks = np.stack([
                infos[agent]["action_mask"]
                for agent in graph_obs.agent_ids
            ])

            # We use policy_output.value.device as a safe device reference
            mask_tensor = (
                torch.from_numpy(masks)
                .bool()
                .to(policy_output.value.device)
            )

            # Handle both standard single-action and multi-discrete (Gifting) actions
            if isinstance(policy_output.logits, (list, tuple)):
                traffic_logits, sharing_logits = policy_output.logits
                
                # Split the flat mask back into traffic and sharing parts
                traffic_dim = traffic_logits.shape[-1]
                traffic_mask = mask_tensor[:, :traffic_dim]
                sharing_mask = mask_tensor[:, traffic_dim:]

                # Apply the correct masks
                masked_traffic_logits = traffic_logits.masked_fill(~traffic_mask, -1e9)
                masked_sharing_logits = sharing_logits.masked_fill(~sharing_mask, -1e9)

                # Create distributions
                dist_traffic = Categorical(logits=masked_traffic_logits)
                dist_sharing = Categorical(logits=masked_sharing_logits)

                # Sample actions
                traffic_actions = dist_traffic.sample()
                sharing_actions = dist_sharing.sample()

                # Calculate joint log probability 
                log_probs = dist_traffic.log_prob(traffic_actions) + dist_sharing.log_prob(sharing_actions)

                # Create the two-part action dictionary for the environment
                action_dict = {
                    agent_id: [int(t_act), int(s_act)]
                    for agent_id, t_act, s_act in zip(graph_obs.agent_ids, traffic_actions, sharing_actions)
                }
                
                transition_logits = [masked_traffic_logits.detach(), masked_sharing_logits.detach()]
                transition_masks = [traffic_mask.cpu(), sharing_mask.cpu()]

            else:
                masked_logits = policy_output.logits.masked_fill(~mask_tensor, -1e9)
                dist = Categorical(logits=masked_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)

                action_dict = {
                    agent_id: int(action)
                    for agent_id, action in zip(graph_obs.agent_ids, actions)
                }
                
                transition_logits = masked_logits.detach()
                transition_masks = mask_tensor.cpu()

            (
                next_graph_obs,
                rewards,
                terminations,
                truncations,
                infos,
            ) = self.env.step(action_dict)

            done = any(terminations.values()) or any(truncations.values())

            rollout.append(
                Transition(
                    observation=graph_obs_device,
                    action_dict=action_dict,
                    log_prob=log_probs.detach(),
                    logits=transition_logits,
                    value=policy_output.value.detach(),
                    reward_dict=rewards,
                    action_masks=transition_masks,
                    done=done,
                )
            )

            graph_obs = next_graph_obs

            if done:
                # CHANGED: episode genuinely ended -- reset now so
                # the NEXT collect_rollout() call starts fresh, but
                # still finish out this rollout's transitions list
                # cleanly via break, same as before.
                graph_obs, infos = self.env.reset()
                break

        # CHANGED: persist state for the next call, and expose the
        # post-rollout observation for the GAE bootstrap fetch.
        self._current_obs = graph_obs
        self._current_infos = infos
        self.last_observation = self._move_graph_obs_to_device(graph_obs)

        return rollout