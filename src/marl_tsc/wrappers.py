"""
wrappers.py

Contains custom PettingZoo wrappers for the MARL traffic signal control environment.
These wrappers modify action spaces, enforce real-world traffic constraints, 
and implement peer-rewarding mechanics.
"""

from __future__ import annotations

import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo.utils.wrappers import BaseParallelWrapper


class MinimumGreenTimeWrapper(BaseParallelWrapper):
    """
    Forces agents to hold a traffic phase for a minimum number of steps 
    before they are allowed to switch again. This prevents agents from 
    spamming the traffic lights and creates realistic traffic flow.
    """

    def __init__(self, env, min_green_steps=10):
        super().__init__(env)
        self.min_green_steps = min_green_steps
        
        # Track the state of each intersection
        self.current_phase = {agent: 0 for agent in self.possible_agents}
        self.steps_since_switch = {agent: 0 for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        # Reset trackers for a new episode
        for agent in self.possible_agents:
            self.current_phase[agent] = 0
            self.steps_since_switch[agent] = self.min_green_steps
            
        return obs, infos

    def step(self, actions):
        overridden_actions = {}

        for agent, requested_action in actions.items():
            # If the agent wants to switch to a new phase
            if requested_action != self.current_phase[agent]:
                # Check if they have waited long enough
                if self.steps_since_switch[agent] >= self.min_green_steps:
                    # Approved! Update the phase and reset the timer
                    overridden_actions[agent] = requested_action
                    self.current_phase[agent] = requested_action
                    self.steps_since_switch[agent] = 1
                else:
                    # Denied! Force them to keep the current phase
                    overridden_actions[agent] = self.current_phase[agent]
                    self.steps_since_switch[agent] += 1
            else:
                # Agent wants to keep the same phase anyway
                overridden_actions[agent] = requested_action
                self.steps_since_switch[agent] += 1

        # Pass the (potentially overridden) actions to the real environment
        return self.env.step(overridden_actions)


class PeerRewardingWrapper(BaseParallelWrapper):
    """
    A wrapper that adds simultaneous peer rewarding to a PettingZoo environment.
    Uses a 'Public Goods' mechanic to prevent agents from exploiting negative 
    rewards, forcing them to balance traffic management with community sharing.
    """

    def __init__(self, env, division=10):
        super().__init__(env)
        self.division = division
        self.portion_size = 1.0 / division

        # Expand the action space to a MultiDiscrete space:
        # [Traffic Phase, Sharing Percentage]
        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                division + 1
            ])
            for agent in self.possible_agents
        }

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        infos = self._update_action_masks(infos)
        return obs, infos

    def step(self, actions):
        env_actions = {}
        sharing_actions = {}

        # 1. Unpack the two separate actions
        for agent, action in actions.items():
            env_actions[agent] = action[0]
            sharing_actions[agent] = action[1]

        # 2. Step the underlying environment using ONLY the traffic actions
        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

        final_rewards = {agent: 0.0 for agent in self.agents}
        sharing_pool = 0.0
        num_agents = len(self.agents)

        # 3. Calculate Public Goods Game contributions
        for agent in self.agents:
            # share_percentage is between 0.0 and 1.0
            share_percentage = sharing_actions[agent] * self.portion_size
            
            # Scaled down to prevent the "infinite money glitch"
            personal_cost = share_percentage * 0.01
            community_contribution = personal_cost * 2.0
            
            sharing_pool += community_contribution
            
            # The agent MUST keep its original traffic penalty, minus the cost of sharing
            final_rewards[agent] = rewards[agent] - personal_cost

        # 4. Distribute the pooled community rewards equally
        payout_per_agent = sharing_pool / max(1, num_agents)

        for agent in self.agents:
            final_rewards[agent] += payout_per_agent
            
            # Sneak the original traffic penalty into infos for apples-to-apples evaluation
            if "raw_traffic_reward" not in infos[agent]:
                infos[agent]["raw_traffic_reward"] = rewards[agent]

        infos = self._update_action_masks(infos)
        return obs, final_rewards, terms, truncs, infos

    def _update_action_masks(self, infos):
        """Append a valid mask for the sharing action to the traffic mask."""
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                # All sharing actions (0% to 100%) are always legal
                sharing_mask = np.ones(self.division + 1, dtype=np.float32)
                
                info["action_mask"] = np.concatenate(
                    [traffic_mask, sharing_mask]
                )
        return infos
