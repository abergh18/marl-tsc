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
    Peer rewarding using alternating timesteps, matching the PhD implementation.
    Even steps: traffic actions only, rewards stored but not returned.
    Odd steps: sharing actions only, stored rewards redistributed and returned.
    """
    def __init__(self, env, division=10):
        super().__init__(env)
        self.division = division
        self.portion_size = 1.0 / division
        self.t = 0
        self.last_rewards = {agent: 0.0 for agent in env.possible_agents}

        # On traffic steps: Discrete action space (unchanged)
        # On sharing steps: Discrete(division + 1) for sharing percentage
        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                division + 1,
            ])
            for agent in self.possible_agents
        }

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        self.t = 0
        self.last_rewards = {agent: 0.0 for agent in self.possible_agents}
        obs, infos = self.env.reset(seed=seed, options=options)
        infos = self._update_action_masks(infos)
        return obs, infos

    def step(self, actions):
        players = list(self.agents)
        num_agents = max(len(players), 1)

        if self.t % 2 == 0:
            env_actions = {agent: action[0] for agent, action in actions.items()}
            obs, rewards, terms, truncs, infos = self.env.step(env_actions)

            # Store rewards for the sharing step; return zero rewards now
            self.last_rewards = {agent: float(rewards.get(agent, 0.0)) for agent in players}
            zero_rewards = {agent: 0.0 for agent in players}

            for agent in players:
                infos[agent]["raw_traffic_reward"] = self.last_rewards[agent]
                
            self._last_obs = obs
            self.t += 1
            infos = self._update_action_masks(infos)
            return obs, zero_rewards, terms, truncs, infos

        else:
            sharing_pool = 0.0
            personal_rewards = {}

            for agent in players:
                share_fraction = actions[agent][1] * self.portion_size
                give = share_fraction * self.last_rewards[agent]
                # Each agent gives to the pool and keeps the rest
                personal_rewards[agent] = self.last_rewards[agent] - give
                sharing_pool += give

            # Distribute pool equally among all agents
            payout_per_agent = sharing_pool / num_agents
            final_rewards = {
                agent: personal_rewards[agent] + payout_per_agent
                for agent in players
            }

            # Return the same obs/terms/truncs from the last traffic step
            obs = self._last_obs
            terms = {agent: False for agent in players}
            truncs = {agent: False for agent in players}
            infos = {agent: {"raw_traffic_reward": self.last_rewards[agent],
                             "action_mask": np.ones(sum(self.action_spaces[agent].nvec), dtype=np.float32)}
                     for agent in players}

            self.t += 1
            infos = self._update_action_masks(infos)
            return obs, final_rewards, terms, truncs, infos

    def _update_action_masks(self, infos):
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                sharing_mask = np.ones(self.division + 1, dtype=np.float32)
                info["action_mask"] = np.concatenate([traffic_mask, sharing_mask])
        return infos
