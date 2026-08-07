from __future__ import annotations

import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo.utils.wrappers import BaseParallelWrapper

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
                division + 1,
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
            share_percentage = sharing_actions[agent] * self.portion_size
            
            personal_cost = share_percentage * 0.01
            community_contribution = personal_cost * 2.0
            
            sharing_pool += community_contribution
            final_rewards[agent] = rewards[agent] - personal_cost

        # 4. Distribute the pooled community rewards equally
        payout_per_agent = sharing_pool / max(1, num_agents)

        for agent in self.agents:
            final_rewards[agent] += payout_per_agent
            
            if "raw_traffic_reward" not in infos[agent]:
                infos[agent]["raw_traffic_reward"] = rewards[agent]

        infos = self._update_action_masks(infos)
        return obs, final_rewards, terms, truncs, infos

    def _update_action_masks(self, infos):
        """Append a valid mask for the sharing action to the traffic mask."""
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                sharing_mask = np.ones(self.division + 1, dtype=np.float32)
                
                info["action_mask"] = np.concatenate(
                    [traffic_mask, sharing_mask]
                )
        return infos
