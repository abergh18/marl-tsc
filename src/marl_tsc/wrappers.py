import numpy as np
from gymnasium.spaces import MultiDiscrete
from pettingzoo.utils.wrappers import BaseParallelWrapper


class PeerRewardingWrapper(BaseParallelWrapper):
    """
    A wrapper that adds simultaneous peer rewarding to a PettingZoo environment.
    Uses a 'Public Goods' mechanic to prevent agents from exploiting negative rewards.
    """

    def __init__(self, env, division=10):
        super().__init__(env)
        self.division = division
        self.portion_size = 1.0 / division

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

        for agent, action in actions.items():
            env_actions[agent] = action[0]
            sharing_actions[agent] = action[1]

        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

final_rewards = {agent: 0.0 for agent in self.agents}
        sharing_pool = 0.0
        num_agents = len(self.agents)

        # 1. Agents generate cooperative value (Scaled down!)
        for agent in self.agents:
            # share_percentage is between 0.0 and 1.0
            share_percentage = sharing_actions[agent] * self.portion_size
            
            # SCALED DOWN: Max cost is now only 0.01 points per step
            personal_cost = share_percentage * 0.01
            
            # The community multiplier is 2.0, so the max contribution is 0.02
            community_contribution = personal_cost * 2.0
            
            sharing_pool += community_contribution
            
            # The agent MUST keep its original traffic penalty, minus the cost of sharing
            final_rewards[agent] = rewards[agent] - personal_cost

        # 2. Distribute the pooled community rewards equally
        # Removed the + 0.05 to close the free money loophole!
        payout_per_agent = sharing_pool / max(1, num_agents)

        for agent in self.agents:
            final_rewards[agent] += payout_per_agent

        infos = self._update_action_masks(infos)
        return obs, final_rewards, terms, truncs, infos

    def _update_action_masks(self, infos):
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                sharing_mask = np.ones(self.division + 1, dtype=np.float32)
                
                info["action_mask"] = np.concatenate(
                    [traffic_mask, sharing_mask]
                )
        return infos
