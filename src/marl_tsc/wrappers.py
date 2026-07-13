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

# ── Zero-sum gifting ──────────────────────────────────────────────────────────


class ZeroSumCalculator:
    """
    Pure zero-sum reward redistribution logic.

    Stateless class containing the zero-sum maths so ZeroSumRewardWrapper
    stays clean and the redistribution logic is testable independently
    of the PettingZoo interface.

    Zero-sum property
    -----------------
    Whatever agent i gives, agent i loses. The gift is split equally
    among all other agents. No reward is created or destroyed.

        gift_i    = action_i * portion_size * abs(reward_i)
        reward_i' = reward_i - gift_i + sum(share_j for j != i)

    Note on negative rewards
    ------------------------
    SUMO rewards are negative queue penalties. Gift amounts are computed
    on abs(reward) so gifting transfers genuine positive value rather
    than offloading penalty onto peers.

    Attribution
    -----------
    Zero-sum mechanic adapted from:

        Lupu, A. & Precup, D. (2020). Gifting in Multi-Agent Reinforcement
        Learning. Proceedings of AAMAS 2020.
    """

    def __init__(self, num_divisions: int):
        self.num_divisions = num_divisions
        self.portion_size = 1.0 / num_divisions

    def redistribute(
        self,
        rewards: dict[str, float],
        gifting_actions: dict[str, int],
        agent_ids: list[str],
    ) -> dict[str, float]:
        """
        Apply zero-sum redistribution to a single timestep's rewards.

        Parameters
        ----------
        rewards : dict[str, float]
            Raw per-agent rewards from env.step().
        gifting_actions : dict[str, int]
            Discrete gifting action per agent (0..num_divisions).
            Action k means gift k/num_divisions of abs(reward) to peers.
        agent_ids : list[str]
            Ordered list of agent IDs.

        Returns
        -------
        dict[str, float]
            Redistributed rewards. Sum equals sum of input rewards.
        """
        num_agents = len(agent_ids)

        if num_agents < 2:
            return rewards

        # Compute gift amounts from abs(reward)
        gifts = {
            agent: gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        }

        # Each gift split equally among num_agents - 1 peers
        shares = {
            agent: gifts[agent] / (num_agents - 1)
            for agent in agent_ids
        }

        # Agent i loses its gift, gains one share from every other agent
        redistributed = {}
        for agent in agent_ids:
            received = sum(
                shares[other]
                for other in agent_ids
                if other != agent
            )
            redistributed[agent] = rewards[agent] - gifts[agent] + received

        return redistributed

    def stats(
        self,
        rewards: dict[str, float],
        gifting_actions: dict[str, int],
        agent_ids: list[str],
    ) -> dict[str, float]:
        """
        Compute gifting statistics for logging.

        Returns
        -------
        dict with keys:
            mean_gift_fraction  — average fraction of reward gifted
            gift_rate           — fraction of agents who gifted non-zero
            mean_gift_amount    — average absolute reward transferred
        """
        fractions = [
            gifting_actions[agent] * self.portion_size
            for agent in agent_ids
        ]
        amounts = [
            gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        ]
        return {
            "mean_gift_fraction": float(sum(fractions) / len(fractions)),
            "gift_rate": float(sum(1 for f in fractions if f > 0) / len(fractions)),
            "mean_gift_amount": float(sum(amounts) / len(amounts)),
        }


class ZeroSumRewardWrapper(BaseParallelWrapper):
    """
    PettingZoo wrapper implementing zero-sum peer reward sharing.

    Extends the action space with a discrete gifting action per agent.
    After each environment step, rewards are redistributed using the
    zero-sum mechanic: whatever an agent gives, it loses; gifts are
    split equally among all other agents.

    Gifting stats (mean_gift_fraction, gift_rate, mean_gift_amount)
    and raw_traffic_reward are stored in infos at each step for
    downstream logging and evaluation comparison.

    Wrapper interface inspired by the PeerRewardingWrapper implementation
    by a project collaborator.

    Parameters
    ----------
    env : ParallelEnv
        Underlying PettingZoo environment.
    division : int, optional
        Number of discrete gifting portions. Defaults to num_agents - 1
        for clean equal-split granularity, but can be set independently.
    """

    def __init__(self, env, division: int | None = None):
        super().__init__(env)

        num_agents = len(self.possible_agents)
        self.division = division if division is not None else max(1, num_agents - 1)
        self.calculator = ZeroSumCalculator(num_divisions=self.division)

        # Extend action space: [Traffic Phase, Gifting Fraction]
        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                self.division + 1,
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
        gifting_actions = {}

        # 1. Unpack traffic and gifting actions
        for agent, action in actions.items():
            env_actions[agent] = action[0]
            gifting_actions[agent] = action[1]

        # 2. Step underlying environment with traffic actions only
        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

        agent_ids = list(self.agents)

        if not agent_ids:
          return obs, {}, terms, truncs, infos

        # 3. Apply zero-sum redistribution
        redistributed = self.calculator.redistribute(
            rewards=rewards,
            gifting_actions=gifting_actions,
            agent_ids=agent_ids,
        )

        # 4. Compute gifting stats for logging
        stats = self.calculator.stats(
            rewards=rewards,
            gifting_actions=gifting_actions,
            agent_ids=agent_ids,
        )

        # 5. Attach raw reward and gifting stats to infos for observability
        for agent in agent_ids:
            infos[agent]["raw_traffic_reward"] = rewards[agent]
            infos[agent]["gift_fraction"] = (
                gifting_actions[agent] / self.division
            )
            infos[agent]["gift_amount"] = (
                gifting_actions[agent] / self.division * abs(rewards[agent])
            )
            infos[agent]["mean_gift_fraction"] = stats["mean_gift_fraction"]
            infos[agent]["gift_rate"] = stats["gift_rate"]
            infos[agent]["mean_gift_amount"] = stats["mean_gift_amount"]

        infos = self._update_action_masks(infos)
        return obs, redistributed, terms, truncs, infos

    def _update_action_masks(self, infos):
        """Append an all-ones mask for the gifting action to the traffic mask."""
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                gifting_mask = np.ones(self.division + 1, dtype=np.float32)
                info["action_mask"] = np.concatenate(
                    [traffic_mask, gifting_mask]
                )
        return infos