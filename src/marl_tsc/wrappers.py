"""
wrappers.py

Contains custom PettingZoo wrappers for the MARL traffic signal control environment.
These wrappers modify action spaces, enforce real-world traffic constraints, 
and implement peer-rewarding and zero-sum gifting mechanics.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import sumolib
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


# ── Zero-sum gifting ──────────────────────────────────────────────────────────


class ZeroSumCalculator:
    """
    Zero-sum reward redistribution logic restricted to local neighbours.

    Stateless class containing the zero-sum maths so PeerRewardingWrapper
    stays clean and the redistribution logic is testable independently
    of the PettingZoo interface.
    """

    def __init__(self, num_divisions: int):
        self.num_divisions = num_divisions
        self.portion_size = 1.0 / num_divisions

    def redistribute(
        self,
        rewards: dict[str, float],
        gifting_actions: dict[str, int],
        agent_ids: list[str],
        neighbours: dict[str, list[str]],
    ) -> dict[str, float]:
        """
        Apply zero-sum redistribution only among neighbouring agents.
        """
        num_agents = len(agent_ids)

        if num_agents < 2:
            return rewards

        # Compute absolute gift amounts 
        gifts = {
            agent: gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        }

        # Distribute each agent's gift equally among its active neighbours
        shares = {agent: 0.0 for agent in agent_ids}
        for agent in agent_ids:
            agent_neighbours = neighbours.get(agent, [])
            
            # Ensure we only share with neighbours currently active in the environment
            active_neighbours = [n for n in agent_neighbours if n in agent_ids]
            
            if active_neighbours:
                share_per_neighbour = gifts[agent] / len(active_neighbours)
                for neighbour in active_neighbours:
                    shares[neighbour] += share_per_neighbour
            else:
                # If an agent tries to share but has no active neighbours, refund the gift
                shares[agent] += gifts[agent]

        # Agent loses its given gift, but gains shares from its incoming neighbours
        redistributed = {}
        for agent in agent_ids:
            redistributed[agent] = rewards[agent] - gifts[agent] + shares[agent]

        # Debug print to verify neighbour gifting is working
        if sum(gifts.values()) > 0:
            print("\n--- Neighbour Gifting Step ---")
            for agent in agent_ids:
                if gifts[agent] > 0 or shares[agent] > 0:
                    net_effect = shares[agent] - gifts[agent]
                    print(
                        f"{agent} | "
                        f"Gave: {gifts[agent]:.2f} | "
                        f"Received: {shares[agent]:.2f} | "
                        f"Net: {net_effect:.2f}"
                    )
            print("------------------------------\n")

        return redistributed

    def stats(
        self,
        rewards: dict[str, float],
        gifting_actions: dict[str, int],
        agent_ids: list[str],
    ) -> dict[str, float]:
        """
        Compute gifting statistics for logging.
        """
        fractions = [
            gifting_actions[agent] * self.portion_size
            for agent in agent_ids
        ]
        amounts = [
            gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        ]
        
        # Prevent division by zero if agent_ids is empty
        if not fractions:
            return {
                "mean_gift_fraction": 0.0,
                "gift_rate": 0.0,
                "mean_gift_amount": 0.0,
            }
            
        return {
            "mean_gift_fraction": float(sum(fractions) / len(fractions)),
            "gift_rate": float(sum(1 for f in fractions if f > 0) / len(fractions)),
            "mean_gift_amount": float(sum(amounts) / len(amounts)),
        }


class PeerRewardingWrapper(BaseParallelWrapper):
    """
    PettingZoo wrapper implementing zero-sum peer reward sharing among neighbours.
    (Renamed from ZeroSumRewardWrapper to match the training script imports).

    Extends the action space with a discrete gifting action per agent.
    After each environment step, rewards are redistributed using the
    zero-sum mechanic: whatever an agent gives, it loses; gifts are
    split equally among its connected neighbours.
    """

    def __init__(self, env, division: int | None = None):
        super().__init__(env)

        self.possible_agents = list(env.possible_agents)
        num_agents = len(self.possible_agents)
        
        # Use provided division, otherwise default to 10 chunks
        self.division = division if division is not None else 10
        self.calculator = ZeroSumCalculator(num_divisions=self.division)

        # Automatically trace the SUMO network to find connected neighbours
        self.neighbours = self._discover_neighbours(env.unwrapped.config_file, self.possible_agents)
        
        # Print to console so you can verify the topology is correct
        print("Discovered Network Neighbours:", self.neighbours)

        # Extend action space: [Traffic Phase, Gifting Fraction]
        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                self.division + 1,
            ])
            for agent in self.possible_agents
        }
        
    def _discover_neighbours(self, config_file: Path, valid_agents: list[str]) -> dict[str, list[str]]:
        """Parses the SUMO network to find adjacent traffic lights."""
        neighbours = {agent: set() for agent in valid_agents}
        
        try:
            config_text = config_file.read_text(encoding="utf-8")
            match = re.search(r'<net-file\s+value="([^"]+)"', config_text)
            if not match:
                print("Warning: Could not find net-file in sumocfg. Defaulting to all-to-all gifting.")
                return {a: [other for other in valid_agents if other != a] for a in valid_agents}
                
            net_file = config_file.parent / match.group(1)
            net = sumolib.net.readNet(str(net_file))
            
            # Iterate through all edges (streets) in the network
            for edge in net.getEdges():
                from_node = edge.getFromNode()
                to_node = edge.getToNode()
                
                from_tls = from_node.getTLS()
                to_tls = to_node.getTLS()
                
                # If both ends of the street have a traffic light, they are neighbours
                if from_tls and to_tls:
                    from_id = from_tls.getID()
                    to_id = to_tls.getID()
                    
                    if from_id in valid_agents and to_id in valid_agents and from_id != to_id:
                        neighbours[from_id].add(to_id)
                        neighbours[to_id].add(from_id)
                        
            return {agent: list(agent_neighbours) for agent, agent_neighbours in neighbours.items()}
            
        except Exception as e:
            print(f"Warning: Failed to parse neighbours ({e}). Defaulting to all-to-all gifting.")
            return {a: [other for other in valid_agents if other != a] for a in valid_agents}

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

        # 3. Apply neighbour-restricted zero-sum redistribution
        redistributed = self.calculator.redistribute(
            rewards=rewards,
            gifting_actions=gifting_actions,
            agent_ids=agent_ids,
            neighbours=self.neighbours,
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
