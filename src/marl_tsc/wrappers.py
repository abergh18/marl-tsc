"""
wrappers.py

Contains custom PettingZoo wrappers for the MARL traffic signal control environment.
These wrappers modify action spaces, enforce real-world traffic constraints, 
and implement peer-rewarding and zero-sum gifting mechanics.
"""

from __future__ import annotations

import math
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
        
        self.current_phase = {agent: 0 for agent in self.possible_agents}
        self.steps_since_switch = {agent: 0 for agent in self.possible_agents}

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        for agent in self.possible_agents:
            self.current_phase[agent] = 0
            self.steps_since_switch[agent] = self.min_green_steps
            
        return obs, infos

    def step(self, actions):
        overridden_actions = {}

        for agent, requested_action in actions.items():
            if requested_action != self.current_phase[agent]:
                if self.steps_since_switch[agent] >= self.min_green_steps:
                    overridden_actions[agent] = requested_action
                    self.current_phase[agent] = requested_action
                    self.steps_since_switch[agent] = 1
                else:
                    overridden_actions[agent] = self.current_phase[agent]
                    self.steps_since_switch[agent] += 1
            else:
                overridden_actions[agent] = requested_action
                self.steps_since_switch[agent] += 1

        return self.env.step(overridden_actions)


# ── Zero-sum gifting ──────────────────────────────────────────────────────────


class ZeroSumCalculator:
    """Zero-sum reward redistribution logic restricted to local neighbours."""

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
        """Apply zero-sum redistribution only among neighbouring agents."""
        num_agents = len(agent_ids)

        if num_agents < 2:
            return rewards

        gifts = {
            agent: gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        }

        shares = {agent: 0.0 for agent in agent_ids}
        for agent in agent_ids:
            agent_neighbours = neighbours.get(agent, [])
            active_neighbours = [n for n in agent_neighbours if n in agent_ids]
            
            if active_neighbours:
                share_per_neighbour = gifts[agent] / len(active_neighbours)
                for neighbour in active_neighbours:
                    shares[neighbour] += share_per_neighbour
            else:
                shares[agent] += gifts[agent]

        redistributed = {}
        for agent in agent_ids:
            redistributed[agent] = rewards[agent] - gifts[agent] + shares[agent]

        return redistributed

    def stats(
        self,
        rewards: dict[str, float],
        gifting_actions: dict[str, int],
        agent_ids: list[str],
    ) -> dict[str, float]:
        """Compute gifting statistics for logging."""
        fractions = [
            gifting_actions[agent] * self.portion_size
            for agent in agent_ids
        ]
        amounts = [
            gifting_actions[agent] * self.portion_size * abs(rewards[agent])
            for agent in agent_ids
        ]
        
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
    """

    def __init__(self, env, division: int | None = None):
        super().__init__(env)

        self.possible_agents = list(env.possible_agents)
        self.division = division if division is not None else 10
        self.calculator = ZeroSumCalculator(num_divisions=self.division)
        
        # We start with this empty, and populate it once the simulation boots up
        self.neighbours = None

        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                self.division + 1,
            ])
            for agent in self.possible_agents
        }
        
    def _discover_neighbours_via_traci(self) -> dict[str, list[str]]:
        """Queries the live TraCI simulation to find the physical centre of each intersection."""
        traci = self.env.unwrapped._traci
        valid_agents = self.possible_agents
        agent_coords = {}
        
        # 1. Ask TraCI for the exact coordinates of every traffic light
        for agent in valid_agents:
            try:
                lanes = traci.trafficlight.getControlledLanes(agent)
                if not lanes:
                    continue
                    
                x_sum, y_sum = 0.0, 0.0
                for lane in lanes:
                    # Get the shape points of the lane
                    shape = traci.lane.getShape(lane)
                    # The last point is where the lane hits the traffic light
                    x, y = shape[-1]
                    x_sum += x
                    y_sum += y
                    
                # Find the centre point of the junction
                avg_x = x_sum / len(lanes)
                avg_y = y_sum / len(lanes)
                agent_coords[agent] = (avg_x, avg_y)
            except Exception:
                pass
                
        # 2. Link each agent to its physically closest neighbours
        neighbours = {agent: set() for agent in valid_agents}
        k_neighbours = min(3, len(valid_agents) - 1)
        
        for agent in valid_agents:
            if agent not in agent_coords:
                continue
                
            x1, y1 = agent_coords[agent]
            distances = []
            
            for other_agent in valid_agents:
                if agent == other_agent or other_agent not in agent_coords:
                    continue
                
                x2, y2 = agent_coords[other_agent]
                # Calculate direct physical distance between the junctions
                dist = math.hypot(x2 - x1, y2 - y1)
                distances.append((dist, other_agent))
                
            # Sort by shortest distance
            distances.sort(key=lambda item: item[0])
            for dist, closest_agent in distances[:k_neighbours]:
                neighbours[agent].add(closest_agent)
                neighbours[closest_agent].add(agent)
                
        # Fallback for any agents that completely failed to register in TraCI
        for agent in valid_agents:
            if not neighbours[agent]:
                neighbours[agent] = [other for other in valid_agents if other != agent]
                
        return {agent: list(agent_neighbours) for agent, agent_neighbours in neighbours.items()}

    def action_space(self, agent):
        return self.action_spaces[agent]

    def reset(self, seed=None, options=None):
        obs, infos = self.env.reset(seed=seed, options=options)
        
        # We trigger the discovery here because the simulation is now guaranteed to be running
        if self.neighbours is None:
            self.neighbours = self._discover_neighbours_via_traci()
            print("\n=============================================")
            print("Discovered Network Neighbours (Live TraCI):")
            for a, n in self.neighbours.items():
                print(f"  {a} -> {n}")
            print("=============================================\n")

        infos = self._update_action_masks(infos)
        return obs, infos

    def step(self, actions):
        env_actions = {}
        gifting_actions = {}

        for agent, action in actions.items():
            env_actions[agent] = action[0]
            gifting_actions[agent] = action[1]

        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

        agent_ids = list(self.agents)

        if not agent_ids:
            return obs, {}, terms, truncs, infos

        # Make sure neighbours are initialised just in case step is somehow called first
        if self.neighbours is None:
            self.neighbours = {a: [] for a in agent_ids}

        redistributed = self.calculator.redistribute(
            rewards=rewards,
            gifting_actions=gifting_actions,
            agent_ids=agent_ids,
            neighbours=self.neighbours,
        )

        stats = self.calculator.stats(
            rewards=rewards,
            gifting_actions=gifting_actions,
            agent_ids=agent_ids,
        )

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
        for agent, info in infos.items():
            if "action_mask" in info:
                traffic_mask = info["action_mask"]
                gifting_mask = np.ones(self.division + 1, dtype=np.float32)
                info["action_mask"] = np.concatenate(
                    [traffic_mask, gifting_mask]
                )
        return infos
