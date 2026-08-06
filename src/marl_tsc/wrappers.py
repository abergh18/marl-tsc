"""
wrappers.py

Contains custom PettingZoo wrappers for the MARL traffic signal control environment.
These wrappers modify action spaces, enforce real-world traffic constraints, 
and implement peer-rewarding and zero-sum gifting mechanics.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

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

        self.neighbours = self._discover_neighbours(env.unwrapped.config_file, self.possible_agents)
        
        print("\n=============================================")
        print("Discovered Network Neighbours (By Distance):")
        for a, n in self.neighbours.items():
            print(f"  {a} -> {n}")
        print("=============================================\n")

        self.action_spaces = {
            agent: MultiDiscrete([
                env.action_space(agent).n,
                self.division + 1,
            ])
            for agent in self.possible_agents
        }
        
    def _discover_neighbours(self, config_file: Path, valid_agents: list[str]) -> dict[str, list[str]]:
        """Parses the SUMO network XML to find the physically closest traffic lights."""
        neighbours = {agent: set() for agent in valid_agents}
        
        try:
            config_path = Path(config_file)
            net_file_path = None
            
            # 1. Read the sumocfg to find the net.xml
            try:
                tree = ET.parse(config_path)
                for element in tree.iter("net-file"):
                    if "value" in element.attrib:
                        net_file_path = config_path.parent / element.attrib["value"]
                        break
            except Exception:
                pass
                
            if net_file_path is None or not net_file_path.exists():
                net_files = list(config_path.parent.glob("*.net.xml"))
                if not net_files:
                    raise FileNotFoundError("Could not find a .net.xml file.")
                net_file_path = net_files[0]
                
            # 2. Extract X/Y coordinates for every junction directly from XML
            net_tree = ET.parse(net_file_path)
            net_root = net_tree.getroot()
            
            agent_coords = {}
            
            for junction in net_root.findall("junction"):
                j_id = junction.get("id")
                tl_id = junction.get("tl")
                
                # Check if this junction belongs to one of our valid agents
                target_agent = None
                if tl_id in valid_agents:
                    target_agent = tl_id
                elif j_id in valid_agents:
                    target_agent = j_id
                    
                if target_agent:
                    try:
                        x = float(junction.get("x"))
                        y = float(junction.get("y"))
                        if target_agent not in agent_coords:
                            agent_coords[target_agent] = []
                        agent_coords[target_agent].append((x, y))
                    except (TypeError, ValueError):
                        pass
                        
            # 3. Average coordinates for clustered junctions
            final_coords = {}
            for agent, coords in agent_coords.items():
                avg_x = sum(c[0] for c in coords) / len(coords)
                avg_y = sum(c[1] for c in coords) / len(coords)
                final_coords[agent] = (avg_x, avg_y)
                
            # 4. Link each agent to its physically closest neighbours
            # Connect to up to 3 nearby agents
            k_neighbours = min(3, len(valid_agents) - 1)
            
            for agent in valid_agents:
                if agent not in final_coords:
                    continue
                    
                x1, y1 = final_coords[agent]
                distances = []
                
                for other_agent in valid_agents:
                    if agent == other_agent or other_agent not in final_coords:
                        continue
                    
                    x2, y2 = final_coords[other_agent]
                    # Calculate physical Euclidean distance
                    dist = math.hypot(x2 - x1, y2 - y1)
                    distances.append((dist, other_agent))
                    
                # Sort by shortest distance and add links
                distances.sort(key=lambda item: item[0])
                for dist, closest_agent in distances[:k_neighbours]:
                    neighbours[agent].add(closest_agent)
                    neighbours[closest_agent].add(agent)  # Ensure the link goes both ways
                    
            return {agent: list(agent_neighbours) for agent, agent_neighbours in neighbours.items()}
            
        except Exception as e:
            print(f"Warning: Failed to map coordinates ({e}). Defaulting to all-to-all gifting.")
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

        for agent, action in actions.items():
            env_actions[agent] = action[0]
            gifting_actions[agent] = action[1]

        obs, rewards, terms, truncs, infos = self.env.step(env_actions)

        agent_ids = list(self.agents)

        if not agent_ids:
            return obs, {}, terms, truncs, infos

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
