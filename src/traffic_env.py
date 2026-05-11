"""PettingZoo ParallelEnv wrapper around a SUMO simulation.

The goal of this file is to keep SUMO-specific environment logic outside the
notebook. The notebook can then stay readable and focus on the experiment flow.
"""

from __future__ import annotations

from pathlib import Path
import os
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class SumoTrafficEnv(ParallelEnv):
    """A minimal multi-agent traffic signal control environment using SUMO.

    Each SUMO traffic light becomes one PettingZoo agent. For a first version,
    each agent chooses one of a fixed number of green phases. The environment
    maps that selected green phase to the real SUMO phase index.
    """

    metadata = {"name": "sumo_traffic_v0", "render_modes": [None, "human"]}

    def __init__(
        self,
        config_file: str | Path,
        max_steps: int = 1000,
        seconds_per_action: int = 5,
        max_lanes_per_tls: int = 16,
        max_queue_value: float = 100.0,
        seed: int = 42,
        render_mode: str | None = None,
    ) -> None:
        self.config_file = Path(config_file)
        self.max_steps = max_steps
        self.seconds_per_action = seconds_per_action
        self.max_lanes_per_tls = max_lanes_per_tls
        self.max_queue_value = max_queue_value
        self.seed = seed
        self.render_mode = render_mode

        self.step_count = 0
        self.possible_agents: list[str] = []
        self.agents: list[str] = []
        self._tls_to_lanes: dict[str, list[str]] = {}
        self._tls_to_green_phases: dict[str, list[int]] = {}
        self._sumo_running = False

        self._traci = None

    def _import_traci(self):
        if self._traci is not None:
            return self._traci

        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Install SUMO and set SUMO_HOME before running the environment."
            )

        tools_path = Path(sumo_home) / "tools"
        if str(tools_path) not in os.sys.path:
            os.sys.path.append(str(tools_path))

        import traci  # type: ignore

        self._traci = traci
        return self._traci

    def _sumo_binary(self) -> str:
        return "sumo-gui" if self.render_mode == "human" else "sumo"

    def _start_sumo(self, seed: int | None = None) -> None:
        traci = self._import_traci()

        if self._sumo_running:
            traci.close()
            self._sumo_running = False

        if not self.config_file.exists():
            raise FileNotFoundError(f"SUMO config file not found: {self.config_file}")

        traci.start(
            [
                self._sumo_binary(),
                "-c",
                str(self.config_file),
                "--seed",
                str(seed if seed is not None else self.seed),
                "--no-warnings",
                "true",
            ]
        )
        self._sumo_running = True

    def _discover_agents_and_lanes(self) -> None:
        traci = self._import_traci()

        tls_ids = list(traci.trafficlight.getIDList())
        self.possible_agents = tls_ids
        self.agents = tls_ids.copy()

        self._tls_to_lanes = {}
        self._tls_to_green_phases = {}

        for tls_id in tls_ids:
            controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
            self._tls_to_lanes[tls_id] = controlled_lanes[: self.max_lanes_per_tls]

            program = traci.trafficlight.getCompleteRedYellowGreenDefinition(tls_id)[0]
            green_phases: list[int] = []
            for phase_index, phase in enumerate(program.phases):
                state = phase.state.lower()
                has_green = "g" in state
                has_yellow = "y" in state
                if has_green and not has_yellow:
                    green_phases.append(phase_index)

            # Fallback: if SUMO creates an unusual program, allow phase 0.
            if not green_phases:
                green_phases = [0]

            self._tls_to_green_phases[tls_id] = green_phases

    def observation_space(self, agent: str) -> Box:
        # Fixed-size observations are important for RL libraries.
        # Values are normalized queue counts for controlled incoming lanes.
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.max_lanes_per_tls,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> Discrete:
        green_phases = self._tls_to_green_phases.get(agent)
        if not green_phases:
            # Before reset(), we do not know the network yet. Return a safe default.
            return Discrete(1)
        return Discrete(len(green_phases))

    def _get_obs_for_agent(self, agent: str) -> np.ndarray:
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])

        values = []
        for lane_id in lanes[: self.max_lanes_per_tls]:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            values.append(min(queue / self.max_queue_value, 1.0))

        while len(values) < self.max_lanes_per_tls:
            values.append(0.0)

        return np.array(values, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {agent: self._get_obs_for_agent(agent) for agent in self.agents}

    def _reward_for_agent(self, agent: str) -> float:
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])
        total_queue = sum(traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes)
        return -float(total_queue)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._start_sumo(seed=seed)
        self.step_count = 0
        self._discover_agents_and_lanes()

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        traci = self._import_traci()

        for agent, action in actions.items():
            if agent not in self.agents:
                continue

            green_phases = self._tls_to_green_phases[agent]
            safe_action = int(action) % len(green_phases)
            phase_index = green_phases[safe_action]
            traci.trafficlight.setPhase(agent, phase_index)

        for _ in range(self.seconds_per_action):
            traci.simulationStep()

        self.step_count += 1

        observations = self._get_obs()
        rewards = {agent: self._reward_for_agent(agent) for agent in self.agents}

        reached_step_limit = self.step_count >= self.max_steps
        no_more_vehicles_expected = traci.simulation.getMinExpectedNumber() <= 0
        truncated = reached_step_limit or no_more_vehicles_expected

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self._sumo_running:
            traci = self._import_traci()
            traci.close()
            self._sumo_running = False
