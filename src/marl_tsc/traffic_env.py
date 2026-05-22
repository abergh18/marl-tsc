"""PettingZoo ParallelEnv wrapper around a SUMO simulation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import os
import sys
from typing import Any

import numpy as np
from gymnasium.spaces import Box, Discrete
from pettingzoo import ParallelEnv


class SumoTrafficEnv(ParallelEnv):
    """A minimal multi-agent traffic signal control environment using SUMO."""

    metadata = {"name": "sumo_traffic_v0", "render_modes": [None, "human"]}

    def __init__(
        self,
        config_file: str | Path,
        max_steps: int = 1000,
        seconds_per_action: int = 5,
        max_lanes_per_tls: int | None = None,
        green_phase_count: int = 4,
        max_queue_value: float = 100.0,
        min_green_seconds: int = 10,
        switch_penalty: float = 0.1,
        seed: int = 42,
        render_mode: str | None = None,
        possible_agents: Sequence[str] | None = None,
    ) -> None:
        if green_phase_count < 1:
            raise ValueError("green_phase_count must be at least 1.")

        self.config_file = Path(config_file)
        self.max_steps = max_steps
        self.seconds_per_action = seconds_per_action
        self.max_lanes_per_tls = max_lanes_per_tls
        self.green_phase_count = green_phase_count
        self.max_queue_value = max_queue_value
        self.min_green_seconds = min_green_seconds
        self.switch_penalty = switch_penalty
        self.seed = seed
        self.render_mode = render_mode

        self.step_count = 0
        self.requested_agents = list(dict.fromkeys(possible_agents)) if possible_agents else None
        self.possible_agents: list[str] = list(self.requested_agents or [])
        self.agents: list[str] = self.possible_agents.copy()
        self._tls_to_lanes: dict[str, list[str]] = {}
        self._tls_to_green_phases: dict[str, list[int]] = {}
        self._current_actions: dict[str, int] = {}
        self._elapsed_green_seconds: dict[str, float] = {}
        self._switched_last_step: dict[str, bool] = {}
        self._sumo_running = False

        self._traci = None

        if self.max_lanes_per_tls is None:
            self._probe_network_for_lane_count()

    def _import_traci(self):
        if self._traci is not None:
            return self._traci

        sumo_home = os.environ.get("SUMO_HOME")
        if not sumo_home:
            raise EnvironmentError(
                "SUMO_HOME is not set. Install SUMO and set SUMO_HOME before running the environment."
            )

        tools_path = Path(sumo_home) / "tools"
        if str(tools_path) not in sys.path:
            sys.path.append(str(tools_path))

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

    def _probe_network_for_lane_count(self) -> None:
        """Briefly start SUMO to size obs to the network's real lane count."""
        traci = self._import_traci()
        self._start_sumo(seed=self.seed)
        try:
            discovered = list(dict.fromkeys(traci.trafficlight.getIDList()))
            if not discovered:
                raise ValueError("No SUMO traffic lights were discovered during probe.")

            if self.requested_agents:
                missing = [a for a in self.requested_agents if a not in discovered]
                if missing:
                    raise ValueError(
                        "Requested traffic lights are missing from SUMO: "
                        + ", ".join(repr(a) for a in missing)
                    )
                target_ids = self.requested_agents
            else:
                target_ids = discovered

            max_lanes = 0
            for tls_id in target_ids:
                lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
                if len(lanes) > max_lanes:
                    max_lanes = len(lanes)

            self.max_lanes_per_tls = max(max_lanes, 1)
        finally:
            traci.close()
            self._sumo_running = False

    def _discover_agents_and_lanes(self) -> None:
        traci = self._import_traci()

        discovered_agents = list(dict.fromkeys(traci.trafficlight.getIDList()))
        if not discovered_agents:
            raise ValueError("No SUMO traffic lights were discovered.")

        if self.requested_agents:
            missing_agents = [
                agent_id for agent_id in self.requested_agents if agent_id not in discovered_agents
            ]
            if missing_agents:
                raise ValueError(
                    "Requested traffic lights are missing from SUMO: "
                    + ", ".join(repr(agent_id) for agent_id in missing_agents)
                )
            selected_agents = [
                agent_id for agent_id in self.requested_agents if agent_id in discovered_agents
            ]
        else:
            selected_agents = discovered_agents

        self.possible_agents = selected_agents.copy()
        self.agents = selected_agents.copy()

        self._tls_to_lanes = {}
        self._tls_to_green_phases = {}
        self._current_actions = {}
        self._elapsed_green_seconds = {}
        self._switched_last_step = {}

        for tls_id in selected_agents:
            controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
            self._tls_to_lanes[tls_id] = controlled_lanes[: self.max_lanes_per_tls]

            program_logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not program_logics:
                raise ValueError(f"Traffic light {tls_id!r} has no program logic in SUMO.")

            program = program_logics[0]
            green_phases = []
            for phase_index, phase in enumerate(program.phases):
                state = getattr(phase, "state", "")
                state_lower = state.lower()
                if "g" in state_lower and "y" not in state_lower:
                    green_phases.append(phase_index)

            if len(green_phases) != self.green_phase_count:
                raise ValueError(
                    f"Traffic light {tls_id!r} has {len(green_phases)} green phases; "
                    f"expected {self.green_phase_count}."
                )

            self._tls_to_green_phases[tls_id] = green_phases
            self._current_actions[tls_id] = 0
            self._elapsed_green_seconds[tls_id] = 0.0
            self._switched_last_step[tls_id] = False
            traci.trafficlight.setPhase(tls_id, green_phases[0])

    def observation_space(self, agent: str) -> Box:
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.max_lanes_per_tls + 2,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> Discrete:
        return Discrete(self.green_phase_count)

    def _current_phase_normalized(self, agent: str) -> float:
        current_action = self._current_actions.get(agent, 0)
        if self.green_phase_count <= 1:
            return 0.0
        return float(current_action / (self.green_phase_count - 1))

    def _min_green_satisfied(self, agent: str) -> float:
        elapsed = self._elapsed_green_seconds.get(agent, 0.0)
        return 1.0 if elapsed >= self.min_green_seconds else 0.0

    def _elapsed_green_normalized(self, agent: str) -> float:
        """Normalised time spent in the current green phase, in [0, 1].

        Scaled so the min-green threshold sits at 0.25, giving the policy
        graded resolution well past the switch boundary.
        """
        elapsed = self._elapsed_green_seconds.get(agent, 0.0)
        scale = max(self.min_green_seconds, 1) * 4.0
        return float(min(elapsed / scale, 1.0))

    def action_mask(self, agent: str) -> np.ndarray:
        """1.0 for legal actions, 0.0 for actions the env would silently ignore."""
        mask = np.zeros(self.green_phase_count, dtype=np.float32)
        if self._elapsed_green_seconds.get(agent, 0.0) >= self.min_green_seconds:
            mask[:] = 1.0
        else:
            current = self._current_actions.get(agent, 0)
            current = max(0, min(current, self.green_phase_count - 1))
            mask[current] = 1.0
        return mask

    def _get_obs_for_agent(self, agent: str) -> np.ndarray:
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])

        values = []
        for lane_id in lanes[: self.max_lanes_per_tls]:
            queue = traci.lane.getLastStepHaltingNumber(lane_id)
            values.append(min(queue / self.max_queue_value, 1.0))

        while len(values) < self.max_lanes_per_tls:
            values.append(0.0)

        values.append(self._current_phase_normalized(agent))
        values.append(self._elapsed_green_normalized(agent))
        return np.array(values, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        return {agent: self._get_obs_for_agent(agent) for agent in self.agents}

    def _local_queue_stats(self, agent: str) -> tuple[float, float]:
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])
        lane_queues = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes]

        if not lane_queues:
            return 0.0, 0.0

        local_queue = float(sum(lane_queues))
        mean_local_queue = float(np.mean(lane_queues))
        return local_queue, mean_local_queue

    def _reward_for_agent(self, agent: str) -> float:
        _, mean_local_queue = self._local_queue_stats(agent)
        switched = self._switched_last_step.get(agent, False)
        penalty = self.switch_penalty if switched else 0.0
        return -mean_local_queue - penalty

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        self._start_sumo(seed=seed)
        self.step_count = 0
        self._discover_agents_and_lanes()

        observations = self._get_obs()
        infos = {agent: {"action_mask": self.action_mask(agent)} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        traci = self._import_traci()

        for agent in self.agents:
            if agent not in actions:
                continue

            requested_action = int(actions[agent])
            if requested_action < 0 or requested_action >= self.green_phase_count:
                raise ValueError(
                    f"Action {requested_action} is out of range for agent {agent!r}; "
                    f"expected 0 <= action < {self.green_phase_count}."
                )

            current_action = self._current_actions.get(agent, 0)
            min_green_satisfied = self._elapsed_green_seconds.get(agent, 0.0) >= self.min_green_seconds
            switched = False

            if requested_action != current_action and min_green_satisfied:
                phase_index = self._tls_to_green_phases[agent][requested_action]
                traci.trafficlight.setPhase(agent, phase_index)
                self._current_actions[agent] = requested_action
                self._elapsed_green_seconds[agent] = 0.0
                switched = True

            self._switched_last_step[agent] = switched

        for agent in self.agents:
            self._elapsed_green_seconds[agent] = self._elapsed_green_seconds.get(agent, 0.0) + self.seconds_per_action

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

        infos = {}
        for agent in self.agents:
            local_queue, mean_local_queue = self._local_queue_stats(agent)
            current_action = self._current_actions.get(agent, 0)
            infos[agent] = {
                "local_queue": local_queue,
                "mean_local_queue": mean_local_queue,
                "switched": self._switched_last_step.get(agent, False),
                "current_action": current_action,
                "min_green_satisfied": self._min_green_satisfied(agent),
                "action_mask": self.action_mask(agent),
            }

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        if self._sumo_running:
            traci = self._import_traci()
            traci.close()
            self._sumo_running = False
