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
        green_phase_count: int | None = None,
        max_queue_value: float = 100.0,
        min_green_seconds: int = 10,
        switch_penalty: float = 0.1,
        seed: int = 42,
        render_mode: str | None = None,
        possible_agents: Sequence[str] | None = None,
    ) -> None:
        if green_phase_count is not None and green_phase_count < 1:
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
        self._latest_lane_queues: dict[str, list[int]] = {}
        self._last_arrived_vehicles = 0
        self._last_mean_waiting_time = 0.0
        self._last_total_time_loss = 0.0
        self._last_vehicle_count = 0
        self._sumo_running = False

        self._traci = None

        if self.max_lanes_per_tls is None or self.green_phase_count is None:
            self._probe_network_structure()

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

    def _probe_network_structure(self) -> None:
        """Briefly start SUMO to size spaces to the network's real configuration."""
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
            detected_phase_counts = set()

            for tls_id in target_ids:
                lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
                max_lanes = max(max_lanes, len(lanes))

                if self.green_phase_count is None:
                    logics = traci.trafficlight.getAllProgramLogics(tls_id)
                    if logics:
                        green_phases = []
                        for phase in logics[0].phases:
                            state = getattr(phase, "state", "").lower()
                            # Identify phases that are "green" but not "yellow"
                            if "g" in state and "y" not in state:
                                green_phases.append(phase)
                        detected_phase_counts.add(len(green_phases))

            if self.max_lanes_per_tls is None:
                self.max_lanes_per_tls = max(max_lanes, 1)

            if self.green_phase_count is None:
                # Default to the most common configuration found, or fallback to 4
                self.green_phase_count = max(detected_phase_counts) if detected_phase_counts else 4
        finally:
            traci.close()
            self._sumo_running = False

    def _discover_agents_and_lanes(self) -> None:
        """Discover traffic lights and their controlled lanes after a simulation start.

        This syncs the environment state with the active SUMO simulation, identifying
        the lanes to monitor and the green phases available for each traffic light.
        """
        traci = self._import_traci()

        discovered_agents = list(dict.fromkeys(traci.trafficlight.getIDList()))
        if not discovered_agents:
            raise ValueError("No SUMO traffic lights were discovered.")

        # Filter discovered agents based on the requested list if provided.
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
        self._latest_lane_queues = {}
        self._last_arrived_vehicles = 0
        self._last_mean_waiting_time = 0.0
        self._last_total_time_loss = 0.0
        self._last_vehicle_count = 0

        # Initialize mapping for each selected traffic light.
        for tls_id in selected_agents:
            controlled_lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
            self._tls_to_lanes[tls_id] = controlled_lanes[: self.max_lanes_per_tls]

            program_logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not program_logics:
                raise ValueError(f"Traffic light {tls_id!r} has no program logic in SUMO.")

            # Parse the traffic light program to identify green phases.
            program = program_logics[0]
            green_phases = []
            for phase_index, phase in enumerate(program.phases):
                state = getattr(phase, "state", "")
                state_lower = state.lower()
                if "g" in state_lower and "y" not in state_lower:
                    green_phases.append(phase_index)

            # Action space validation: ensure SUMO program matches RL agent config.
            if len(green_phases) != self.green_phase_count:
                raise ValueError(
                    f"Traffic light {tls_id!r} has {len(green_phases)} green phases; "
                    f"expected {self.green_phase_count}."
                )

            self._tls_to_green_phases[tls_id] = green_phases
            self._current_actions[tls_id] = 0
            self._elapsed_green_seconds[tls_id] = 0.0
            self._switched_last_step[tls_id] = False
            self._latest_lane_queues[tls_id] = []
            traci.trafficlight.setPhase(tls_id, green_phases[0])

    def observation_space(self, agent: str) -> Box:
        """Returns the observation space for a specific agent."""
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.max_lanes_per_tls + 2,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> Discrete:
        """Returns the action space for a specific agent.

        Action 0 holds the current green phase. Action 1 switches to the next
        green phase when the minimum green time has passed.
        """
        return Discrete(2)

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
        """Returns a binary mask indicating legal actions for the agent.

        Actions are restricted if the minimum green time requirement has not
        been met, forcing the agent to stay in its current phase.
        """
        mask = np.zeros(2, dtype=np.float32)
        mask[0] = 1.0
        if (
            self.green_phase_count > 1
            and self._elapsed_green_seconds.get(agent, 0.0) >= self.min_green_seconds
        ):
            mask[1] = 1.0
        return mask

    def _get_obs_for_agent(self, agent: str) -> np.ndarray:
        """Constructs the observation vector for a single agent.

        Includes normalized vehicle counts on incoming lanes, current phase index,
        and elapsed time in the current phase.
        """
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])
        cached_queues = self._latest_lane_queues.get(agent)

        values = []
        if cached_queues is None or len(cached_queues) != len(lanes[: self.max_lanes_per_tls]):
            cached_queues = [traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes[: self.max_lanes_per_tls]]
            self._latest_lane_queues[agent] = cached_queues

        for queue in cached_queues[: self.max_lanes_per_tls]:
            values.append(min(queue / self.max_queue_value, 1.0))

        while len(values) < self.max_lanes_per_tls:
            values.append(0.0)

        values.append(self._current_phase_normalized(agent))
        values.append(self._elapsed_green_normalized(agent))
        return np.array(values, dtype=np.float32)

    def _get_obs(self) -> dict[str, np.ndarray]:
        """Returns observations for all active agents."""
        return {agent: self._get_obs_for_agent(agent) for agent in self.agents}

    def _local_queue_stats(self, agent: str) -> tuple[float, float, float]:
        lane_queues = self._latest_lane_queues.get(agent, [])

        if not lane_queues:
            return 0.0, 0.0, 0.0

        local_queue = float(sum(lane_queues))
        mean_local_queue = float(np.mean(lane_queues))
        max_local_queue = float(max(lane_queues))
        return local_queue, mean_local_queue, max_local_queue

    def _reward_for_agent(self, agent: str) -> float:
        """Computes the reward for an agent based on local traffic metrics.

        The default implementation penalizes high queue lengths and phase
        switches to encourage stability and flow.
        """
        _, mean_local_queue, max_local_queue = self._local_queue_stats(agent)
        switched = self._switched_last_step.get(agent, False)
        penalty = self.switch_penalty if switched else 0.0
        queue_penalty = 0.8 * mean_local_queue + 0.2 * max_local_queue
        return (-queue_penalty - penalty) * 0.1

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Resets the SUMO simulation and environment state."""
        self._start_sumo(seed=seed)
        self.step_count = 0
        self._discover_agents_and_lanes()

        observations = self._get_obs()
        infos = {agent: {"action_mask": self.action_mask(agent)} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        """Advances the simulation by applying actions for each agent.

        Note: Actions are only applied if the minimum green time is satisfied.
        If a switch is requested before the threshold, it is ignored until
        the next environment step where the condition is met.
        """
        traci = self._import_traci()

        for agent in self.agents:
            if agent not in actions:
                continue

            requested_action = int(actions[agent])
            if requested_action < 0 or requested_action > 1:
                raise ValueError(
                    f"Action {requested_action} is out of range for agent {agent!r}; "
                    "expected 0 for hold or 1 for switch."
                )

            current_action = self._current_actions.get(agent, 0)
            min_green_satisfied = self._elapsed_green_seconds.get(agent, 0.0) >= self.min_green_seconds
            switched = False

            if requested_action == 1 and min_green_satisfied:
                next_action = (current_action + 1) % self.green_phase_count
                phase_index = self._tls_to_green_phases[agent][next_action]
                traci.trafficlight.setPhase(agent, phase_index)
                self._current_actions[agent] = next_action
                self._elapsed_green_seconds[agent] = 0.0
                switched = True

            # Track whether the junction actually changed state this step.
            self._switched_last_step[agent] = switched

        for agent in self.agents:
            self._elapsed_green_seconds[agent] = self._elapsed_green_seconds.get(agent, 0.0) + self.seconds_per_action

        # Perform the actual physics steps in SUMO.
        arrived_vehicles = 0
        for _ in range(self.seconds_per_action):
            traci.simulationStep()
            arrived_vehicles += int(traci.simulation.getArrivedNumber())

        self.step_count += 1
        self._last_arrived_vehicles = arrived_vehicles

        vehicle_ids = list(traci.vehicle.getIDList())
        self._last_vehicle_count = len(vehicle_ids)
        if vehicle_ids:
            waiting_times = [traci.vehicle.getWaitingTime(vehicle_id) for vehicle_id in vehicle_ids]
            time_losses = [traci.vehicle.getTimeLoss(vehicle_id) for vehicle_id in vehicle_ids]
            self._last_mean_waiting_time = float(np.mean(waiting_times))
            self._last_total_time_loss = float(sum(time_losses))
        else:
            self._last_mean_waiting_time = 0.0
            self._last_total_time_loss = 0.0

        # Cache the latest lane queues once per step so observations, rewards,
        # and infos do not each re-query TraCI for the same values.
        for agent in self.agents:
            lanes = self._tls_to_lanes.get(agent, [])
            self._latest_lane_queues[agent] = [
                traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lanes[: self.max_lanes_per_tls]
            ]

        observations = self._get_obs()
        rewards = {agent: self._reward_for_agent(agent) for agent in self.agents}

        # Determine if the episode has ended.
        reached_step_limit = self.step_count >= self.max_steps
        no_more_vehicles_expected = traci.simulation.getMinExpectedNumber() <= 0
        truncated = reached_step_limit or no_more_vehicles_expected

        terminations = {agent: False for agent in self.agents}
        truncations = {agent: truncated for agent in self.agents}

        infos = {}
        for agent in self.agents:
            local_queue, mean_local_queue, max_local_queue = self._local_queue_stats(agent)
            current_action = self._current_actions.get(agent, 0)
            infos[agent] = {
                "local_queue": local_queue,
                "mean_local_queue": mean_local_queue,
                "max_local_queue": max_local_queue,
                "switched": self._switched_last_step.get(agent, False),
                "current_action": current_action,
                "min_green_satisfied": self._min_green_satisfied(agent),
                "action_mask": self.action_mask(agent),
                "arrived_vehicles": self._last_arrived_vehicles,
                "vehicle_count": self._last_vehicle_count,
                "mean_waiting_time": self._last_mean_waiting_time,
                "total_time_loss": self._last_total_time_loss,
            }

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        """Terminates the SUMO simulation and cleans up TraCI."""
        if self._sumo_running:
            traci = self._import_traci()
            traci.close()
            self._sumo_running = False
