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
        min_green_seconds: int = 5,
        max_green_seconds: int = 60,
        max_red_seconds: int = 60,
        yellow_seconds: int = 3,
        switch_penalty: float = 0.01,
        phase_action_mode: str = "direct",
        seed: int = 42,
        render_mode: str | None = None,
        possible_agents: Sequence[str] | None = None,
        collect_global_metrics: bool = True,
        global_metric_interval: int = 1,
        include_phase_queue_features: bool = True,
        include_action_mask_features: bool = True,
    ) -> None:
        if green_phase_count is not None and green_phase_count < 1:
            raise ValueError("green_phase_count must be at least 1.")
        if global_metric_interval < 1:
            raise ValueError("global_metric_interval must be at least 1.")
        if min_green_seconds < 0:
            raise ValueError("min_green_seconds cannot be negative.")
        if max_green_seconds < max(min_green_seconds, 1):
            raise ValueError("max_green_seconds must be at least min_green_seconds.")
        if max_red_seconds < max(min_green_seconds, 1):
            raise ValueError("max_red_seconds must be at least min_green_seconds.")
        if yellow_seconds < 0 or yellow_seconds >= seconds_per_action:
            raise ValueError("yellow_seconds must be non-negative and less than seconds_per_action.")
        if phase_action_mode not in {"cycle", "direct"}:
            raise ValueError("phase_action_mode must be 'cycle' or 'direct'.")

        self.config_file = Path(config_file)
        self.max_steps = max_steps
        self.seconds_per_action = seconds_per_action
        self.max_lanes_per_tls = max_lanes_per_tls
        self.green_phase_count = green_phase_count
        self.max_queue_value = max_queue_value
        self.min_green_seconds = min_green_seconds
        self.max_green_seconds = max_green_seconds
        self.max_red_seconds = max_red_seconds
        self.yellow_seconds = yellow_seconds
        self.switch_penalty = switch_penalty
        self.phase_action_mode = phase_action_mode
        self.seed = seed
        self.render_mode = render_mode
        self.collect_global_metrics = collect_global_metrics
        self.global_metric_interval = global_metric_interval
        self.include_phase_queue_features = include_phase_queue_features
        self.include_action_mask_features = include_action_mask_features

        self.step_count = 0
        self.requested_agents = list(dict.fromkeys(possible_agents)) if possible_agents else None
        self.possible_agents: list[str] = list(self.requested_agents or [])
        self.agents: list[str] = self.possible_agents.copy()
        self._tls_to_lanes: dict[str, list[str]] = {}
        self._tls_to_green_phases: dict[str, list[int]] = {}
        self._tls_to_green_states: dict[str, list[str]] = {}
        self._tls_to_program_ids: dict[str, str] = {}
        self._tls_to_phase_lanes: dict[str, list[list[str]]] = {}
        self._current_actions: dict[str, int] = {}
        self._elapsed_green_seconds: dict[str, float] = {}
        self._phase_red_seconds: dict[str, list[float]] = {}
        self._switched_last_step: dict[str, bool] = {}
        self._latest_lane_queues: dict[str, list[int]] = {}
        self._prev_waiting_time: dict[str, float] = {}
        self._latest_max_waiting_times: dict[str, float] = {}
        self._previous_mean_local_queue: dict[str, float] = {}
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
    
            # Check if libsumo has been explicitly requested
            use_libsumo = os.environ.get("USE_LIBSUMO", "0") == "1"
    
            if use_libsumo:
                try:
                    import libsumo as traci  # type: ignore
                except ImportError:
                    import traci  # type: ignore
            else:
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
                "--no-step-log",
                "true",
            ]
        )
        self._sumo_running = True

    def _selected_agents(self, discovered_agents: Sequence[str]) -> list[str]:
        """Return requested traffic lights, or all discovered lights if none were requested."""
        discovered_agents = list(dict.fromkeys(discovered_agents))
        if not discovered_agents:
            raise ValueError("No SUMO traffic lights were discovered.")

        if not self.requested_agents:
            return discovered_agents

        missing_agents = [
            agent_id for agent_id in self.requested_agents if agent_id not in discovered_agents
        ]
        if missing_agents:
            missing = ", ".join(repr(agent_id) for agent_id in missing_agents)
            raise ValueError(f"Requested traffic lights are missing from SUMO: {missing}")

        return self.requested_agents.copy()

    @staticmethod
    def _green_phase_indexes(phases) -> list[int]:
        green_phases = []
        for phase_index, phase in enumerate(phases):
            state = getattr(phase, "state", "").lower()
            if "g" in state and "y" not in state:
                green_phases.append(phase_index)
        return green_phases

    def _set_green_phase(self, traffic_light_id: str, phase_index: int) -> None:
        """Select and hold a green phase until the environment changes it.

        SUMO otherwise advances to the next programmed phase when the phase's
        original fixed-time duration expires.  That would make the physical
        signal disagree with ``_current_actions`` and with the observation and
        action mask exposed to the policy.
        """
        traci = self._import_traci()
        program_id = self._tls_to_program_ids.get(traffic_light_id)
        if program_id is not None:
            traci.trafficlight.setProgram(traffic_light_id, program_id)
        traci.trafficlight.setPhase(traffic_light_id, phase_index)
        traci.trafficlight.setPhaseDuration(traffic_light_id, 1_000_000.0)

    @staticmethod
    def _yellow_transition_state(current_state: str, target_state: str) -> str:
        """Build a safe yellow state for movements losing right of way."""
        return "".join(
            current
            if current in "gG" and target in "gG"
            else "y"
            if current in "gG"
            else "r"
            for current, target in zip(current_state, target_state)
        )

    def _probe_network_structure(self) -> None:
        """Briefly start SUMO to size spaces to the network's real configuration."""
        traci = self._import_traci()
        self._start_sumo(seed=self.seed)
        try:
            target_ids = self._selected_agents(traci.trafficlight.getIDList())

            max_lanes = 0
            detected_phase_counts = set()

            for tls_id in target_ids:
                lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(tls_id)))
                max_lanes = max(max_lanes, len(lanes))

                if self.green_phase_count is None:
                    logics = traci.trafficlight.getAllProgramLogics(tls_id)
                    if logics:
                        detected_phase_counts.add(len(self._green_phase_indexes(logics[0].phases)))

            if self.max_lanes_per_tls is None:
                self.max_lanes_per_tls = max(max_lanes, 1)

            if self.green_phase_count is None:
                self.green_phase_count = max(detected_phase_counts) if detected_phase_counts else 4
        finally:
            traci.close()
            self._sumo_running = False

    def _discover_agents_and_lanes(self) -> None:
        """Discover traffic lights, lanes, and green phases after SUMO starts."""
        traci = self._import_traci()

        selected_agents = self._selected_agents(traci.trafficlight.getIDList())

        self.possible_agents = selected_agents.copy()
        self.agents = selected_agents.copy()

        self._tls_to_lanes = {}
        self._tls_to_green_phases = {}
        self._tls_to_green_states = {}
        self._tls_to_program_ids = {}
        self._tls_to_phase_lanes = {}
        self._current_actions = {}
        self._elapsed_green_seconds = {}
        self._phase_red_seconds = {}
        self._switched_last_step = {}
        self._latest_lane_queues = {}
        self._latest_max_waiting_times = {}
        self._previous_mean_local_queue = {}
        self._last_arrived_vehicles = 0
        self._last_mean_waiting_time = 0.0
        self._last_total_time_loss = 0.0
        self._last_vehicle_count = 0
        self._prev_waiting_time = {agent: 0.0 for agent in selected_agents}

        for tls_id in selected_agents:
            controlled_links_lanes = list(traci.trafficlight.getControlledLanes(tls_id))
            controlled_lanes = list(dict.fromkeys(controlled_links_lanes))
            self._tls_to_lanes[tls_id] = controlled_lanes[: self.max_lanes_per_tls]

            program_logics = traci.trafficlight.getAllProgramLogics(tls_id)
            if not program_logics:
                raise ValueError(f"Traffic light {tls_id!r} has no program logic in SUMO.")

            program = program_logics[0]
            green_phases = self._green_phase_indexes(program.phases)

            if len(green_phases) > self.green_phase_count:
                raise ValueError(
                    f"Traffic light {tls_id!r} has {len(green_phases)} green phases; "
                    f"configured action capacity is {self.green_phase_count}."
                )

            self._tls_to_green_phases[tls_id] = green_phases
            self._tls_to_program_ids[tls_id] = program.programID
            self._tls_to_green_states[tls_id] = [
                program.phases[phase_index].state for phase_index in green_phases
            ]
            self._tls_to_phase_lanes[tls_id] = [
                list(
                    dict.fromkeys(
                        lane_id
                        for lane_id, signal_state in zip(
                            controlled_links_lanes,
                            program.phases[phase_index].state,
                        )
                        if signal_state.lower() == "g"
                    )
                )
                for phase_index in green_phases
            ]
            self._current_actions[tls_id] = 0
            self._elapsed_green_seconds[tls_id] = 0.0
            self._phase_red_seconds[tls_id] = [0.0] * len(green_phases)
            self._switched_last_step[tls_id] = False
            self._latest_lane_queues[tls_id] = []
            self._latest_max_waiting_times[tls_id] = 0.0
            self._previous_mean_local_queue[tls_id] = 0.0
            self._set_green_phase(tls_id, green_phases[0])

    def observation_space(self, agent: str) -> Box:
        """Returns the observation space for a specific agent."""
        phase_feature_count = self.green_phase_count if self.include_phase_queue_features else 0
        mask_feature_count = self._action_count() if self.include_action_mask_features else 0
        return Box(
            low=0.0,
            high=1.0,
            shape=(self.max_lanes_per_tls + 2 + phase_feature_count + mask_feature_count,),
            dtype=np.float32,
        )

    def action_space(self, agent: str) -> Discrete:
        """Returns the action space for a specific agent.

        In cycle mode, action 0 holds the current phase and action 1 advances
        to the next green phase. Direct mode selects a green phase by index.
        """
        return Discrete(self._action_count())

    def _action_count(self) -> int:
        return 2 if self.phase_action_mode == "cycle" else self.green_phase_count

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
        scale = max(self.max_green_seconds, 1)
        return float(min(elapsed / scale, 1.0))

    def action_mask(self, agent: str) -> np.ndarray:
        """Returns a binary mask indicating legal actions for the agent.

        Switching is disabled before the minimum green time. Holding is disabled
        at the maximum green time so even a deterministic policy cannot starve
        all other movements indefinitely.
        """
        mask = np.zeros(self._action_count(), dtype=np.float32)

        current_action = self._current_actions.get(agent, 0)
        elapsed = self._elapsed_green_seconds.get(agent, 0.0)
        valid_phase_count = len(self._tls_to_green_phases.get(agent, []))

        if self.phase_action_mode == "cycle":
            mask[0] = 1.0
            if elapsed >= self.min_green_seconds and valid_phase_count > 1:
                mask[1] = 1.0
            if elapsed >= self.max_green_seconds and valid_phase_count > 1:
                mask[0] = 0.0
        else:
            mask[current_action] = 1.0
            if elapsed >= self.min_green_seconds:
                mask[:valid_phase_count] = 1.0
            if elapsed >= self.max_green_seconds and valid_phase_count > 1:
                mask[current_action] = 0.0

        return mask

    def _phase_queue_totals(self, agent: str) -> list[float]:
        lane_ids = self._tls_to_lanes.get(agent, [])[: self.max_lanes_per_tls]
        cached_queues = self._latest_lane_queues.get(agent, [])
        lane_to_queue = {
            lane_id: float(queue)
            for lane_id, queue in zip(lane_ids, cached_queues[: self.max_lanes_per_tls])
        }

        phase_lane_groups = self._tls_to_phase_lanes.get(agent, [])
        phase_queues = []
        for phase_index in range(self.green_phase_count):
            phase_lanes = (
                phase_lane_groups[phase_index]
                if phase_index < len(phase_lane_groups)
                else []
            )
            phase_queues.append(
                float(sum(lane_to_queue.get(lane_id, 0.0) for lane_id in phase_lanes))
            )
        return phase_queues

    def _get_obs_for_agent(self, agent: str) -> np.ndarray:
        """Constructs the observation vector for a single agent.

        Includes normalized vehicle counts on incoming lanes, current phase index,
        and elapsed time in the current phase.
        """
        traci = self._import_traci()
        lanes = self._tls_to_lanes.get(agent, [])
        cached_queues = self._latest_lane_queues.get(agent)
        lane_ids = lanes[: self.max_lanes_per_tls]

        if cached_queues is None or len(cached_queues) != len(lane_ids):
            cached_queues = [
                traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids
            ]
            self._latest_lane_queues[agent] = cached_queues

        values = []
        for queue in cached_queues[: self.max_lanes_per_tls]:
            values.append(min(queue / self.max_queue_value, 1.0))

        while len(values) < self.max_lanes_per_tls:
            values.append(0.0)

        values.append(self._current_phase_normalized(agent))
        values.append(self._elapsed_green_normalized(agent))

        if self.include_phase_queue_features:
            for phase_queue in self._phase_queue_totals(agent):
                values.append(min(phase_queue / self.max_queue_value, 1.0))

        if self.include_action_mask_features:
            values.extend(float(value) for value in self.action_mask(agent))

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
        """
        Computes the reward for a traffic light agent.
        
        This uses a positive baseline score and squares the queue lengths 
        to ensure fairness across all lanes and prevent the cartel exploit.
        """
        traci = self._import_traci()
        
        lanes = self._tls_to_lanes.get(agent, [])
        
        total_penalty = 0.0
        
        for lane_id in lanes:
            halted = traci.lane.getLastStepHaltingNumber(lane_id)
            
            total_penalty += float(halted ** 2)
            
        switched = self._switched_last_step.get(agent, False)
        switch_penalty = self.switch_penalty if switched else 0.0
        
        baseline_score = 100.0
        reward = baseline_score - total_penalty - switch_penalty
        
        return max(0.0, reward)

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        """Resets the SUMO simulation and environment state."""
        self._start_sumo(seed=seed)
        self.step_count = 0
        self._discover_agents_and_lanes()

        observations = self._get_obs()
        for agent in self.agents:
            _, mean_local_queue, _ = self._local_queue_stats(agent)
            self._previous_mean_local_queue[agent] = mean_local_queue

        infos = {agent: {"action_mask": self.action_mask(agent)} for agent in self.agents}
        return observations, infos

    def step(self, actions: dict[str, int]):
        """Advances the simulation by applying actions for each agent.

        Note: Phase changes are only applied if the minimum green time is
        satisfied. Early change requests are ignored.
        """
        traci = self._import_traci()
        pending_switches: dict[str, list[int]] = {}

        for agent in self.agents:
            self._switched_last_step[agent] = False

        for agent in self.agents:
            if agent not in actions:
                continue

            requested_action = int(actions[agent])
            action_count = self._action_count()
            if requested_action < 0 or requested_action >= action_count:
                raise ValueError(
                    f"Action {requested_action} is out of range for agent {agent!r}; "
                    f"expected an action from 0 to {action_count - 1}."
                )

            current_action = self._current_actions.get(agent, 0)
            elapsed_green = self._elapsed_green_seconds.get(agent, 0.0)
            min_green_satisfied = elapsed_green >= self.min_green_seconds
            switched = False

            valid_phase_count = len(self._tls_to_green_phases.get(agent, []))
            max_green_reached = elapsed_green >= self.max_green_seconds
            if self.phase_action_mode == "cycle":
                advance = requested_action == 1 or max_green_reached
                target_action = (
                    (current_action + 1) % valid_phase_count
                    if advance and valid_phase_count > 1
                    else current_action
                )
            else:
                target_action = requested_action
                if max_green_reached and target_action == current_action and valid_phase_count > 1:
                    target_action = (current_action + 1) % valid_phase_count
            should_switch = (
                target_action < valid_phase_count
                and target_action != current_action
                and min_green_satisfied
            )

            if should_switch:
                phase_index = self._tls_to_green_phases[agent][target_action]
                if self.yellow_seconds:
                    current_state = traci.trafficlight.getRedYellowGreenState(agent)
                    target_state = self._tls_to_green_states[agent][target_action]
                    yellow_state = self._yellow_transition_state(current_state, target_state)
                    traci.trafficlight.setRedYellowGreenState(agent, yellow_state)
                    pending_switches[agent] = [
                        self.yellow_seconds,
                        phase_index,
                        target_action,
                    ]
                else:
                    self._set_green_phase(agent, phase_index)
                    self._current_actions[agent] = target_action
                    self._elapsed_green_seconds[agent] = 0.0
                switched = True

            # Track whether the junction actually changed state this step.
            self._switched_last_step[agent] = switched

        # Perform the actual physics steps in SUMO.
        arrived_vehicles = 0
        for _ in range(self.seconds_per_action):
            traci.simulationStep()
            arrived_vehicles += int(traci.simulation.getArrivedNumber())
            for agent in self.agents:
                pending = pending_switches.get(agent)
                if pending is None:
                    self._elapsed_green_seconds[agent] = (
                        self._elapsed_green_seconds.get(agent, 0.0) + 1.0
                    )
                    continue

                pending[0] -= 1
                if pending[0] <= 0:
                    _, phase_index, target_action = pending
                    self._set_green_phase(agent, phase_index)
                    self._current_actions[agent] = target_action
                    self._elapsed_green_seconds[agent] = 0.0
                    del pending_switches[agent]

        self.step_count += 1
        self._last_arrived_vehicles = arrived_vehicles
        global_metrics_updated = False

        if (
            self.collect_global_metrics
            and self.step_count % self.global_metric_interval == 0
        ):
            vehicle_ids = list(traci.vehicle.getIDList())
            self._last_vehicle_count = len(vehicle_ids)
            if vehicle_ids:
                waiting_times = [
                    traci.vehicle.getWaitingTime(vehicle_id) for vehicle_id in vehicle_ids
                ]
                time_losses = [traci.vehicle.getTimeLoss(vehicle_id) for vehicle_id in vehicle_ids]
                self._last_mean_waiting_time = float(np.mean(waiting_times))
                self._last_total_time_loss = float(sum(time_losses))
            else:
                self._last_mean_waiting_time = 0.0
                self._last_total_time_loss = 0.0
            global_metrics_updated = True

        # Cache the latest lane queues once per step so observations, rewards,
        # and infos do not each re-query TraCI for the same values.
        for agent in self.agents:
            lanes = self._tls_to_lanes.get(agent, [])
            lane_ids = lanes[: self.max_lanes_per_tls]
            self._latest_lane_queues[agent] = [
                traci.lane.getLastStepHaltingNumber(lane_id) for lane_id in lane_ids
            ]
            waiting_times = []
            for lane_id in lane_ids:
                waiting_times.extend(
                    traci.vehicle.getWaitingTime(vehicle_id)
                    for vehicle_id in traci.lane.getLastStepVehicleIDs(lane_id)
                )
            self._latest_max_waiting_times[agent] = (
                float(max(waiting_times)) if waiting_times else 0.0
            )

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
                "max_waiting_time": self._latest_max_waiting_times.get(agent, 0.0),
                "switched": self._switched_last_step.get(agent, False),
                "current_action": current_action,
                "min_green_satisfied": self._min_green_satisfied(agent),
                "action_mask": self.action_mask(agent),
                "arrived_vehicles": self._last_arrived_vehicles,
                "vehicle_count": self._last_vehicle_count,
                "mean_waiting_time": self._last_mean_waiting_time,
                "total_time_loss": self._last_total_time_loss,
                "global_metrics_updated": global_metrics_updated,
            }
            self._previous_mean_local_queue[agent] = mean_local_queue

        if truncated:
            self.agents = []

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        """Terminates the SUMO simulation and cleans up TraCI."""
        if self._sumo_running:
            traci = self._import_traci()
            traci.close()
            self._sumo_running = False
