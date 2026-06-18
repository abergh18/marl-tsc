"""Simple baseline action helpers for the notebook."""

from __future__ import annotations

import random
from typing import Any


def random_actions(env: Any, step_index: int | None = None) -> dict[str, int]:
    """Return random legal actions for each current agent."""

    agents = list(env.agents or env.possible_agents)
    actions = {}
    for agent in agents:
        if hasattr(env, "action_mask"):
            legal_actions = [
                action_index
                for action_index, is_legal in enumerate(env.action_mask(agent))
                if is_legal
            ]
            actions[agent] = int(random.choice(legal_actions))
        else:
            actions[agent] = int(env.action_space(agent).sample())
    return actions


def fixed_time_actions(env: Any, step_index: int) -> dict[str, int]:
    """Select the next phase on a simple fixed schedule."""

    agents = list(env.agents or env.possible_agents)
    if not agents:
        return {}

    phase_seconds = max(30, env.min_green_seconds)
    steps_per_phase = max(round(phase_seconds / env.seconds_per_action), 1)
    should_advance = step_index > 0 and step_index % steps_per_phase == 0

    actions = {}
    for agent in agents:
        current_phase = int(getattr(env, "_current_actions", {}).get(agent, 0))
        if getattr(env, "phase_action_mode", "cycle") == "direct":
            if should_advance:
                actions[agent] = (current_phase + 1) % env.green_phase_count
            else:
                actions[agent] = current_phase
        elif should_advance:
            actions[agent] = 1
        else:
            actions[agent] = 0
    return actions
