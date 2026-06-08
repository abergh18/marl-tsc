"""Simple baseline action helpers for the notebook."""

from __future__ import annotations

from typing import Any


def random_actions(env: Any, step_index: int | None = None) -> dict[str, int]:
    """Return random legal actions for each current agent."""

    agents = list(env.agents or env.possible_agents)
    return {agent: int(env.action_space(agent).sample()) for agent in agents}


def fixed_time_actions(env: Any, step_index: int) -> dict[str, int]:
    """Switch phases on a simple fixed schedule."""

    agents = list(env.agents or env.possible_agents)
    if not agents:
        return {}

    steps_per_phase = max(env.min_green_seconds // env.seconds_per_action, 1)
    action = 1 if (step_index + 1) % steps_per_phase == 0 else 0
    return {agent: action for agent in agents}
