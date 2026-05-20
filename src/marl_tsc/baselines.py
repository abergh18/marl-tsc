"""Simple baseline action helpers for the notebook."""

from __future__ import annotations

from typing import Any


def random_actions(env: Any, step_index: int | None = None) -> dict[str, int]:
    """Return random legal actions for each current agent."""

    agents = list(env.agents or env.possible_agents)
    return {agent: int(env.action_space(agent).sample()) for agent in agents}


def fixed_time_actions(env: Any, step_index: int) -> dict[str, int]:
    """Cycle all agents through the same action at a given step."""

    agents = list(env.agents or env.possible_agents)
    if not agents:
        return {}

    action = step_index % env.green_phase_count
    return {agent: action for agent in agents}
