"""MARL traffic signal control helpers."""

from importlib import import_module

__all__ = [
    "CityNetwork",
    "DEFAULT_GRID_TRAFFIC_LIGHT_IDS",
    "DEFAULT_TRAFFIC_LIGHT_IDS",
    "GridNetwork",
    "NetworkType",
    "SimulationGenerator",
    "SimulationPaths",
    "SumoTrafficEnv",
]


def __getattr__(name):
    if name in {"SimulationGenerator", "SimulationPaths"}:
        module = import_module("marl_tsc.simulation_generator")
        return getattr(module, name)

    if name in {
        "CityNetwork",
        "DEFAULT_GRID_TRAFFIC_LIGHT_IDS",
        "DEFAULT_TRAFFIC_LIGHT_IDS",
        "GridNetwork",
        "NetworkType",
    }:
        module = import_module("marl_tsc.network_types")
        return getattr(module, name)

    if name == "SumoTrafficEnv":
        module = import_module("marl_tsc.traffic_env")
        return getattr(module, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
