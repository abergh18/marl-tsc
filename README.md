# marl-tsc
Group project - multi-agent reinforcement learning for traffic signal control

## Setup

Install the project dependencies and package in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## Layout

- `notebooks/main_flow.ipynb` is the main experiment recipe.
- `src/marl_tsc/simulation_generator.py` builds the SUMO network, trips, routes, and config.
- `src/marl_tsc/traffic_env.py` contains the PettingZoo multi-agent SUMO environment.
- `src/marl_tsc/training.py` contains training and evaluation helpers.
- `src/marl_tsc/mappo.py` contains the minimal educational MAPPO baseline.
- `src/marl_tsc/baselines.py` contains simple baseline action helpers.
