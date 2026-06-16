"""
run_training.py

Generic training orchestration utilities.

Why this exists
---------------
Many graph-based algorithms share the same experiment workflow:

    Create trainer
        ↓
    Run train_step repeatedly
        ↓
    Collect training history
        ↓
    Save trained model
        ↓
    Return results

The learning algorithm itself remains inside the trainer.

This module exists to avoid duplicating experiment-management code across:

    GAT CTDE
    Hypergraph CTDE
    Graph PPO
    Future graph-based methods

while keeping the training logic itself algorithm-specific.
"""

from __future__ import annotations

from pathlib import Path

import torch


def run_training(
    trainer,
    total_timesteps,
    rollout_steps,
    algorithm_name,
    model_path=None,
    log_interval=10,
):

    history = []

    timestep = 0
    update = 0

    while timestep < total_timesteps:

        update += 1

        stats = trainer.train_step()

        timestep += rollout_steps

        history.append(
            {
                "algorithm": algorithm_name,
                "timestep": timestep,
                "mean_training_reward": 0.0,
                "actor_loss": stats.get(
                    "actor_loss",
                    0.0,
                ),
                "critic_loss": stats.get(
                    "critic_loss",
                    0.0,
                ),
                "total_loss": stats.get(
                    "total_loss",
                    0.0,
                ),
            }
        )

        if update % log_interval == 0:

            print(
                f"Iter {update:>3} | "
                f"Actor {stats.get('actor_loss', 0.0):.4f} | "
                f"Critic {stats.get('critic_loss', 0.0):.4f}"
            )

    if model_path is not None:

        model_path = Path(model_path)

        model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        torch.save(
            trainer.policy.state_dict(),
            model_path,
        )

    return (
        trainer.policy,
        history,
        str(model_path)
        if model_path is not None
        else None,
    )