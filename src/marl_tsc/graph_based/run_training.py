"""
run_training.py  — v2

Changes from v1
---------------
- Extended console logging: entropy, total loss always shown.
- Per-agent gifting detail printed when gifting is active.
- Per-agent gifting added to history dict.
- History saved to JSON at end of training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch


def run_training(
    trainer,
    total_timesteps: int,
    rollout_steps: int,
    algorithm_name: str,
    model_path: str,
    log_interval: int = 10,
):
    """
    Run PPO training loop.

    Parameters
    ----------
    trainer : TrueMAPPOTrainer
    total_timesteps : int
    rollout_steps : int
        Must match trainer.rollout_steps.
    algorithm_name : str
    model_path : str
    log_interval : int
        Print stats every N updates.
    """
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    history  = []
    timestep = 0
    update   = 0

    while timestep < total_timesteps:

        update += 1

        stats = trainer.train_step()

        timestep += rollout_steps

        stats["algorithm"] = algorithm_name
        stats["timestep"]  = timestep
        history.append(stats)

        if update % log_interval == 0:

            # ── Standard metrics ──────────────────────────────────────────────
            print(
                f"Iter {update:5d} | "
                f"Actor {stats.get('actor_loss', 0):8.4f} | "
                f"Critic {stats.get('critic_loss', 0):8.4f} | "
                f"Total {stats.get('total_loss', 0):8.4f} | "
                f"Reward {stats.get('mean_training_reward', 0):8.4f} | "
                f"Entropy {stats.get('entropy_loss', 0):6.4f}"
            )

            # ── Gifting metrics ───────────────────────────────────────────────
            if "gifting_loss" in stats:
                print(
                    f"         Gifting | "
                    f"Loss {stats['gifting_loss']:8.4f} | "
                    f"Rate {stats['gift_rate']:.3f} | "
                    f"MeanFrac {stats['mean_gift_fraction']:.3f} | "
                    f"MeanAmt {stats['mean_gift_amount']:.4f}"
                )

            # ── Per-agent gifting detail ───────────────────────────────────────
            if "per_agent_gifting" in stats and stats["per_agent_gifting"]:
                print("         Per-agent gifting (rollout mean):")
                for agent_id, detail in stats["per_agent_gifting"].items():
                    short_id = agent_id[:30] + "…" if len(agent_id) > 30 else agent_id
                    print(
                        f"           {short_id:<32} | "
                        f"frac={detail['mean_gift_fraction']:.3f} | "
                        f"amt={detail['mean_gift_amount']:.4f} | "
                        f"raw_r={detail['mean_raw_reward']:.4f}"
                    )

    # ── Save model ────────────────────────────────────────────────────────────
    torch.save(trainer.policy.state_dict(), model_path)
    print(f"\nModel saved to {model_path}")

    # ── Save history ──────────────────────────────────────────────────────────
    history_path = model_path.parent / f"{algorithm_name}_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"History saved to {history_path}")

    return trainer.policy, history, str(model_path)