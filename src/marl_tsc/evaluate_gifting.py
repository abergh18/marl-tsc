from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


# ── Palette ───────────────────────────────────────────────────────────────────

PALETTE = {
    "mappo":          "#1565C0",   # dark blue
    "reward_sharing": "#C62828",   # dark red
    "accent":         "#E65100",   # amber
    "muted":          "#90A4AE",   # blue-grey
    "bg":             "#F5F5F5",
    "panel":          "#FFFFFF",
}

AGENT_COLOURS = [
    "#1565C0", "#C62828", "#2E7D32", "#6A1B9A",
    "#E65100", "#00695C", "#AD1457", "#4E342E", "#37474F",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _short(agent_id: str, n: int = 20) -> str:
    return agent_id[:n] + "…" if len(agent_id) > n else agent_id


def _smooth(values: list, window: int = 20) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="valid")


def _timesteps(history: list) -> np.ndarray:
    return np.array([h["timestep"] for h in history])


def _styled_ax(ax):
    ax.set_facecolor(PALETTE["panel"])
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    return ax


# ── 1. Gifting run overview ───────────────────────────────────────────────────

def _visualise_gifting_run(
    history: list,
    algorithm_name: str,
    output_dir: Path,
    smooth_window: int,
    save: bool,
    show: bool,
):
    """
    3-panel overview:
    1. Training reward (smoothed)
    2. Gift rate + mean gift fraction over time
    3. Mean gift amount over time
    """
    if not history:
        print("Empty history — nothing to plot.")
        return

    ts      = _timesteps(history)
    rewards = [h["mean_training_reward"] for h in history]

    is_gifting = "gift_rate" in history[0]
    gift_rate  = [h.get("gift_rate",          0.0) for h in history] if is_gifting else []
    gift_frac  = [h.get("mean_gift_fraction", 0.0) for h in history] if is_gifting else []
    gift_amt   = [h.get("mean_gift_amount",   0.0) for h in history] if is_gifting else []

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=PALETTE["bg"])
    for ax in axes:
        _styled_ax(ax)

    # ── Panel 1: Training reward ──────────────────────────────────────────────
    ax = axes[0]
    ax.fill_between(ts, rewards, alpha=0.15, color=PALETTE["reward_sharing"])
    if len(rewards) >= smooth_window:
        ax.plot(ts[smooth_window - 1:], _smooth(rewards, smooth_window),
                color=PALETTE["reward_sharing"], lw=2, label="Smoothed")
    ax.set_title("Training Reward", fontsize=11, fontweight="500")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Mean reward", fontsize=9)
    ax.legend(fontsize=8)

    # ── Panel 2: Gift rate + fraction ────────────────────────────────────────
    ax = axes[1]
    if is_gifting:
        ax2 = ax.twinx()
        ax.plot(ts, gift_rate, color=PALETTE["reward_sharing"],
                lw=1.5, label="Gift rate", alpha=0.85)
        ax2.plot(ts, gift_frac, color=PALETTE["mappo"],
                 lw=1.5, label="Mean fraction", alpha=0.85, linestyle="--")
        ax.set_ylabel("Gift rate (proportion of steps)", fontsize=9)
        ax2.set_ylabel("Mean gift fraction", fontsize=9)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="lower right")
        ax2.spines[["top"]].set_visible(False)
    ax.set_title("Gifting Rate & Fraction", fontsize=11, fontweight="500")
    ax.set_xlabel("Timestep", fontsize=9)

    # ── Panel 3: Mean gift amount ─────────────────────────────────────────────
    ax = axes[2]
    if is_gifting:
        ax.fill_between(ts, gift_amt, alpha=0.2, color=PALETTE["accent"])
        ax.plot(ts, gift_amt, color=PALETTE["accent"], lw=1.5)
        if len(gift_amt) >= smooth_window:
            ax.plot(ts[smooth_window - 1:], _smooth(gift_amt, smooth_window),
                    color=PALETTE["reward_sharing"], lw=2, label="Smoothed")
        ax.legend(fontsize=8)
    ax.set_title("Mean Gift Amount", fontsize=11, fontweight="500")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Absolute gift amount", fontsize=9)

    fig.suptitle(f"{algorithm_name} — Gifting Overview",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        out = output_dir / f"{algorithm_name}_overview.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"Saved to {out}")
    if show:
        plt.show()
    plt.close()


# ── 2. Per-agent gifting detail ───────────────────────────────────────────────

def _visualise_per_agent(
    history: list,
    algorithm_name: str,
    output_dir: Path,
    smooth_window: int,
    save: bool,
    show: bool,
):
    """
    3-panel per-agent breakdown:
    1. Mean gift amount per agent over time
    2. Raw reward per agent over time
    3. Gift fraction vs raw reward scatter (final 20%)
    """
    if "per_agent_gifting" not in history[0]:
        print("No per_agent_gifting in history.")
        return

    agent_ids = list(history[0]["per_agent_gifting"].keys())
    n_agents  = len(agent_ids)
    ts        = _timesteps(history)
    cols      = [AGENT_COLOURS[i % len(AGENT_COLOURS)] for i in range(n_agents)]

    # Build per-agent time series
    agent_data = {
        aid: {
            "gift_amt":        [],
            "raw_reward":      [],
            "gift_frac":       [],
            "gift_rate":       [],
            "received_amount": [],
            "net_transfer":    [],
        }
        for aid in agent_ids
    }
    for step in history:
        pag = step.get("per_agent_gifting", {})
        for aid in agent_ids:
            d = pag.get(aid, {})
            agent_data[aid]["gift_amt"].append(d.get("mean_gift_amount",   0.0))
            agent_data[aid]["raw_reward"].append(d.get("mean_raw_reward",  0.0))
            agent_data[aid]["gift_frac"].append(d.get("mean_gift_fraction",0.0))
            agent_data[aid]["gift_rate"].append(d.get("gift_rate",         0.0))
            agent_data[aid]["received_amount"].append(d.get("mean_received_amount", 0.0))
            agent_data[aid]["net_transfer"].append(d.get("mean_net_transfer",       0.0))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), facecolor=PALETTE["bg"])
    for ax in axes:
        _styled_ax(ax)

    # ── Panel 1: Gift amount per agent ────────────────────────────────────────
    ax = axes[0]
    for i, aid in enumerate(agent_ids):
        vals = agent_data[aid]["gift_amt"]
        if len(vals) >= smooth_window:
            ax.plot(ts[smooth_window - 1:], _smooth(vals, smooth_window),
                    color=cols[i], lw=1.5, label=_short(aid, 16))
        else:
            ax.plot(ts, vals, color=cols[i], lw=1.0, label=_short(aid, 16))
    ax.set_title("Mean Gift Amount per Agent", fontsize=10, fontweight="500")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Gift amount", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # ── Panel 2: Raw reward per agent ─────────────────────────────────────────
    ax = axes[1]
    for i, aid in enumerate(agent_ids):
        vals = agent_data[aid]["raw_reward"]
        if len(vals) >= smooth_window:
            ax.plot(ts[smooth_window - 1:], _smooth(vals, smooth_window),
                    color=cols[i], lw=1.5, label=_short(aid, 16))
        else:
            ax.plot(ts, vals, color=cols[i], lw=1.0, label=_short(aid, 16))
    ax.set_title("Raw Traffic Reward per Agent", fontsize=10, fontweight="500")
    ax.set_xlabel("Timestep", fontsize=9)
    ax.set_ylabel("Raw reward (pre-redistribution)", fontsize=9)
    ax.legend(fontsize=6, ncol=2)

    # ── Panel 3: Gift fraction vs raw reward scatter ──────────────────────────
    ax = axes[2]
    cutoff = int(len(history) * 0.8)
    for i, aid in enumerate(agent_ids):
        net   = np.mean(agent_data[aid]["net_transfer"][cutoff:])
        raw_r = np.mean(agent_data[aid]["raw_reward"][cutoff:])
        amt   = np.mean(agent_data[aid]["gift_amt"][cutoff:])
        size  = max(30, min(500, abs(amt) * 100 + 30))
        ax.scatter(net, raw_r, c=cols[i], s=size, alpha=0.85,
                   edgecolors="white", linewidths=0.8, zorder=5)
        ax.annotate(_short(aid, 12), (net, raw_r),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=7, color=cols[i])
    ax.axvline(0, color=PALETTE["muted"], lw=1, linestyle="--", alpha=0.6)  # zero line
    ax.set_xlabel("Mean net transfer (positive = net receiver)", fontsize=9)
    ax.set_ylabel("Mean raw reward", fontsize=9)

    fig.suptitle(f"{algorithm_name} — Per-Agent Gifting Detail",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save:
        out = output_dir / f"{algorithm_name}_per_agent.png"
        plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
        print(f"Saved to {out}")
    if show:
        plt.show()
    plt.close()


# ── Public API ────────────────────────────────────────────────────────────────

def gifting_visualisation(
    history: list,
    agent_ids: list,
    algorithm_name: str = None,
    output_dir: str = ".",
    smooth_window: int = 20,
    save: bool = True,
    show: bool = True,
):
    """
    Generate gifting visualisations from a training history.

    Produces two figures:
    - Overview: training reward, gift rate/fraction, mean gift amount
    - Per-agent: gift amount over time, raw reward over time, scatter
    """
    if not history:
        print("Empty history — nothing to plot.")
        return

    algo       = algorithm_name or history[0].get("algorithm", "reward_sharing_mappo")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _visualise_gifting_run(
        history=history,
        algorithm_name=algo,
        output_dir=output_dir,
        smooth_window=smooth_window,
        save=save,
        show=show,
    )

    _visualise_per_agent(
        history=history,
        algorithm_name=algo,
        output_dir=output_dir,
        smooth_window=smooth_window,
        save=save,
        show=show,
    )


def print_gifting_summary(history: list, agent_ids: list) -> None:
    """Print per-agent gifting summary aggregated across the full training run."""
    gifting_entries = [h for h in history if "per_agent_gifting" in h]

    if not gifting_entries:
        print("No gifting data found in history.")
        return

    print("\n" + "=" * 70)
    print("GIFTING SUMMARY — FULL TRAINING RUN")
    print("=" * 70)

    for agent_id in agent_ids:
        fractions  = [h["per_agent_gifting"][agent_id]["mean_gift_fraction"] for h in gifting_entries]
        amounts    = [h["per_agent_gifting"][agent_id]["mean_gift_amount"]   for h in gifting_entries]
        raw_rs     = [h["per_agent_gifting"][agent_id]["mean_raw_reward"]    for h in gifting_entries]
        gift_rates = [h["per_agent_gifting"][agent_id]["gift_rate"]          for h in gifting_entries]
        received  = [h["per_agent_gifting"][agent_id]["mean_received_amount"] for h in gifting_entries]
        net       = [h["per_agent_gifting"][agent_id]["mean_net_transfer"]    for h in gifting_entries]

        short = agent_id[:30] + "…" if len(agent_id) > 30 else agent_id
        print(f"\n  {short}")
        print(f"Giving:")
        print(f"    Mean gift fraction : {np.mean(fractions):.3f}")
        print(f"    Mean gift amount   : {np.mean(amounts):.4f}")
        print(f"    Mean raw reward    : {np.mean(raw_rs):.4f}")
        print(f"    Gift rate          : {np.mean(gift_rates):.3f}")
        print(f"Receiving and Net:    ")
        print(f"    Mean received amount : {np.mean(received):.4f}")
        print(f"    Mean net transfer    : {np.mean(net):.4f}  {'(net receiver)' if np.mean(net) > 0 else '(net donor)'}")


    print("\n" + "=" * 70)
    print("AGGREGATE")
    print("=" * 70)
    all_fractions = [h["mean_gift_fraction"] for h in gifting_entries]
    all_rates     = [h["gift_rate"]          for h in gifting_entries]
    all_amounts   = [h["mean_gift_amount"]   for h in gifting_entries]
    print(f"  Overall gift rate     : {np.mean(all_rates):.3f}")
    print(f"  Overall mean fraction : {np.mean(all_fractions):.3f}")
    print(f"  Overall mean amount   : {np.mean(all_amounts):.4f}")
    print("=" * 70)


def load_history_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def load_history_pickle(path: str) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)