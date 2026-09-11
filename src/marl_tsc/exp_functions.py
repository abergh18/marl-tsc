import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def plot_variance(histories_list, label, colour, ax, metric="mean_training_reward", smooth=50):
    # Align to shortest history
    min_len = min(len(h) for h in histories_list)
    ts      = np.array([h["timestep"] for h in histories_list[0][:min_len]])
    
    # Stack metric across seeds
    values = np.array([
        [step.get(metric, 0.0) for step in h[:min_len]]
        for h in histories_list
    ])  # shape (n_seeds, n_steps)
    
    # Smooth each seed
    def smooth_fn(arr, w):
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")
    
    smoothed = np.array([smooth_fn(v, smooth) for v in values])
    ts_smooth = ts[smooth - 1:]
    
    mean = smoothed.mean(axis=0)
    std  = smoothed.std(axis=0)
    
    ax.plot(ts_smooth, mean, color=colour, lw=2, label=label)
    ax.fill_between(ts_smooth, mean - std, mean + std, alpha=0.2, color=colour)

def save_history(history, seed, name, OUTPUT_DIR):
    with open(Path(OUTPUT_DIR) / f"history_{name}_seed{seed}.json", "w") as f:
        json.dump(history, f)