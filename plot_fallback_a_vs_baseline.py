"""
Generate Fallback A vs No Fallback comparison plots for README.
"""

import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size'] = 12
matplotlib.rcParams['font.family'] = 'serif'

DATA_DIR = "logs/multi_seed"
OUTPUT_DIR = "plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_seeds(prefix):
    seeds = {}
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.startswith(prefix) and "seed" in fname and fname.endswith(".csv"):
            seed = int(fname.split("seed")[1].split(".")[0])
            with open(os.path.join(DATA_DIR, fname)) as f:
                seeds[seed] = list(csv.DictReader(f))
    return seeds


def extract_rewards(seeds_data):
    n_seeds = len(seeds_data)
    n_rounds = min(len(v) for v in seeds_data.values())
    rewards = np.zeros((n_seeds, n_rounds))
    for i, (seed, rows) in enumerate(sorted(seeds_data.items())):
        for j in range(n_rounds):
            rewards[i, j] = float(rows[j]["eval_reward_mean"])
    return rewards


def extract_field(seeds_data, field):
    n_seeds = len(seeds_data)
    n_rounds = min(len(v) for v in seeds_data.values())
    data = np.zeros((n_seeds, n_rounds))
    for i, (seed, rows) in enumerate(sorted(seeds_data.items())):
        for j in range(n_rounds):
            val = rows[j].get(field, "0")
            if val in ("True", "False"):
                data[i, j] = 1.0 if val == "True" else 0.0
            else:
                try:
                    data[i, j] = float(val)
                except:
                    data[i, j] = 0.0
    return data


nf_data = load_seeds("Nitro_no_fallback~Hopper-v4~ppo_")
a_data = load_seeds("Nitro_fallback_A~Hopper-v4~ppo_")

nf_rewards = extract_rewards(nf_data)
a_rewards = extract_rewards(a_data)
rounds = np.arange(1, 51)

C_NF = "#2196F3"
C_A = "#E53935"

# ---- Figure 1: Reward Curve ----
fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))
for rewards, label, color in [
    (nf_rewards, "No Fallback (Original Nitro)", C_NF),
    (a_rewards, "Nitro + Fallback (Ours)", C_A),
]:
    mean = rewards.mean(axis=0)
    std = rewards.std(axis=0)
    ax.plot(rounds, mean, color=color, linewidth=2.2, label=label)
    ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.15)

ax.set_xlabel("Training Round", fontsize=13)
ax.set_ylabel("Mean Episodic Reward", fontsize=13)
ax.set_title("Hopper-v4 (PPO): Nitro vs Nitro + Fallback", fontsize=14)
ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3)
ax.set_xlim(1, 50)
ax.set_ylim(0, 350)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fallback_a_reward_curve.png"), dpi=150)
print("Saved: fallback_a_reward_curve.png")

# ---- Figure 2: Bar Chart ----
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
labels = ["Original Nitro", "Nitro + Fallback"]
colors = [C_NF, C_A]

last20 = [nf_rewards[:, -20:].mean(), a_rewards[:, -20:].mean()]
last20_err = [nf_rewards[:, -20:].mean(axis=1).std(), a_rewards[:, -20:].mean(axis=1).std()]
axes[0].bar(labels, last20, yerr=last20_err, color=colors, alpha=0.85, capsize=5, edgecolor='black', linewidth=0.5)
axes[0].set_ylabel("Mean Reward")
axes[0].set_title("(a) Last 20 Rounds", fontsize=12)
axes[0].set_ylim(0, 300)
for i, v in enumerate(last20):
    axes[0].text(i, v + last20_err[i] + 5, f"{v:.0f}", ha='center', fontsize=11, fontweight='bold')

peaks = [nf_rewards.mean(axis=0).max(), a_rewards.mean(axis=0).max()]
axes[1].bar(labels, peaks, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[1].set_ylabel("Peak Mean Reward")
axes[1].set_title("(b) Peak Performance", fontsize=12)
axes[1].set_ylim(0, 320)
for i, v in enumerate(peaks):
    axes[1].text(i, v + 5, f"{v:.0f}", ha='center', fontsize=11, fontweight='bold')

stab = [nf_rewards[:, -20:].std(axis=0).mean(), a_rewards[:, -20:].std(axis=0).mean()]
axes[2].bar(labels, stab, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
axes[2].set_ylabel("Avg Std (lower = stabler)")
axes[2].set_title("(c) Stability (Last 20)", fontsize=12)
axes[2].set_ylim(0, 60)
for i, v in enumerate(stab):
    axes[2].text(i, v + 1.5, f"{v:.1f}", ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fallback_a_bars.png"), dpi=150)
print("Saved: fallback_a_bars.png")

# ---- Figure 3: Rollback Events ----
a_rollbacks = extract_field(a_data, "rollback")
a_workers = extract_field(a_data, "num_rollout_workers")

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

ax = axes[0]
mean_a = a_rewards.mean(axis=0)
std_a = a_rewards.std(axis=0)
ax.plot(rounds, mean_a, color=C_A, linewidth=2, label="Nitro + Fallback")
ax.fill_between(rounds, mean_a - std_a, mean_a + std_a, color=C_A, alpha=0.15)
for i in range(a_rollbacks.shape[0]):
    rb_rounds = rounds[a_rollbacks[i] > 0]
    rb_rewards = a_rewards[i][a_rollbacks[i] > 0]
    if len(rb_rounds) > 0:
        ax.scatter(rb_rounds, rb_rewards, color='black', marker='x', s=120, zorder=5, linewidths=2,
                   label="Rollback Event" if i == 0 else None)
ax.set_ylabel("Episodic Reward", fontsize=12)
ax.set_title("Fallback A: Reward with Rollback Events", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 380)

ax = axes[1]
mean_w = a_workers.mean(axis=0)
ax.bar(rounds, mean_w, color=C_A, alpha=0.6, width=0.8)
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label="Baseline (1 worker)")
ax.set_ylabel("Num Workers", fontsize=12)
ax.set_xlabel("Training Round", fontsize=12)
ax.set_title("Dynamic Actor Scaling", fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(0.5, 50.5)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "fallback_a_rollback.png"), dpi=150)
print("Saved: fallback_a_rollback.png")

print("\nDone!")
