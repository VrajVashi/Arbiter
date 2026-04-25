"""Arms-race visualisation for ARBITER GRPO training logs.

Reads one or more JSONL reward logs and plots:
  1. Reward curve with level-advance markers
  2. Defender evasion rate over episodes
  3. Per-level reward distribution (box plot)

Usage:
    python -m arbiter.training.analyze_arms_race \
        logs/grpo_level1.jsonl logs/grpo_level2.jsonl \
        logs/grpo_level3.jsonl logs/grpo_level4.jsonl \
        --out plots/arms_race.png
"""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("logs", nargs="+", help="JSONL reward log files in training order")
parser.add_argument("--out", default="plots/arms_race.png")
parser.add_argument("--smooth", type=int, default=10, help="Rolling window for smoothing")
args = parser.parse_args()

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("matplotlib not installed — printing table instead.")


def load_log(path: str):
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def rolling_mean(vals, w):
    out = []
    for i in range(len(vals)):
        window = vals[max(0, i - w + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


# ── Load all logs ──────────────────────────────────────────────────────────────
all_entries = []
for log_path in args.logs:
    p = Path(log_path)
    if not p.exists():
        print(f"WARNING: {log_path} not found, skipping.")
        continue
    entries = load_log(log_path)
    # offset episode index to be globally sequential
    offset = all_entries[-1]["_global_ep"] + 1 if all_entries else 0
    for i, e in enumerate(entries):
        e["_global_ep"] = offset + i
        e["_source"]    = p.stem
    all_entries.extend(entries)
    print(f"Loaded {len(entries)} entries from {log_path}")

if not all_entries:
    print("No log entries found.")
    exit(1)

global_eps    = [e["_global_ep"]         for e in all_entries]
rewards       = [e["mean_reward"]        for e in all_entries]
evasion       = [e["defender_evasion"]   for e in all_entries]
levels        = [e["level"]              for e in all_entries]
level_advances = [(e["_global_ep"], e["level"]) for e in all_entries if e.get("level_advanced")]

smooth_rewards = rolling_mean(rewards, args.smooth)
smooth_evasion = rolling_mean(evasion, args.smooth)

# ── Print text summary ─────────────────────────────────────────────────────────
print("\n=== ARMS RACE SUMMARY ===")
print(f"{'Level':<8} {'Episodes':<10} {'Mean Reward':<14} {'Mean Evasion':<14} {'Final Reward'}")
print("-" * 60)
for lvl in sorted(set(levels)):
    mask = [i for i, l in enumerate(levels) if l == lvl]
    if not mask:
        continue
    lvl_rewards  = [rewards[i]  for i in mask]
    lvl_evasion  = [evasion[i]  for i in mask]
    print(f"{lvl:<8} {len(mask):<10} {np.mean(lvl_rewards):<14.2f} "
          f"{np.mean(lvl_evasion):<14.3f} {lvl_rewards[-1]:.2f}")

# Defender activation point
defender_start = next((e["_global_ep"] for e in all_entries if e["level"] >= 4), None)
if defender_start is not None:
    print(f"\nDefender activated at global episode {defender_start} (level 4+)")
    pre  = [r for e, r in zip(all_entries, rewards) if e["level"] < 4]
    post = [r for e, r in zip(all_entries, rewards) if e["level"] >= 4]
    if pre and post:
        print(f"  Mean reward before Defender: {np.mean(pre[-20:]):.2f}")
        print(f"  Mean reward after  Defender: {np.mean(post[:20]):.2f}  (first 20 eps)")
        print(f"  Recovery by end:             {np.mean(post[-20:]):.2f}  (last 20 eps)")
else:
    print("\nDefender not yet activated (need level 4+ data).")

# ── Plot ───────────────────────────────────────────────────────────────────────
if not HAS_PLT:
    exit(0)

Path(args.out).parent.mkdir(parents=True, exist_ok=True)

LEVEL_COLORS = {1: "#4CAF50", 2: "#8BC34A", 3: "#FFC107",
                4: "#FF5722", 5: "#E91E63", 6: "#9C27B0", 7: "#3F51B5"}

fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig.suptitle("ARBITER Arms Race: Auditor vs Defender", fontsize=14, fontweight="bold")

# ── Panel 1: Reward curve ──────────────────────────────────────────────────────
ax1 = axes[0]
# shade background by level
prev_ep, prev_lvl = global_eps[0], levels[0]
for i, (ep, lvl) in enumerate(zip(global_eps, levels)):
    if lvl != prev_lvl or i == len(global_eps) - 1:
        ax1.axvspan(prev_ep, ep, alpha=0.08, color=LEVEL_COLORS.get(prev_lvl, "gray"))
        prev_ep, prev_lvl = ep, lvl

ax1.plot(global_eps, rewards,        alpha=0.25, color="steelblue", linewidth=0.8, label="raw")
ax1.plot(global_eps, smooth_rewards, color="steelblue", linewidth=2, label=f"{args.smooth}-ep mean")

# mark level advances
for ep, lvl in level_advances:
    ax1.axvline(ep, color=LEVEL_COLORS.get(lvl, "red"), linestyle="--", alpha=0.8, linewidth=1.5)
    ax1.text(ep + 0.5, ax1.get_ylim()[1] * 0.95 if ax1.get_ylim()[1] > 0 else 20,
             f"L{lvl}", fontsize=8, color=LEVEL_COLORS.get(lvl, "red"))

ax1.set_ylabel("Mean Episode Reward")
ax1.legend(loc="upper left", fontsize=8)
ax1.grid(True, alpha=0.3)

# ── Panel 2: Defender evasion rate ────────────────────────────────────────────
ax2 = axes[1]
ax2.plot(global_eps, evasion,        alpha=0.25, color="crimson", linewidth=0.8)
ax2.plot(global_eps, smooth_evasion, color="crimson", linewidth=2, label="evasion rate")
ax2.axhline(0.5, color="gray", linestyle=":", alpha=0.6, label="50% baseline")

if defender_start is not None:
    ax2.axvline(defender_start, color="black", linestyle="-.", alpha=0.5, linewidth=1.2)
    ax2.text(defender_start + 0.5, 0.95, "Defender ON", fontsize=8, color="black")

ax2.set_ylabel("Defender Evasion Rate")
ax2.set_ylim(0, 1.05)
ax2.legend(loc="upper right", fontsize=8)
ax2.grid(True, alpha=0.3)

# ── Panel 3: Per-level reward distribution ────────────────────────────────────
ax3 = axes[2]
lvl_data   = {}
for lvl, r in zip(levels, rewards):
    lvl_data.setdefault(lvl, []).append(r)

sorted_lvls = sorted(lvl_data.keys())
bp = ax3.boxplot(
    [lvl_data[l] for l in sorted_lvls],
    labels=[f"L{l}" for l in sorted_lvls],
    patch_artist=True,
    medianprops={"color": "black", "linewidth": 2},
)
for patch, lvl in zip(bp["boxes"], sorted_lvls):
    patch.set_facecolor(LEVEL_COLORS.get(lvl, "gray"))
    patch.set_alpha(0.7)

ax3.set_ylabel("Episode Reward Distribution")
ax3.set_xlabel("Curriculum Level")
ax3.grid(True, alpha=0.3, axis="y")

# Legend for level colors
patches = [mpatches.Patch(color=LEVEL_COLORS.get(l, "gray"), alpha=0.7, label=f"Level {l}")
           for l in sorted_lvls]
ax3.legend(handles=patches, loc="upper right", fontsize=8, ncol=min(4, len(sorted_lvls)))

plt.tight_layout()
plt.savefig(args.out, dpi=150, bbox_inches="tight")
print(f"\nPlot saved to {args.out}")
plt.show()
