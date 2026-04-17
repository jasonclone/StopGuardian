# compare.py
import pandas as pd

# Load episode-level metrics
rl = pd.read_csv("results/rl/rl_episode_metrics.csv")
base = pd.read_csv("results/b1/b1_episode_metrics.csv")

# Align lengths
min_len = min(len(rl), len(base))
rl = rl.iloc[:min_len]
base = base.iloc[:min_len]

# Metrics to compare
avg_queue_rl = rl["avg_vehicle_queue"].mean()
avg_queue_base = base["avg_vehicle_queue"].mean()

total_queue_rl = rl["avg_total_queue"].mean()
total_queue_base = base["avg_total_queue"].mean()

cum_reward_rl = rl["cumulative_reward"].sum()
cum_reward_base = base["cumulative_reward"].sum()

# Percent improvement helper
def pct_improve(old, new):
    return 100 * (old - new) / old if old != 0 else float('nan')

print("\n=== RL vs Baseline Comparison ===")

print("\nAverage Vehicle Queue:")
print(f"  Baseline: {avg_queue_base:.2f}")
print(f"  RL:       {avg_queue_rl:.2f}")
print(f"  Improvement: {pct_improve(avg_queue_base, avg_queue_rl):.2f}%")

print("\nAverage Total Queue:")
print(f"  Baseline: {total_queue_base:.2f}")
print(f"  RL:       {total_queue_rl:.2f}")
print(f"  Improvement: {pct_improve(total_queue_base, total_queue_rl):.2f}%")

print("\nTotal Cumulative Reward:")
print(f"  Baseline: {cum_reward_base:.2f}")
print(f"  RL:       {cum_reward_rl:.2f}")
print(f"  Improvement: {pct_improve(abs(cum_reward_base), abs(cum_reward_rl)):.2f}%")

# Optional throughput comparison
if "vehicle_throughput" in base.columns:
    print("\nVehicle Throughput (last episode):")
    print(f"  Baseline: {base['vehicle_throughput'].iloc[-1]}")
    print(f"  RL:       {rl['vehicle_throughput'].iloc[-1]}")
