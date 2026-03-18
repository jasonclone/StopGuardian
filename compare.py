#compare.py
import pandas as pd

rl = pd.read_csv("results/rl_metrics.csv")
base = pd.read_csv("results/fixed_time_metrics.csv")

# Align lengths if needed
min_len = min(len(rl), len(base))
rl = rl.iloc[:min_len]
base = base.iloc[:min_len]

# Compute metrics
avg_queue_rl = rl["queue_length"].mean()
avg_queue_base = base["vehicle_queue"].mean()

max_queue_rl = rl["queue_length"].max()
max_queue_base = base["vehicle_queue"].max()

cum_reward_rl = rl["cumulative_reward"].iloc[-1]
cum_reward_base = base["cumulative_reward"].iloc[-1]

# Percent improvements
def pct_improve(old, new):
    return 100 * (old - new) / old

avg_queue_improve = pct_improve(avg_queue_base, avg_queue_rl)
max_queue_improve = pct_improve(max_queue_base, max_queue_rl)
reward_improve = pct_improve(abs(cum_reward_base), abs(cum_reward_rl))

print("\n=== RL vs Baseline Comparison ===")
print(f"Average Queue Length:")
print(f"  Baseline: {avg_queue_base:.2f}")
print(f"  RL:       {avg_queue_rl:.2f}")
print(f"  Improvement: {avg_queue_improve:.2f}%")

print(f"\nMax Queue Length:")
print(f"  Baseline: {max_queue_base:.2f}")
print(f"  RL:       {max_queue_rl:.2f}")
print(f"  Improvement: {max_queue_improve:.2f}%")

print(f"\nCumulative Reward:")
print(f"  Baseline: {cum_reward_base:.2f}")
print(f"  RL:       {cum_reward_rl:.2f}")
print(f"  Improvement: {reward_improve:.2f}%")

# Optional: throughput comparison
if "vehicle_throughput" in base.columns:
    print("\nVehicle Throughput:")
    print(f"  Baseline: {base['vehicle_throughput'].iloc[-1]}")
    print(f"  RL:       (not logged in RL)")
