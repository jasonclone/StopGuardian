# significance_test.py
import pandas as pd
from scipy.stats import ttest_rel

rl = pd.read_csv("results/rl_metrics.csv")
base = pd.read_csv("results/fixed_time_metrics.csv")

# Align lengths
min_len = min(len(rl), len(base))
rl = rl.iloc[:min_len]
base = base.iloc[:min_len]

# Extract queues
rl_q = rl["queue_length"]
base_q = base["vehicle_queue"]

# Paired t-test
t_stat, p_value = ttest_rel(base_q, rl_q)

print("\n=== Paired t-test: Baseline vs RL Queue Lengths ===")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value:     {p_value:.6f}")

if p_value < 0.05:
    print("\nResult: RL is statistically significantly better than baseline (p < 0.05).")
else:
    print("\nResult: No statistically significant difference detected.")
