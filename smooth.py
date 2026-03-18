# smooth.py
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/rl_metrics.csv")

def smooth(series, weight=0.95):
    smoothed = []
    last = series[0]
    for val in series:
        last = last * weight + (1 - weight) * val
        smoothed.append(last)
    return smoothed

df["reward_smooth"] = smooth(df["cumulative_reward"])
df["queue_smooth"] = smooth(df["queue_length"])

plt.figure(figsize=(10,6))
plt.plot(df["reward_smooth"], label="Smoothed Cumulative Reward")
plt.title("Smoothed RL Reward Curve")
plt.grid(True)
plt.legend()
plt.savefig("results/rl_reward_smoothed.png")

plt.figure(figsize=(10,6))
plt.plot(df["queue_smooth"], label="Smoothed Queue Length")
plt.title("Smoothed RL Queue Curve")
plt.grid(True)
plt.legend()
plt.savefig("results/rl_queue_smoothed.png")
