import pandas as pd
import matplotlib.pyplot as plt
import os

B1_EP = "results/b1/b1_episode_metrics.csv"
B1_STEP="results/b1/b1_step_metrics.csv"
RL_EP = "results/rl/rl_episode_metrics.csv"
RL_STEP = "results/rl/rl_step_metrics.csv"

# === CONFIG ===
EPISODE_PATH = B1_EP   # episode-level metrics
STEP_PATH    = B1_STEP     # step-level metrics (contains loss)

# === LOAD EPISODE CSV ===
df = pd.read_csv(EPISODE_PATH)
df.columns = df.columns.str.strip().str.lower()

# === LOAD STEP CSV (OPTIONAL) ===
has_step_loss = False
if os.path.exists(STEP_PATH):
    df_steps = pd.read_csv(STEP_PATH)
    df_steps.columns = df_steps.columns.str.strip().str.lower()
    if "loss" in df_steps.columns:
        has_step_loss = True
        train_steps = df_steps[df_steps["mode"] == "train"]

# === DETECT IF 'mode' EXISTS IN EPISODE CSV ===
if "mode" in df.columns:
    has_train = "train" in df["mode"].unique()
    has_eval  = "eval"  in df["mode"].unique()

    train = df[df["mode"] == "train"] if has_train else None
    evals = df[df["mode"] == "eval"]  if has_eval  else None

    if has_train and has_eval:
        title_prefix = "RL Agent"
    elif has_eval and not has_train:
        title_prefix = "RL Evaluation Only"
    else:
        title_prefix = "RL Metrics"

else:
    # BASELINE FILE (NO MODE COLUMN)
    train = None
    evals = df
    title_prefix = "Baseline"

# === SAFE PLOT ===
def safe_plot(col, idx, title, ylabel):
    plt.subplot(5, 2, idx)
    if train is not None and col in train.columns:
        plt.plot(train["episode"], train[col], label="Train")
    if evals is not None and col in evals.columns:
        plt.plot(evals["episode"], evals[col], label="Eval")
    plt.title(f"{title_prefix}: {title}")
    plt.xlabel("Episode")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)

# === PLOTS ===
plt.figure(figsize=(18, 26))

safe_plot("cumulative_reward",        1, "Cumulative Reward",          "Reward")
safe_plot("avg_vehicle_queue",        2, "Average Vehicle Queue",      "Queue")
safe_plot("avg_ped_queue",            3, "Average Pedestrian Queue",   "Queue")
safe_plot("avg_vehicle_wait",         4, "Average Vehicle Wait",       "Wait")
safe_plot("avg_ped_wait",             5, "Average Pedestrian Wait",    "Wait")
safe_plot("vehicle_total_throughput", 6, "Vehicle Throughput",         "Throughput")
safe_plot("ped_total_throughput",     7, "Pedestrian Throughput",      "Throughput")
safe_plot("switch_count",             8, "Switch Count",               "Switches")

# === TRAINING LOSS PLOT (ONLY IF STEP CSV EXISTS) ===
if has_step_loss:
    plt.subplot(5, 2, 9)
    plt.plot(train_steps["global_step"], train_steps["loss"], alpha=0.6)
    plt.title("Training Loss (C51 Cross-Entropy)")
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.grid(True)

plt.tight_layout()

outfile = os.path.basename(EPISODE_PATH).replace(".csv", "_curves_with_loss.png")
plt.savefig(outfile, dpi=200)
plt.show()

print(f"Saved: {outfile}")
