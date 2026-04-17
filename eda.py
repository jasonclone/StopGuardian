import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# ============================================================
# Paths
# ============================================================

RESULTS_DIR = "results"
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

RUNS = {
    "b1": "b1_step_metrics.csv",
    "rl": "rl_step_metrics.csv",
}

NUM_FEATURES = [
    "queue_N", "queue_E", "queue_S", "queue_W",
    "ped_queue",
]

CAT_FEATURES = ["phase"]


# ============================================================
# EDA Function (runs for baseline AND RL)
# ============================================================

def run_eda(run_id, csv_path):
    label = f"{run_id.upper()}"

    print(f"\n=== Running EDA for {label} ===")

    if not os.path.exists(csv_path):
        print(f"Skipping {label}: file not found → {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # ------------------------------
    # Detect correct step column
    # ------------------------------
    if "global_step" in df.columns:
        step_col = "global_step"
    elif "total_steps_global" in df.columns:
        step_col = "total_steps_global"
    elif "Step" in df.columns:
        step_col = "Step"
    else:
        raise KeyError(f"{label}: No valid step column found in CSV.")

    # ------------------------------
    # Count steps
    # ------------------------------
    num_steps = len(df)
    print(f"{label}: Total steps logged = {num_steps}")

    # Save step count to text file
    with open(os.path.join(EDA_DIR, f"{run_id}_step_count.txt"), "w") as f:
        f.write(f"{label} total steps: {num_steps}\n")

    # ------------------------------
    # 1. Distribution Plots
    # ------------------------------
    print(f"{label}: Distribution plots...")

    for feature in NUM_FEATURES:
        # Histogram
        plt.figure(figsize=(8, 5))
        sns.histplot(df[feature], kde=True)
        plt.title(f"{label} — Histogram of {feature} (steps={num_steps})")
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_hist_{feature}.png"))
        plt.close()

        # Boxplot
        plt.figure(figsize=(8, 5))
        sns.boxplot(x=df[feature])
        plt.title(f"{label} — Boxplot of {feature} (steps={num_steps})")
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_box_{feature}.png"))
        plt.close()

        # Violin
        plt.figure(figsize=(8, 5))
        sns.violinplot(x=df[feature])
        plt.title(f"{label} — Violin Plot of {feature} (steps={num_steps})")
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_violin_{feature}.png"))
        plt.close()

    # Categorical
    for feature in CAT_FEATURES:
        plt.figure(figsize=(10, 5))
        sns.countplot(x=df[feature])
        plt.title(f"{label} — Count Plot of {feature} (steps={num_steps})")
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_count_{feature}.png"))
        plt.close()

    # ------------------------------
    # 2. Correlation Heatmap
    # ------------------------------
    print(f"{label}: Correlation heatmap...")

    corr_cols = NUM_FEATURES + CAT_FEATURES + ["reward"]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm")
    plt.title(f"{label} — Correlation Heatmap (steps={num_steps})")
    plt.savefig(os.path.join(EDA_DIR, f"{run_id}_correlation_heatmap.png"))
    plt.close()

    # ------------------------------
    # 3. Time-Series Plots
    # ------------------------------
    print(f"{label}: Time-series plots...")

    SAMPLE_STEPS = min(10000, len(df))

    for feature in NUM_FEATURES + CAT_FEATURES:
        plt.figure(figsize=(10, 5))
        plt.plot(df[step_col][:SAMPLE_STEPS], df[feature][:SAMPLE_STEPS])
        plt.title(f"{label} — Time-Series: {feature} (steps={num_steps})")
        plt.xlabel(step_col)
        plt.ylabel(feature)
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_timeseries_{feature}.png"))
        plt.close()

    # ------------------------------
    # 4. PCA + t-SNE
    # ------------------------------
    print(f"{label}: PCA + t-SNE...")

    X = df[NUM_FEATURES].fillna(0)

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
    plt.title(f"{label} — PCA (2D) (steps={num_steps})")
    plt.savefig(os.path.join(EDA_DIR, f"{run_id}_pca_2d.png"))
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)
    tsne_result = tsne.fit_transform(X_sample)

    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5)
    plt.title(f"{label} — t-SNE (2D) (steps={num_steps})")
    plt.savefig(os.path.join(EDA_DIR, f"{run_id}_tsne_2d.png"))
    plt.close()

    # ------------------------------
    # 5. RL-Specific Plots
    # ------------------------------
    print(f"{label}: RL-specific plots...")

    # Reward distribution
    plt.figure(figsize=(8, 5))
    sns.histplot(df["reward"], kde=True)
    plt.title(f"{label} — Reward Distribution (steps={num_steps})")
    plt.savefig(os.path.join(EDA_DIR, f"{run_id}_reward_distribution.png"))
    plt.close()

    # Episode lengths
    if "step_in_episode" in df.columns:
        ep_lengths = df.groupby("episode")["step_in_episode"].max()
    elif "Step" in df.columns:
        ep_lengths = df.groupby("episode")["Step"].max()
    else:
        ep_lengths = None

    if ep_lengths is not None:
        plt.figure(figsize=(8, 5))
        plt.plot(ep_lengths.index, ep_lengths.values)
        plt.title(f"{label} — Episode Lengths (steps={num_steps})")
        plt.savefig(os.path.join(EDA_DIR, f"{run_id}_episode_lengths.png"))
        plt.close()

    # State-space coverage
    plt.figure(figsize=(8, 6))
    plt.scatter(df["queue_N"], df["queue_E"], s=5, alpha=0.3)
    plt.title(f"{label} — State-Space: queue_N vs queue_E (steps={num_steps})")
    plt.savefig(os.path.join(EDA_DIR, f"{run_id}_state_space_NE.png"))
    plt.close()

    print(f"{label}: EDA complete.")


# ============================================================
# Run EDA for Baseline and RL
# ============================================================

for run_id, filename in RUNS.items():
    csv_path = os.path.join(RESULTS_DIR, run_id, filename)
    run_eda(run_id, csv_path)

print("\nAll EDA plots saved to results/eda/")
