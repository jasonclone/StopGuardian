# eda.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

RESULTS_DIR = "results"
EDA_DIR = os.path.join(RESULTS_DIR, "eda")
os.makedirs(EDA_DIR, exist_ok=True)

RUN_ID = "b1"  # change if different
BASELINE_CSV = os.path.join(RESULTS_DIR, RUN_ID, "b1_step_metrics.csv")
RL_RUN_ID = "rl"
RL_CSV = os.path.join(RESULTS_DIR, RL_RUN_ID, "rl_step_metrics.csv")

# ============================================================
# Load Data
# ============================================================

print("Loading baseline data...")
df = pd.read_csv(BASELINE_CSV)

# Optional RL data
if os.path.exists(RL_CSV):
    print("Loading RL data...")
    df_rl = pd.read_csv(RL_CSV)
else:
    df_rl = None

# Numerical features for EDA
NUM_FEATURES = [
    "queue_N", "queue_E", "queue_S", "queue_W",
    "wait_N", "wait_E", "wait_S", "wait_W",
    "ped_queue", "ped_wait"
]

# ============================================================
# 2.2.1 Distribution Plots
# ============================================================

print("Generating distribution plots...")

for feature in NUM_FEATURES:
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f"Histogram of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"hist_{feature}.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[feature])
    plt.title(f"Boxplot of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"box_{feature}.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.violinplot(x=df[feature])
    plt.title(f"Violin Plot of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"violin_{feature}.png"))
    plt.close()

# ============================================================
# 2.2.2 Correlation Heatmap
# ============================================================

print("Generating correlation heatmap...")

plt.figure(figsize=(12, 10))
corr = df[NUM_FEATURES + ["reward"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap (Features + Reward)")
plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"))
plt.close()

# ============================================================
# 2.2.3 Sample Visualizations (Time-Series)
# ============================================================

print("Generating sample time-series plots...")

SAMPLE_STEPS = 500  # first 500 steps

for feature in NUM_FEATURES:
    plt.figure(figsize=(10, 5))
    plt.plot(df["total_steps_global"][:SAMPLE_STEPS], df[feature][:SAMPLE_STEPS])
    plt.title(f"Sample Time-Series: {feature}")
    plt.xlabel("Global Step")
    plt.ylabel(feature)
    plt.savefig(os.path.join(EDA_DIR, f"sample_timeseries_{feature}.png"))
    plt.close()

# ============================================================
# 2.2.4 Dimensionality Reduction (PCA + t-SNE)
# ============================================================

print("Generating PCA and t-SNE plots...")

X = df[NUM_FEATURES].fillna(0)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
plt.title("PCA (2D) of State Features")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(EDA_DIR, "pca_2d.png"))
plt.close()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_sample = X.sample(n=min(2000, len(X)), random_state=42)
tsne_result = tsne.fit_transform(X_sample)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5)
plt.title("t-SNE (2D) of State Features")
plt.savefig(os.path.join(EDA_DIR, "tsne_2d.png"))
plt.close()

# ============================================================
# 2.2.5 RL-Specific Plots
# ============================================================

print("Generating RL-specific plots...")

# Reward distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["reward"], kde=True)
plt.title("Reward Distribution (Baseline)")
plt.savefig(os.path.join(EDA_DIR, "reward_distribution.png"))
plt.close()

# Episode lengths
episode_lengths = df.groupby("episode")["Step"].max()

plt.figure(figsize=(8, 5))
plt.plot(episode_lengths.index, episode_lengths.values)
plt.title("Episode Lengths (Baseline)")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.savefig(os.path.join(EDA_DIR, "episode_lengths.png"))
plt.close()

# State-space coverage (queue_N vs queue_E)
plt.figure(figsize=(8, 6))
plt.scatter(df["queue_N"], df["queue_E"], s=5, alpha=0.3)
plt.title("State-Space Coverage: queue_N vs queue_E")
plt.xlabel("queue_N")
plt.ylabel("queue_E")
plt.savefig(os.path.join(EDA_DIR, "state_space_coverage.png"))
plt.close()

print("EDA complete. All plots saved to results/eda/")
