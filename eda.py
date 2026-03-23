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

RUN_ID = "b1"  # baseline run ID
BASELINE_CSV = os.path.join(RESULTS_DIR, RUN_ID, "b1_step_metrics.csv")

RL_RUN_ID = "rl"
RL_CSV = os.path.join(RESULTS_DIR, RL_RUN_ID, "rl_step_metrics.csv")

BASELINE_LABEL = f"Baseline ({RUN_ID})"
RL_LABEL = f"RL ({RL_RUN_ID})"

# ============================================================
# Load Data
# ============================================================

print("Loading baseline data...")
df = pd.read_csv(BASELINE_CSV)

df_rl = None
if os.path.exists(RL_CSV):
    print("Loading RL data...")
    df_rl = pd.read_csv(RL_CSV)

# ============================================================
# Feature Lists
# ============================================================

NUM_FEATURES = [
    "queue_N", "queue_E", "queue_S", "queue_W",
    "ped_queue",
]

CAT_FEATURES = ["phase"]  # categorical feature

# ============================================================
# 2.2.1 Distribution Plots (Numerical + Categorical)
# ============================================================

print("Generating distribution plots...")

# Numerical features
for feature in NUM_FEATURES:
    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(df[feature], kde=True)
    plt.title(f"{BASELINE_LABEL} — Histogram of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_hist_{feature}.png"))
    plt.close()

    # Boxplot
    plt.figure(figsize=(8, 5))
    sns.boxplot(x=df[feature])
    plt.title(f"{BASELINE_LABEL} — Boxplot of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_box_{feature}.png"))
    plt.close()

    # Violin plot
    plt.figure(figsize=(8, 5))
    sns.violinplot(x=df[feature])
    plt.title(f"{BASELINE_LABEL} — Violin Plot of {feature}")
    plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_violin_{feature}.png"))
    plt.close()

# Categorical features (phase)
print("Generating categorical distribution plots...")

for feature in CAT_FEATURES:
    plt.figure(figsize=(10, 5))
    sns.countplot(x=df[feature])
    plt.title(f"{BASELINE_LABEL} — Count Plot of {feature}")
    plt.xlabel(feature)
    plt.ylabel("Count")
    plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_count_{feature}.png"))
    plt.close()

# ============================================================
# 2.2.2 Correlation Heatmap
# ============================================================

print("Generating correlation heatmap...")

plt.figure(figsize=(12, 10))
corr = df[NUM_FEATURES + CAT_FEATURES + ["reward"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title(f"{BASELINE_LABEL} — Correlation Heatmap")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_correlation_heatmap.png"))
plt.close()

# ============================================================
# 2.2.3 Sample Visualizations (Time-Series)
# ============================================================

print("Generating sample time-series plots...")

SAMPLE_STEPS = 10000  # first 10k steps

for feature in NUM_FEATURES + CAT_FEATURES:
    plt.figure(figsize=(10, 5))
    plt.plot(df["total_steps_global"][:SAMPLE_STEPS], df[feature][:SAMPLE_STEPS])
    plt.title(f"{BASELINE_LABEL} — Time-Series: {feature}")
    plt.xlabel("Global Step")
    plt.ylabel(feature)
    plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_timeseries_{feature}.png"))
    plt.close()

# ============================================================
# 2.2.4 Dimensionality Reduction (PCA + t-SNE)
# ============================================================

print("Generating PCA and t-SNE plots...")

# PCA and t-SNE only on numerical features (phase excluded)
X = df[NUM_FEATURES].fillna(0)

# PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

plt.figure(figsize=(8, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
plt.title(f"{BASELINE_LABEL} — PCA (2D)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_pca_2d.png"))
plt.close()

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
X_sample = X.sample(n=min(2000, len(X)), random_state=42)
tsne_result = tsne.fit_transform(X_sample)

plt.figure(figsize=(8, 6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5)
plt.title(f"{BASELINE_LABEL} — t-SNE (2D)")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_tsne_2d.png"))
plt.close()

# ============================================================
# 2.2.5 RL-Specific Plots
# ============================================================

print("Generating RL-specific plots...")

# Reward distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["reward"], kde=True)
plt.title(f"{BASELINE_LABEL} — Reward Distribution")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_reward_distribution.png"))
plt.close()

# Episode lengths
episode_lengths = df.groupby("episode")["Step"].max()

plt.figure(figsize=(8, 5))
plt.plot(episode_lengths.index, episode_lengths.values)
plt.title(f"{BASELINE_LABEL} — Episode Lengths")
plt.xlabel("Episode")
plt.ylabel("Steps")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_episode_lengths.png"))
plt.close()

# State-space coverage
plt.figure(figsize=(8, 6))
plt.scatter(df["queue_N"], df["queue_E"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: queue_N vs queue_E")
plt.xlabel("queue_N")
plt.ylabel("queue_E")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage.png"))
plt.close()

print("Generating additional state-space coverage plot examples...")

# 1. queue_S vs queue_W
plt.figure(figsize=(8, 6))
plt.scatter(df["queue_S"], df["queue_W"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: queue_S vs queue_W")
plt.xlabel("queue_S")
plt.ylabel("queue_W")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage_SW.png"))
plt.close()

# 2. ped_queue vs queue_N
plt.figure(figsize=(8, 6))
plt.scatter(df["ped_queue"], df["queue_N"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: ped_queue vs queue_N")
plt.xlabel("ped_queue")
plt.ylabel("queue_N")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage_ped_vs_qN.png"))
plt.close()

# 3. ped_queue vs queue_E
plt.figure(figsize=(8, 6))
plt.scatter(df["ped_queue"], df["queue_E"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: ped_queue vs queue_E")
plt.xlabel("ped_queue")
plt.ylabel("queue_E")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage_ped_vs_qE.png"))
plt.close()

# 4. phase vs queue_N
plt.figure(figsize=(8, 6))
plt.scatter(df["phase"], df["queue_N"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: phase vs queue_N")
plt.xlabel("phase")
plt.ylabel("queue_N")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage_phase_vs_qN.png"))
plt.close()

# 5. phase vs ped_queue
plt.figure(figsize=(8, 6))
plt.scatter(df["phase"], df["ped_queue"], s=5, alpha=0.3)
plt.title(f"{BASELINE_LABEL} — State-Space Coverage: phase vs ped_queue")
plt.xlabel("phase")
plt.ylabel("ped_queue")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_state_space_coverage_phase_vs_ped.png"))
plt.close()




# Phase over time (RL-specific)
plt.figure(figsize=(10, 5))
plt.plot(df["total_steps_global"], df["phase"])
plt.title(f"{BASELINE_LABEL} — Phase Over Time")
plt.xlabel("Global Step")
plt.ylabel("Phase")
plt.savefig(os.path.join(EDA_DIR, f"{RUN_ID}_phase_over_time.png"))
plt.close()

print("EDA complete. All plots saved to results/eda/")
