#!/usr/bin/env python3
# eda.py
"""
EDA for RL / baseline results (reads CSVs produced by train.py).

This script:
- Discovers results/<run_id>/<model>/seed_<N>/... and legacy layouts.
- Loads per-seed CSVs (rl_episode_metrics.csv, rl_step_metrics.csv) and aggregated CSVs.
- Produces per-seed and aggregated learning curves (mean + 95% CI).
- Produces model-comparison visuals: bar charts with CI, radar profiles, overlays, scatter perf vs cost.
- Best-effort policy visualizations: trajectory plots, approximate value heatmaps, short episode videos (if torch, model builders, TrafficEnv, imageio available and checkpoints present).
- Writes logs to results/eda/<run_id>/eda_errors.log and a JSON summary.
"""

import os
import imageio
import json
import argparse
import traceback
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy import stats

sns.set(style="whitegrid", context="talk")

# Optional dependencies for policy visualization (best-effort)
try:
    import imageio
except Exception:
    imageio = None

try:
    import torch
except Exception:
    torch = None

# Try to import model builders if available (best-effort)
try:
    from rl.model import build_rainbow_model, build_standard_model
except Exception:
    build_rainbow_model = None
    build_standard_model = None

# -----------------------
# Defaults / Features
# -----------------------
DEFAULT_RESULTS_DIR = "results"
EDA_DIRNAME = "eda"

NUM_FEATURES = [
    "vehicle_queue", "queue_n", "queue_e", "queue_s", "queue_w",
    "ped_queue", "vehicle_wait", "ped_wait", "veh_thru_step", "ped_thru_step",
]

CAT_FEATURES = ["phase", "mode"]

STATE_HEATMAP_PAIRS = [("queue_n", "queue_e"), ("vehicle_queue", "ped_queue")]

POSITION_COL_PAIRS = [
    ("x", "y"),
    ("pos_x", "pos_y"),
    ("veh_x", "veh_y"),
    ("vehicle_x", "vehicle_y"),
    ("px", "py"),
]

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def log_exception(out_dir: str, context: str, exc: Exception):
    try:
        ensure_dir(out_dir)
        log_path = os.path.join(out_dir, "eda_errors.log")
        with open(log_path, "a") as f:
            f.write(f"\n--- Exception in {context} ---\n")
            traceback.print_exc(file=f)
            f.write("\n")
    except Exception:
        print("[EDA][WARN] Failed to write exception log")

def detect_step_col(df: pd.DataFrame) -> str:
    for c in ("global_step", "total_steps_global", "step"):
        if c in df.columns:
            return c
    raise KeyError("No valid step column found in CSV (expected one of global_step, total_steps_global, step)")

def mean_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    a = np.array(data, dtype=float)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    m = np.mean(a)
    if n == 1:
        return float(m), float(m), float(m)
    se = stats.sem(a)
    h = se * stats.t.ppf((1 + confidence) / 2., n - 1)
    return float(m), float(m - h), float(m + h)

def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    return np.convolve(x, np.ones(window) / window, mode="valid")

def _normalize_columns_lower(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df

# -----------------------
# Plot utilities
# -----------------------
def save_fig(fig, path: str):
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_distribution(df: pd.DataFrame, feature: str, out_dir: str, label: str):
    try:
        if feature not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df[feature].dropna(), kde=True, ax=ax)
        ax.set_title(f"{label} — Histogram of {feature}")
        save_fig(fig, os.path.join(out_dir, f"{label}_hist_{feature}.png"))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(x=df[feature].dropna(), ax=ax)
        ax.set_title(f"{label} — Boxplot of {feature}")
        save_fig(fig, os.path.join(out_dir, f"{label}_box_{feature}.png"))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.violinplot(x=df[feature].dropna(), ax=ax)
        ax.set_title(f"{label} — Violin of {feature}")
        save_fig(fig, os.path.join(out_dir, f"{label}_violin_{feature}.png"))
    except Exception as e:
        log_exception(out_dir, f"plot_distribution({feature})", e)

def plot_correlation(df: pd.DataFrame, cols: List[str], out_dir: str, label: str):
    try:
        present = [c for c in cols if c in df.columns]
        if not present:
            return
        numeric_cols = [c for c in present if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) < 2:
            return
        corr = df[numeric_cols].corr()
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        ax.set_title(f"{label} — Correlation Heatmap (numeric features only)")
        save_fig(fig, os.path.join(out_dir, f"{label}_correlation_heatmap.png"))
    except Exception as e:
        log_exception(out_dir, "plot_correlation", e)

def plot_time_series(df: pd.DataFrame, step_col: str, features: List[str], out_dir: str, label: str, sample_steps: int = 10000):
    try:
        sample_steps = min(sample_steps, len(df))
        for feature in features:
            if feature not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(10, 4.5))
            ax.plot(df[step_col].values[:sample_steps], df[feature].values[:sample_steps], lw=0.8)
            ax.set_title(f"{label} — Time-Series: {feature}")
            ax.set_xlabel(step_col)
            ax.set_ylabel(feature)
            save_fig(fig, os.path.join(out_dir, f"{label}_timeseries_{feature}.png"))
    except Exception as e:
        log_exception(out_dir, "plot_time_series", e)

def plot_pca_tsne(df: pd.DataFrame, features: List[str], out_dir: str, label: str):
    try:
        X = df[features].fillna(0).values
        if X.shape[0] < 2:
            return
        try:
            pca = PCA(n_components=2)
            pca_result = pca.fit_transform(X)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(pca_result[:, 0], pca_result[:, 1], s=5, alpha=0.5)
            ax.set_title(f"{label} — PCA (2D)")
            save_fig(fig, os.path.join(out_dir, f"{label}_pca_2d.png"))
        except Exception:
            pass
        try:
            n_samples = min(2000, X.shape[0])
            idx = np.random.choice(X.shape[0], n_samples, replace=False)
            tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, init="pca", random_state=42)
            tsne_result = tsne.fit_transform(X[idx])
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(tsne_result[:, 0], tsne_result[:, 1], s=5, alpha=0.5)
            ax.set_title(f"{label} — t-SNE (2D) sample={n_samples}")
            save_fig(fig, os.path.join(out_dir, f"{label}_tsne_2d.png"))
        except Exception:
            pass
    except Exception as e:
        log_exception(out_dir, "plot_pca_tsne", e)

def plot_reward_distribution(df: pd.DataFrame, out_dir: str, label: str):
    try:
        if "reward" not in df.columns:
            return
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(df["reward"].dropna(), kde=True, ax=ax)
        ax.set_title(f"{label} — Reward Distribution")
        save_fig(fig, os.path.join(out_dir, f"{label}_reward_distribution.png"))
    except Exception as e:
        log_exception(out_dir, "plot_reward_distribution", e)

def plot_episode_lengths(step_df: pd.DataFrame, out_dir: str, label: str):
    try:
        if "episode" in step_df.columns and "step_in_episode" in step_df.columns:
            ep_lengths = step_df.groupby("episode")["step_in_episode"].max().values + 1
            fig, ax = plt.subplots(figsize=(8, 4.5))
            ax.plot(np.arange(len(ep_lengths)), ep_lengths, marker="o", lw=1)
            ax.set_title(f"{label} — Episode Lengths")
            ax.set_xlabel("episode")
            ax.set_ylabel("length (steps)")
            save_fig(fig, os.path.join(out_dir, f"{label}_episode_lengths.png"))
    except Exception as e:
        log_exception(out_dir, "plot_episode_lengths", e)

# -----------------------
# Learning curve & aggregation
# -----------------------
def aggregate_episode_rewards_across_seeds(episode_rows_by_seed: List[pd.DataFrame]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        max_ep = max([len(df) for df in episode_rows_by_seed]) if episode_rows_by_seed else 0
        rewards_by_ep = []
        for ep in range(max_ep):
            vals = []
            for df in episode_rows_by_seed:
                if ep < len(df):
                    try:
                        vals.append(float(df.iloc[ep]["cumulative_reward"]))
                    except Exception:
                        pass
            rewards_by_ep.append(vals)
        means = []
        lcis = []
        ucis = []
        for vals in rewards_by_ep:
            if len(vals) == 0:
                means.append(np.nan)
                lcis.append(np.nan)
                ucis.append(np.nan)
            else:
                m, l, u = mean_confidence_interval(vals, confidence=0.95)
                means.append(m)
                lcis.append(l)
                ucis.append(u)
        return np.array(means), np.array(lcis), np.array(ucis)
    except Exception as e:
        log_exception(".", "aggregate_episode_rewards_across_seeds", e)
        return np.array([]), np.array([]), np.array([])

def plot_learning_curve(episode_rows_by_seed: List[pd.DataFrame], out_dir: str, label: str, smooth_window: int = 1):
    try:
        means, lcis, ucis = aggregate_episode_rewards_across_seeds(episode_rows_by_seed)
        episodes = np.arange(len(means))
        if smooth_window > 1:
            ma = moving_average(np.nan_to_num(means, nan=np.nan), smooth_window)
            ma_x = np.arange(len(ma)) + (smooth_window - 1) // 2
        else:
            ma = means
            ma_x = episodes
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(episodes, lcis, ucis, color="C0", alpha=0.2, label="95% CI")
        ax.plot(episodes, means, color="C0", lw=1, alpha=0.6, label="Mean (per-episode)")
        if smooth_window > 1:
            ax.plot(ma_x, ma, color="C1", lw=2, label=f"MA (window={smooth_window})")
        ax.set_title(f"{label} — Learning Curve (mean cumulative reward across seeds)")
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative reward")
        ax.legend()
        save_fig(fig, os.path.join(out_dir, f"{label}_learning_curve.png"))
        fig, ax = plt.subplots(figsize=(10, 5))
        for i, df in enumerate(episode_rows_by_seed):
            if "cumulative_reward" in df.columns:
                ax.plot(np.arange(len(df)), df["cumulative_reward"].values, lw=1, alpha=0.6, label=f"seed_{i}")
        ax.set_title(f"{label} — Per-seed Episode Rewards")
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative reward")
        ax.legend(ncol=2, fontsize="small")
        save_fig(fig, os.path.join(out_dir, f"{label}_per_seed_rewards.png"))
    except Exception as e:
        log_exception(out_dir, "plot_learning_curve", e)

def plot_learning_curve_vs_steps(episode_rows_by_seed: List[pd.DataFrame], out_dir: str, label: str, smooth_window: int = 1):
    try:
        per_seed_series = []
        for df in episode_rows_by_seed:
            if "episode_start_step" not in df.columns or "cumulative_reward" not in df.columns:
                continue
            s = df[["episode_start_step", "cumulative_reward"]].dropna().sort_values("episode_start_step")
            if s.empty:
                continue
            per_seed_series.append(s.set_index("episode_start_step")["cumulative_reward"])
        if not per_seed_series:
            return
        all_steps = sorted(set().union(*[s.index.tolist() for s in per_seed_series]))
        aligned = []
        for s in per_seed_series:
            try:
                r = s.reindex(all_steps).interpolate().ffill().bfill()
            except Exception:
                r = s.reindex(all_steps).ffill().bfill()
            aligned.append(r)
        df_aligned = pd.DataFrame(aligned).T
        df_aligned.columns = [f"seed_{i}" for i in range(df_aligned.shape[1])]
        mean_series = df_aligned.mean(axis=1)
        std_series = df_aligned.std(axis=1)
        if smooth_window > 1:
            ma = mean_series.rolling(window=smooth_window, min_periods=1, center=True).mean()
        else:
            ma = mean_series
        fig, ax = plt.subplots(figsize=(12, 6))
        for col in df_aligned.columns:
            ax.plot(df_aligned.index.values, df_aligned[col].values, color="gray", alpha=0.25, lw=0.8)
        ax.fill_between(df_aligned.index, mean_series - std_series, mean_series + std_series, color="C0", alpha=0.2, label="±1 std")
        ax.plot(df_aligned.index, mean_series, color="C0", lw=1.5, alpha=0.9, label="Mean")
        if smooth_window > 1:
            ax.plot(df_aligned.index, ma, color="C1", lw=2.2, label=f"MA (window={smooth_window})")
        ax.set_xlabel("global training step (episode_start_step)")
        ax.set_ylabel("episodic cumulative reward")
        ax.set_title(f"{label} — Learning curve vs training steps")
        ax.legend()
        save_fig(fig, os.path.join(out_dir, f"{label}_learning_curve_vs_steps.png"))
        try:
            df_aligned.to_csv(os.path.join(out_dir, f"{label}_aligned_learning_curve.csv"), index_label="episode_start_step")
        except Exception:
            pass
    except Exception as e:
        log_exception(out_dir, "plot_learning_curve_vs_steps", e)

# -----------------------
# Sample-efficiency reading (train.py writes these)
# -----------------------
def read_sample_efficiency_summary(model_dir: str) -> Dict[int, Optional[int]]:
    results = {}
    try:
        model_summary = os.path.join(model_dir, "sample_efficiency_summary.csv")
        if os.path.exists(model_summary):
            try:
                df = pd.read_csv(model_summary, header=None)
                for _, row in df.iterrows():
                    if len(row.dropna()) >= 2:
                        try:
                            seed = int(row[0])
                            step = row[1]
                            results[seed] = (int(step) if not pd.isna(step) else None)
                        except Exception:
                            continue
            except Exception:
                pass
        if os.path.exists(model_dir):
            for d in os.listdir(model_dir):
                if not d.startswith("seed_"):
                    continue
                seed_dir = os.path.join(model_dir, d)
                p = os.path.join(seed_dir, "sample_efficiency_seed.csv")
                if os.path.exists(p):
                    try:
                        sdf = pd.read_csv(p)
                        if "seed" in sdf.columns and "sample_efficiency_step" in sdf.columns:
                            for _, r in sdf.iterrows():
                                try:
                                    s = int(r["seed"])
                                    st = r["sample_efficiency_step"]
                                    results[s] = (int(st) if not pd.isna(st) else None)
                                except Exception:
                                    pass
                    except Exception:
                        pass
    except Exception as e:
        log_exception(model_dir, "read_sample_efficiency_summary", e)
    return results

# -----------------------
# State-visitation / heatmaps
# -----------------------
def plot_state_visitation(df: pd.DataFrame, out_dir: str, label: str, pairs: List[Tuple[str, str]]):
    for (xcol, ycol) in pairs:
        try:
            if xcol not in df.columns or ycol not in df.columns:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(df[xcol].values, df[ycol].values, s=4, alpha=0.3)
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(f"{label} — State Visitation: {xcol} vs {ycol}")
            save_fig(fig, os.path.join(out_dir, f"{label}_state_vis_{xcol}_{ycol}.png"))
            heatmap, xedges, yedges = np.histogram2d(df[xcol].values, df[ycol].values, bins=50)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(heatmap.T, cmap="magma", ax=ax, cbar=True)
            ax.set_title(f"{label} — State Occupancy Heatmap: {xcol} vs {ycol}")
            save_fig(fig, os.path.join(out_dir, f"{label}_state_heatmap_{xcol}_{ycol}.png"))
        except Exception as e:
            log_exception(out_dir, f"plot_state_visitation_{xcol}_{ycol}", e)

# -----------------------
# Policy visualization helpers (best-effort)
# -----------------------
def _find_checkpoint_in_seed(seed_dir: str) -> Optional[str]:
    try:
        ckpt_dir = os.path.join(seed_dir, "checkpoints")
        if not os.path.exists(ckpt_dir):
            return None
        for name in ("best_model.pth", "last_model.pth"):
            p = os.path.join(ckpt_dir, name)
            if os.path.exists(p):
                return p
        return None
    except Exception:
        return None

def record_episode_video_from_model(seed_dir: str, model_name: str, run_id: str, out_dir: str, max_steps: int = 500, fps: int = 15):
    try:
        if torch is None or (build_rainbow_model is None and build_standard_model is None) or imageio is None:
            return False, "Missing dependencies (torch/model builders/imageio)"
        ckpt = _find_checkpoint_in_seed(seed_dir)
        if ckpt is None:
            return False, "No checkpoint found"
        model_type = None
        if "rainbow" in seed_dir.lower():
            model_type = "rainbow"
        elif "standard" in seed_dir.lower() or "dqn" in seed_dir.lower():
            model_type = "standard"
        else:
            if "rainbow" in run_id.lower():
                model_type = "rainbow"
            elif "standard" in run_id.lower():
                model_type = "standard"
        try:
            from traffic_env import TrafficEnv
        except Exception as e:
            return False, f"TrafficEnv import failed: {e}"
        env = TrafficEnv("C")
        try:
            sumocfg = env.make_sumo_config() if hasattr(env, "make_sumo_config") else None
        except Exception:
            sumocfg = None
        try:
            if sumocfg is not None:
                env.start(sumocfg)
                env.step()
                env.initialize_from_sumo()
            else:
                try:
                    env.start()
                    env.step()
                    env.initialize_from_sumo()
                except Exception:
                    pass
        except Exception:
            pass
        state_size = None
        try:
            state_size = env.compute_state_size()
        except Exception:
            pass
        model = None
        optimizer = None
        try:
            if model_type == "rainbow" and build_rainbow_model is not None:
                model, optimizer = build_rainbow_model(state_size=state_size, lr_decay_steps=1)
            elif model_type == "standard" and build_standard_model is not None:
                model, optimizer = build_standard_model(state_size=state_size, lr_decay_steps=None)
            else:
                if build_rainbow_model is not None:
                    model, optimizer = build_rainbow_model(state_size=state_size, lr_decay_steps=1)
                    model_type = "rainbow"
                elif build_standard_model is not None:
                    model, optimizer = build_standard_model(state_size=state_size, lr_decay_steps=None)
                    model_type = "standard"
        except Exception as e:
            env.close()
            return False, f"Model builder failed: {e}"
        if model is None:
            env.close()
            return False, "No model builder available"
        try:
            ckpt_data = torch.load(ckpt, map_location="cpu")
            if "model" in ckpt_data:
                model.load_state_dict(ckpt_data["model"])
            else:
                model.load_state_dict(ckpt_data)
            model.eval()
        except Exception as e:
            try:
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
            except Exception as e2:
                env.close()
                return False, f"Failed loading checkpoint: {e}; {e2}"
        frames = []
        try:
            try:
                if sumocfg is not None:
                    env.start(sumocfg)
                    env.step()
                    env.initialize_from_sumo()
                else:
                    env.start()
                    env.step()
                    env.initialize_from_sumo()
            except Exception:
                pass
            env.reset_tracking()
            total_steps = 0
            for t in range(max_steps):
                state_raw, _ = env.get_state()
                try:
                    state_t = env.normalize_state_torch(state_raw) if hasattr(env, "normalize_state_torch") else None
                except Exception:
                    state_t = None
                try:
                    if state_t is None:
                        s = np.array(state_raw, dtype=np.float32)
                        state_t = torch.from_numpy(s).unsqueeze(0)
                    with torch.no_grad():
                        if hasattr(model, "q_values"):
                            q = model.q_values(state_t.to(next(model.parameters()).device))
                            action = int(torch.argmax(q, dim=-1).item())
                        else:
                            out = model(state_t.to(next(model.parameters()).device))
                            action = int(torch.argmax(out, dim=-1).item())
                except Exception:
                    try:
                        action = int(np.random.randint(0, env.get_num_actions()))
                    except Exception:
                        action = 0
                try:
                    from rl.action import apply_action_safe
                    apply_action_safe(action, env, current_step_global=total_steps)
                except Exception:
                    try:
                        env.step_action(action)
                    except Exception:
                        try:
                            env.step()
                        except Exception:
                            pass
                try:
                    if hasattr(env, "render_rgb"):
                        img = env.render_rgb()
                        frames.append(img)
                    elif hasattr(env, "render"):
                        img = env.render(mode="rgb_array")
                        frames.append(img)
                    else:
                        state_plot = env.get_state()[0] if hasattr(env, "get_state") else None
                        fig = plt.figure(figsize=(6, 4))
                        ax = fig.add_subplot(111)
                        if isinstance(state_plot, (list, np.ndarray)):
                            ax.plot(np.array(state_plot).flatten()[:50])
                        ax.set_title(f"step {t}")
                        fig.canvas.draw()
                        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                        frames.append(img)
                        plt.close(fig)
                except Exception:
                    pass
                total_steps += 1
            if frames:
                out_path = os.path.join(out_dir, f"{os.path.basename(seed_dir)}_{model_type}_episode.mp4")
                try:
                    imageio.mimwrite(out_path, frames, fps=fps, macro_block_size=None)
                    env.close()
                    return True, out_path
                except Exception as e:
                    env.close()
                    return False, f"Failed writing video: {e}"
            else:
                env.close()
                return False, "No frames captured"
        except Exception as e:
            try:
                env.close()
            except Exception:
                pass
            return False, f"Episode run failed: {e}"
    except Exception as e:
        return False, f"record_episode_video_from_model error: {e}"

def plot_value_function_heatmap_from_model(seed_dir: str, model_name: str, run_id: str, out_dir: str, step_rows: List[pd.DataFrame], grid_size: int = 50):
    try:
        if torch is None or (build_rainbow_model is None and build_standard_model is None):
            return False, "Missing torch or model builders"
        ckpt = _find_checkpoint_in_seed(seed_dir)
        if ckpt is None:
            return False, "No checkpoint found"
        if not step_rows:
            return False, "No step rows available to sample states"
        combined = pd.concat(step_rows, ignore_index=True)
        numeric_cols = [c for c in combined.columns if pd.api.types.is_numeric_dtype(combined[c])]
        exclude = {"global_step", "episode", "step_in_episode", "reward", "loss"}
        numeric_cols = [c for c in numeric_cols if c not in exclude]
        if len(numeric_cols) < 2:
            return False, "Not enough numeric state columns to build 2D grid"
        xcol, ycol = numeric_cols[0], numeric_cols[1]
        xmin, xmax = combined[xcol].min(), combined[xcol].max()
        ymin, ymax = combined[ycol].min(), combined[ycol].max()
        if pd.isna(xmin) or pd.isna(xmax) or pd.isna(ymin) or pd.isna(ymax):
            return False, "Invalid numeric ranges"
        xs = np.linspace(xmin, xmax, grid_size)
        ys = np.linspace(ymin, ymax, grid_size)
        grid = np.array([[xi, yi] for yi in ys for xi in xs], dtype=np.float32)
        model = None
        try:
            if build_rainbow_model is not None:
                state_size = grid.shape[1]
                model, _ = build_rainbow_model(state_size=state_size, lr_decay_steps=1)
            elif build_standard_model is not None:
                state_size = grid.shape[1]
                model, _ = build_standard_model(state_size=state_size, lr_decay_steps=None)
        except Exception:
            model = None
        if model is None:
            return False, "No model builder available"
        try:
            ckpt_data = torch.load(ckpt, map_location="cpu")
            if "model" in ckpt_data:
                model.load_state_dict(ckpt_data["model"])
            else:
                model.load_state_dict(ckpt_data)
            model.eval()
        except Exception as e:
            try:
                model.load_state_dict(torch.load(ckpt, map_location="cpu"))
                model.eval()
            except Exception as e2:
                return False, f"Failed loading checkpoint: {e}; {e2}"
        try:
            grid_t = torch.from_numpy(grid).float()
            try:
                sample_in = next(model.parameters()).shape[1]
            except Exception:
                sample_in = grid_t.shape[1]
            if grid_t.shape[1] != sample_in:
                if grid_t.shape[1] < sample_in:
                    pad = torch.zeros((grid_t.shape[0], sample_in - grid_t.shape[1]), dtype=grid_t.dtype)
                    grid_t = torch.cat([grid_t, pad], dim=1)
                else:
                    grid_t = grid_t[:, :sample_in]
            with torch.no_grad():
                if hasattr(model, "q_values"):
                    qvals = model.q_values(grid_t)
                    if qvals.ndim == 3:
                        qvals = qvals.mean(dim=-1)
                    values = qvals.max(dim=-1)[0].cpu().numpy()
                else:
                    out = model(grid_t)
                    values = out.max(dim=-1)[0].cpu().numpy()
        except Exception as e:
            return False, f"Model inference failed: {e}"
        try:
            Z = values.reshape((grid_size, grid_size))
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(Z, origin="lower", extent=(xmin, xmax, ymin, ymax), aspect="auto", cmap="viridis")
            ax.set_xlabel(xcol)
            ax.set_ylabel(ycol)
            ax.set_title(f"{run_id} {model_name} — Learned value function (approx)")
            fig.colorbar(im, ax=ax, label="value")
            out_path = os.path.join(out_dir, f"{os.path.basename(seed_dir)}_{model_name}_value_heatmap.png")
            save_fig(fig, out_path)
            return True, out_path
        except Exception as e:
            return False, f"Heatmap plotting failed: {e}"
    except Exception as e:
        return False, f"plot_value_function_heatmap_from_model error: {e}"

def plot_trajectory_if_positions(step_df: pd.DataFrame, out_dir: str, label: str):
    try:
        if step_df is None or step_df.empty:
            return False, "No step_df"
        cols = [c.lower() for c in step_df.columns]
        found = None
        for (a, b) in POSITION_COL_PAIRS:
            if a in cols and b in cols:
                found = (a, b)
                break
        if not found:
            numeric_cols = [c for c in cols if pd.api.types.is_numeric_dtype(step_df[c])]
            if len(numeric_cols) >= 2:
                found = (numeric_cols[0], numeric_cols[1])
            else:
                return False, "No positional columns found"
        xcol, ycol = found
        fig, ax = plt.subplots(figsize=(8, 6))
        if "episode" in step_df.columns:
            for ep, g in step_df.groupby("episode"):
                ax.plot(g[xcol].values, g[ycol].values, lw=1, alpha=0.6, label=f"ep{int(ep)}")
        else:
            ax.plot(step_df[xcol].values, step_df[ycol].values, lw=1)
        ax.set_xlabel(xcol)
        ax.set_ylabel(ycol)
        ax.set_title(f"{label} — Trajectory plot ({xcol},{ycol})")
        if "episode" in step_df.columns:
            ax.legend(fontsize="small", ncol=2)
        out_path = os.path.join(out_dir, f"{label}_trajectory_{xcol}_{ycol}.png")
        save_fig(fig, out_path)
        return True, out_path
    except Exception as e:
        return False, f"plot_trajectory_if_positions error: {e}"

# -----------------------
# Curves with loss (per-seed and aggregated)
# -----------------------
def plot_rl_curves_with_loss_for_seed(ep_df: pd.DataFrame, step_df: Optional[pd.DataFrame], out_path: str, title_prefix: str):
    try:
        df = _normalize_columns_lower(ep_df)
        train = None
        evals = None
        if "mode" in df.columns:
            modes = df["mode"].astype(str).unique().tolist()
            if "train" in modes:
                train = df[df["mode"] == "train"]
            if "eval" in modes:
                evals = df[df["mode"] == "eval"]
        else:
            evals = df
        has_step_loss = False
        train_steps = None
        if step_df is not None:
            sdf = _normalize_columns_lower(step_df)
            if "loss" in sdf.columns:
                has_step_loss = True
                if "mode" in sdf.columns:
                    train_steps = sdf[sdf["mode"] == "train"]
                else:
                    train_steps = sdf
        def safe_plot(col, ax, title, ylabel):
            if train is not None and col in train.columns:
                ax.plot(train["episode"].values, train[col].values, label="Train")
            if evals is not None and col in evals.columns:
                ax.plot(evals["episode"].values, evals[col].values, label="Eval")
            ax.set_title(f"{title_prefix}: {title}")
            ax.set_xlabel("Episode")
            ax.set_ylabel(ylabel)
            ax.legend()
            ax.grid(True)
        fig = plt.figure(figsize=(18, 26))
        axes = [fig.add_subplot(5, 2, i) for i in range(1, 10)]
        metrics = [
            ("cumulative_reward", "Cumulative Reward", "Reward"),
            ("avg_vehicle_queue", "Average Vehicle Queue", "Queue"),
            ("avg_ped_queue", "Average Pedestrian Queue", "Queue"),
            ("avg_vehicle_wait", "Average Vehicle Wait", "Wait"),
            ("avg_ped_wait", "Average Pedestrian Wait", "Wait"),
            ("vehicle_total_throughput", "Vehicle Throughput", "Throughput"),
            ("ped_total_throughput", "Pedestrian Throughput", "Throughput"),
            ("switch_count", "Switch Count", "Switches"),
        ]
        for ax, (col, title, ylabel) in zip(axes, metrics):
            safe_plot(col, ax, title, ylabel)
        ax_loss = fig.add_subplot(5, 2, 9)
        if has_step_loss and train_steps is not None and not train_steps.empty:
            xcol = "global_step" if "global_step" in train_steps.columns else ("step" if "step" in train_steps.columns else train_steps.index)
            ax_loss.plot(train_steps[xcol].values, train_steps["loss"].values, alpha=0.6)
            ax_loss.set_title("Training Loss (per-step)")
            ax_loss.set_xlabel("Global Step")
            ax_loss.set_ylabel("Loss")
            ax_loss.grid(True)
        else:
            ax_loss.text(0.5, 0.5, "No per-step loss available", ha="center", va="center")
            ax_loss.set_axis_off()
        ax10 = fig.add_subplot(5, 2, 10)
        ax10.set_axis_off()
        plt.tight_layout()
        save_fig(fig, out_path)
        return True
    except Exception:
        raise

def plot_rl_curves_with_loss(episode_rows_by_seed: List[pd.DataFrame], step_rows_by_seed: List[pd.DataFrame], out_dir: str, run_id: str, model_name: str):
    try:
        for i, ep_df in enumerate(episode_rows_by_seed):
            seed_label = f"seed_{i}"
            step_df = step_rows_by_seed[i] if i < len(step_rows_by_seed) else None
            out_path = os.path.join(out_dir, f"{seed_label}_curves_with_loss.png")
            try:
                plot_rl_curves_with_loss_for_seed(ep_df, step_df, out_path, title_prefix=seed_label.upper())
                print(f"[EDA] Saved per-seed curves: {out_path}")
            except Exception as e:
                log_exception(out_dir, f"plot_rl_curves_with_loss_seed_{i}", e)
        try:
            if not episode_rows_by_seed:
                return
            dfs = [_normalize_columns_lower(df) for df in episode_rows_by_seed]
            max_ep = max(len(df) for df in dfs)
            agg_rows = []
            metrics = ["cumulative_reward", "avg_vehicle_queue", "avg_ped_queue", "avg_vehicle_wait", "avg_ped_wait", "vehicle_total_throughput", "ped_total_throughput", "switch_count"]
            for ep in range(max_ep):
                row = {"episode": ep}
                for metric in metrics:
                    vals = []
                    for df in dfs:
                        if ep < len(df) and metric in df.columns:
                            try:
                                vals.append(float(df.iloc[ep][metric]))
                            except Exception:
                                pass
                    row[f"{metric}_mean"] = np.mean(vals) if vals else np.nan
                agg_rows.append(row)
            agg_df = pd.DataFrame(agg_rows)
            fig, axes = plt.subplots(4, 2, figsize=(18, 20))
            axes = axes.flatten()
            plot_map = [
                ("cumulative_reward_mean", "Cumulative Reward", "Reward"),
                ("avg_vehicle_queue_mean", "Average Vehicle Queue", "Queue"),
                ("avg_ped_queue_mean", "Average Vehicle Queue", "Queue"),
                ("avg_vehicle_wait_mean", "Average Vehicle Wait", "Wait"),
                ("avg_ped_wait_mean", "Average Vehicle Wait", "Wait"),
                ("vehicle_total_throughput_mean", "Vehicle Throughput", "Throughput"),
                ("ped_total_throughput_mean", "Pedestrian Throughput", "Throughput"),
                ("switch_count_mean", "Switch Count", "Switches"),
            ]
            for ax, (col, title, ylabel) in zip(axes, plot_map):
                if col in agg_df.columns:
                    ax.plot(agg_df["episode"].values, agg_df[col].values, lw=1.5)
                    ax.set_title(f"{run_id.upper()} {model_name} — {title} (mean across seeds)")
                    ax.set_xlabel("episode")
                    ax.set_ylabel(ylabel)
                    ax.grid(True)
            out_path = os.path.join(out_dir, f"{run_id}_{model_name}_curves_with_loss.png")
            save_fig(fig, out_path)
            print(f"[EDA] Saved aggregated curves: {out_path}")
        except Exception as e:
            log_exception(out_dir, "plot_rl_curves_with_loss_aggregated", e)
    except Exception as e:
        log_exception(out_dir, "plot_rl_curves_with_loss_top", e)

# -----------------------
# Model comparison visuals
# -----------------------
def compute_model_level_metrics(episode_rows_by_seed: List[pd.DataFrame], sample_eff: Dict[int, Optional[int]]) -> Dict[str, Any]:
    try:
        final_rewards = []
        lengths = []
        for df in episode_rows_by_seed:
            if "cumulative_reward" in df.columns and len(df) > 0:
                try:
                    final_rewards.append(float(df.iloc[-1]["cumulative_reward"]))
                except Exception:
                    pass
            if "episode_length" in df.columns:
                try:
                    lengths.append(float(df["episode_length"].mean()))
                except Exception:
                    pass
        mean_r, lci, uci = mean_confidence_interval(final_rewards) if final_rewards else (np.nan, np.nan, np.nan)
        mean_len, len_lci, len_uci = mean_confidence_interval(lengths) if lengths else (np.nan, np.nan, np.nan)
        steps = [v for v in sample_eff.values() if v is not None]
        median_step = int(np.median(steps)) if steps else None
        mean_step = float(np.mean(steps)) if steps else None
        return {
            "final_mean_reward": mean_r,
            "final_reward_ci_lower": lci,
            "final_reward_ci_upper": uci,
            "mean_episode_length": mean_len,
            "length_ci_lower": len_lci,
            "length_ci_upper": len_uci,
            "sample_eff_median": median_step,
            "sample_eff_mean": mean_step,
            "num_seeds": len(episode_rows_by_seed),
        }
    except Exception as e:
        log_exception(".", "compute_model_level_metrics", e)
        return {}

def plot_bar_comparison(models_metrics: Dict[str, Dict[str, Any]], out_dir: str, run_id: str):
    try:
        labels = []
        means = []
        lowers = []
        uppers = []
        for model, m in models_metrics.items():
            labels.append(model)
            means.append(m.get("final_mean_reward", np.nan))
            lowers.append(m.get("final_reward_ci_lower", np.nan))
            uppers.append(m.get("final_reward_ci_upper", np.nan))
        if not labels:
            return
        x = np.arange(len(labels))
        fig, ax = plt.subplots(figsize=(10, 6))
        yerr = np.array([np.array(means) - np.array(lowers), np.array(uppers) - np.array(means)])
        ax.bar(x, means, yerr=yerr, capsize=6, color=sns.color_palette("tab10", len(labels)))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Final mean cumulative reward")
        ax.set_title(f"{run_id.upper()} — Model comparison: final mean reward (95% CI)")
        save_fig(fig, os.path.join(out_dir, f"{run_id}_model_comparison_final_reward.png"))
    except Exception as e:
        log_exception(out_dir, "plot_bar_comparison", e)

def plot_radar_profiles(models_metrics: Dict[str, Dict[str, Any]], out_dir: str, run_id: str):
    try:
        keys = ["final_mean_reward", "mean_episode_length", "sample_eff_median"]
        labels = ["Reward", "EpisodeLen", "SampleEffMedian"]
        models = list(models_metrics.keys())
        if not models:
            return
        N = len(labels)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, polar=True)
        for model in models:
            vals = []
            mm = models_metrics[model]
            for k in keys:
                v = mm.get(k, np.nan)
                vals.append(0 if v is None or (isinstance(v, float) and np.isnan(v)) else float(v))
            vals += vals[:1]
            ax.plot(angles, vals, label=model)
            ax.fill(angles, vals, alpha=0.15)
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        ax.set_title(f"{run_id.upper()} — Model performance profiles")
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        save_fig(fig, os.path.join(out_dir, f"{run_id}_model_radar_profiles.png"))
    except Exception as e:
        log_exception(out_dir, "plot_radar_profiles", e)

def plot_learning_overlay_across_models(models_episode_by_seed: Dict[str, List[pd.DataFrame]], out_dir: str, run_id: str, smooth_window: int = 1):
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        for model, ep_list in models_episode_by_seed.items():
            means, lcis, ucis = aggregate_episode_rewards_across_seeds(ep_list)
            episodes = np.arange(len(means))
            if len(episodes) == 0:
                continue
            ax.plot(episodes, means, lw=1.8, label=model)
            ax.fill_between(episodes, lcis, ucis, alpha=0.15)
        ax.set_title(f"{run_id.upper()} — Learning curves overlay (mean across seeds)")
        ax.set_xlabel("episode")
        ax.set_ylabel("cumulative reward")
        ax.legend()
        save_fig(fig, os.path.join(out_dir, f"{run_id}_learning_overlay_models.png"))
    except Exception as e:
        log_exception(out_dir, "plot_learning_overlay_across_models", e)

def plot_scatter_perf_vs_cost(models_metrics: Dict[str, Dict[str, Any]], out_dir: str, run_id: str):
    try:
        rows = []
        for model, m in models_metrics.items():
            rows.append({
                "model": model,
                "reward": m.get("final_mean_reward", np.nan),
                "sample_eff": m.get("sample_eff_median", np.nan),
                "num_seeds": m.get("num_seeds", 0),
            })
        df = pd.DataFrame(rows)
        if df.empty:
            return
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x="sample_eff", y="reward", size="num_seeds", hue="model", data=df, ax=ax, legend="brief", sizes=(50, 300))
        ax.set_xlabel("Sample-efficiency (median global step to threshold)")
        ax.set_ylabel("Final mean cumulative reward")
        ax.set_title(f"{run_id.upper()} — Performance vs Computational Cost")
        save_fig(fig, os.path.join(out_dir, f"{run_id}_perf_vs_cost_scatter.png"))
    except Exception as e:
        log_exception(out_dir, "plot_scatter_perf_vs_cost", e)

# -----------------------
# Discovery and I/O helpers
# -----------------------
def discover_models_under_run(run_dir: str) -> List[str]:
    try:
        if not os.path.exists(run_dir):
            return []
        models = []
        for name in sorted(os.listdir(run_dir)):
            p = os.path.join(run_dir, name)
            if os.path.isdir(p) and not name.startswith("seed_"):
                models.append(name)
        run_level_candidates = [
            os.path.join(run_dir, f"{os.path.basename(run_dir)}_episode_metrics.csv"),
            os.path.join(run_dir, "rl_episode_metrics.csv"),
        ]
        run_level_exists = any(os.path.exists(p) for p in run_level_candidates)
        if run_level_exists:
            return ["."]
        if not models:
            return ["."]
        return models
    except Exception:
        return []

def read_episode_and_step_for_seed(seed_dir: str, run_id: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    try:
        ep_csv = os.path.join(seed_dir, "rl_episode_metrics.csv")
        step_csv = os.path.join(seed_dir, "rl_step_metrics.csv")
        alt_ep = os.path.join(seed_dir, f"{run_id}_episode_metrics.csv")
        alt_step = os.path.join(seed_dir, f"{run_id}_step_metrics.csv")
        ep_df = None
        step_df = None
        for p in (ep_csv, alt_ep):
            if os.path.exists(p):
                try:
                    ep_df = pd.read_csv(p)
                    break
                except Exception:
                    continue
        for p in (step_csv, alt_step):
            if os.path.exists(p):
                try:
                    step_df = pd.read_csv(p)
                    break
                except Exception:
                    continue
        return ep_df, step_df
    except Exception as e:
        return None, None

# -----------------------
# Main per-run EDA
# -----------------------
def run_eda_for_run(run_id: str, results_dir: str, requested_seeds: List[int], out_root: str, smooth_window: int = 1):
    run_dir = os.path.join(results_dir, run_id)
    out_dir = ensure_dir(os.path.join(out_root, run_id))
    summary: Dict[str, Any] = {"run_id": run_id, "models": {}, "notes": []}

    try:
        print(f"[EDA] run_dir = {run_dir}")
        print(f"[EDA] out_dir = {out_dir}")
        print(f"[EDA] requested seeds = {requested_seeds}")
        if not os.path.exists(run_dir):
            print(f"[EDA][WARN] run_dir does not exist: {run_dir}")
        else:
            models = discover_models_under_run(run_dir)
            if models == ["."]:
                seeds_present = sorted([d for d in os.listdir(run_dir) if d.startswith("seed_")])
                print(f"[EDA][INFO] Legacy layout detected. seed folders present under {run_dir}: {seeds_present}")
            else:
                print(f"[EDA][INFO] Model folders discovered under {run_dir}: {models}")
                for m in models:
                    mdir = os.path.join(run_dir, m)
                    seed_folders = sorted([d for d in os.listdir(mdir) if d.startswith("seed_")]) if os.path.exists(mdir) else []
                    print(f"[EDA][INFO] Model '{m}' seed folders: {seed_folders}")
    except Exception as e:
        log_exception(out_dir, "discovery_logging", e)

    models = discover_models_under_run(run_dir)
    if not models:
        print(f"[EDA][WARN] No models or legacy run-level files found under {run_dir}. Skipping.")
        return

    models_episode_by_seed: Dict[str, List[pd.DataFrame]] = {}
    models_step_by_seed: Dict[str, List[pd.DataFrame]] = {}
    models_sample_eff: Dict[str, Dict[int, Optional[int]]] = {}
    models_metrics: Dict[str, Dict[str, Any]] = {}

    for model_name in models:
        try:
            if model_name == ".":
                model_dir = run_dir
                model_label = "legacy"
            else:
                model_dir = os.path.join(run_dir, model_name)
                model_label = model_name
            model_out_dir = ensure_dir(os.path.join(out_dir, model_label))
            seeds = []
            if requested_seeds:
                for s in requested_seeds:
                    seed_dir = os.path.join(model_dir, f"seed_{s}")
                    if os.path.exists(seed_dir):
                        seeds.append(s)
                    else:
                        print(f"[EDA][WARN] requested seed {s} not found under {model_dir}")
            else:
                if os.path.exists(model_dir):
                    for name in sorted(os.listdir(model_dir)):
                        if name.startswith("seed_"):
                            try:
                                seeds.append(int(name.split("_", 1)[1]))
                            except Exception:
                                continue
            episode_rows_by_seed = []
            step_rows_by_seed = []
            if seeds:
                for s in seeds:
                    seed_dir = os.path.join(model_dir, f"seed_{s}")
                    ep_df, step_df = read_episode_and_step_for_seed(seed_dir, run_id)
                    if ep_df is not None:
                        episode_rows_by_seed.append(ep_df)
                    if step_df is not None:
                        step_rows_by_seed.append(step_df)
            else:
                ep_csv = os.path.join(model_dir, f"{run_id}_episode_metrics.csv")
                step_csv = os.path.join(model_dir, f"{run_id}_step_metrics.csv")
                alt_ep = os.path.join(model_dir, "rl_episode_metrics.csv")
                alt_step = os.path.join(model_dir, "rl_step_metrics.csv")
                found_any = False
                for p in (ep_csv, alt_ep):
                    if os.path.exists(p):
                        try:
                            episode_rows_by_seed.append(pd.read_csv(p))
                            print(f"[EDA] Loaded run-level episode CSV: {p}")
                            found_any = True
                            break
                        except Exception as e:
                            log_exception(model_out_dir, "read_run_level_episode_csv", e)
                for p in (step_csv, alt_step):
                    if os.path.exists(p):
                        try:
                            step_rows_by_seed.append(pd.read_csv(p))
                            print(f"[EDA] Loaded run-level step CSV: {p}")
                            found_any = True
                            break
                        except Exception as e:
                            log_exception(model_out_dir, "read_run_level_step_csv", e)
                if not found_any and model_name != ".":
                    top_ep = os.path.join(run_dir, f"{run_id}_episode_metrics.csv")
                    top_alt_ep = os.path.join(run_dir, "rl_episode_metrics.csv")
                    top_step = os.path.join(run_dir, f"{run_id}_step_metrics.csv")
                    top_alt_step = os.path.join(run_dir, "rl_step_metrics.csv")
                    for p in (top_ep, top_alt_ep):
                        if os.path.exists(p):
                            try:
                                episode_rows_by_seed.append(pd.read_csv(p))
                                print(f"[EDA] Loaded top-level run-level episode CSV for model fallback: {p}")
                                break
                            except Exception as e:
                                log_exception(model_out_dir, "read_top_level_episode_csv", e)
                    for p in (top_step, top_alt_step):
                        if os.path.exists(p):
                            try:
                                step_rows_by_seed.append(pd.read_csv(p))
                                print(f"[EDA] Loaded top-level run-level step CSV for model fallback: {p}")
                                break
                            except Exception as e:
                                log_exception(model_out_dir, "read_top_level_step_csv", e)

            models_episode_by_seed[model_label] = episode_rows_by_seed
            models_step_by_seed[model_label] = step_rows_by_seed
            sample_eff = read_sample_efficiency_summary(model_dir)
            models_sample_eff[model_label] = sample_eff

            try:
                if step_rows_by_seed:
                    all_steps_df = pd.concat(step_rows_by_seed, ignore_index=True)
                else:
                    all_steps_df = pd.DataFrame()
                label = f"{run_id}_{model_label}"
                if not all_steps_df.empty:
                    try:
                        step_col = detect_step_col(all_steps_df)
                    except KeyError:
                        step_col = "global_step" if "global_step" in all_steps_df.columns else all_steps_df.columns[0]
                    with open(os.path.join(model_out_dir, f"{label}_step_count.txt"), "w") as f:
                        f.write(f"{label} total steps: {len(all_steps_df)}\n")
                    for feature in NUM_FEATURES:
                        if feature in all_steps_df.columns:
                            plot_distribution(all_steps_df, feature, model_out_dir, label)
                    for feature in CAT_FEATURES:
                        if feature in all_steps_df.columns:
                            try:
                                fig, ax = plt.subplots(figsize=(8, 4.5))
                                sns.countplot(x=all_steps_df[feature].astype(str), ax=ax)
                                ax.set_title(f"{label} — Count Plot: {feature}")
                                save_fig(fig, os.path.join(model_out_dir, f"{label}_count_{feature}.png"))
                            except Exception as e:
                                log_exception(model_out_dir, f"countplot_{feature}", e)
                    corr_cols = [c for c in NUM_FEATURES + CAT_FEATURES + ["reward"] if c in all_steps_df.columns]
                    if len(corr_cols) >= 2:
                        plot_correlation(all_steps_df, corr_cols, model_out_dir, label)
                    plot_time_series(all_steps_df, step_col, [c for c in NUM_FEATURES + CAT_FEATURES if c in all_steps_df.columns], model_out_dir, label, sample_steps=10000)
                    present_num_features = [c for c in NUM_FEATURES if c in all_steps_df.columns]
                    if present_num_features:
                        plot_pca_tsne(all_steps_df, present_num_features, model_out_dir, label)
                    if "reward" in all_steps_df.columns:
                        plot_reward_distribution(all_steps_df, model_out_dir, label)
                    plot_episode_lengths(all_steps_df, model_out_dir, label)
                    plot_state_visitation(all_steps_df, model_out_dir, label, STATE_HEATMAP_PAIRS)
            except Exception as e:
                log_exception(model_out_dir, "model_baseline_plots", e)

            try:
                if episode_rows_by_seed:
                    try:
                        missing_start = any("episode_start_step" not in df.columns for df in episode_rows_by_seed)
                        if missing_start:
                            for df in episode_rows_by_seed:
                                if "episode_start_step" not in df.columns:
                                    if "episode" in df.columns and "episode_length" in df.columns and len(df):
                                        typical = int(df["episode_length"].median()) if len(df) else 1000
                                        df["episode_start_step"] = df["episode"].astype(int) * typical
                                        summary["notes"].append(f"Estimated episode_start_step for model {model_label} using episode * median(episode_length)")
                                    else:
                                        df["episode_start_step"] = df.index.astype(int) * 1000
                                        summary["notes"].append(f"Estimated episode_start_step for model {model_label} using index * 1000")
                    except Exception as e:
                        log_exception(model_out_dir, "estimate_episode_start_step", e)

                    try:
                        plot_learning_curve(episode_rows_by_seed, model_out_dir, f"{run_id}_{model_label}", smooth_window=smooth_window)
                    except Exception as e:
                        log_exception(model_out_dir, "plot_learning_curve", e)

                    try:
                        has_start_step = any("episode_start_step" in df.columns for df in episode_rows_by_seed)
                        if has_start_step:
                            plot_learning_curve_vs_steps(episode_rows_by_seed, model_out_dir, f"{run_id}_{model_label}", smooth_window=smooth_window)
                        else:
                            print(f"[EDA][WARN] No episode_start_step found for model {model_label}; step-aligned plots skipped.")
                    except Exception as e:
                        log_exception(model_out_dir, "plot_learning_curve_vs_steps_call", e)

                    try:
                        plot_rl_curves_with_loss(episode_rows_by_seed, step_rows_by_seed, model_out_dir, run_id, model_label)
                    except Exception as e:
                        log_exception(model_out_dir, "plot_rl_curves_with_loss", e)
            except Exception as e:
                log_exception(model_out_dir, "episode_level_plots", e)

            # Policy visualization (best-effort)
            try:
                seed_list = seeds if seeds else []
                if not seed_list and episode_rows_by_seed:
                    seed_list = [0]
                for idx, s in enumerate(seed_list):
                    seed_dir = os.path.join(model_dir, f"seed_{s}") if seeds else model_dir
                    seed_out_dir = ensure_dir(os.path.join(model_out_dir, f"seed_{s}")) if seeds else model_out_dir
                    try:
                        ok, info = record_episode_video_from_model(seed_dir, model_label, run_id, seed_out_dir, max_steps=500, fps=15)
                        if ok:
                            print(f"[EDA] Saved episode video: {info}")
                        else:
                            summary["notes"].append(f"Video not created for {seed_dir}: {info}")
                    except Exception as e:
                        log_exception(seed_out_dir, f"record_video_seed_{s}", e)
                    try:
                        step_rows = []
                        ep_df, step_df = read_episode_and_step_for_seed(seed_dir, run_id)
                        if step_df is not None:
                            step_rows.append(step_df)
                        ok, info = plot_value_function_heatmap_from_model(seed_dir, model_label, run_id, seed_out_dir, step_rows, grid_size=40)
                        if ok:
                            print(f"[EDA] Saved value heatmap: {info}")
                        else:
                            summary["notes"].append(f"Value heatmap not created for {seed_dir}: {info}")
                    except Exception as e:
                        log_exception(seed_out_dir, f"value_heatmap_seed_{s}", e)
                    try:
                        if step_df is not None:
                            ok, info = plot_trajectory_if_positions(step_df, seed_out_dir, f"{run_id}_{model_label}_seed_{s}")
                            if ok:
                                print(f"[EDA] Saved trajectory plot: {info}")
                            else:
                                summary["notes"].append(f"Trajectory not created for {seed_dir}: {info}")
                    except Exception as e:
                        log_exception(seed_out_dir, f"trajectory_seed_{s}", e)
            except Exception as e:
                log_exception(model_out_dir, "policy_visualization", e)

            try:
                metrics = compute_model_level_metrics(episode_rows_by_seed, sample_eff)
                models_metrics[model_label] = metrics
                models_sample_eff[model_label] = sample_eff
                summary["models"][model_label] = {
                    "num_seeds": metrics.get("num_seeds", 0),
                    "metrics": metrics,
                    "files": {
                        "episode_csvs": [os.path.join(model_dir, f"seed_{s}", "rl_episode_metrics.csv") for s in seeds] if seeds else [],
                    }
                }
            except Exception as e:
                log_exception(model_out_dir, "compute_model_level_metrics", e)
        except Exception as e:
            log_exception(out_dir, f"model_{model_name}_top", e)

    # After processing all models, produce model-comparison visuals
    try:
        if models_metrics:
            plot_bar_comparison(models_metrics, out_dir, run_id)
            plot_radar_profiles(models_metrics, out_dir, run_id)
            plot_learning_overlay_across_models(models_episode_by_seed, out_dir, run_id, smooth_window=smooth_window)
            plot_scatter_perf_vs_cost(models_metrics, out_dir, run_id)
            try:
                fig, ax = plt.subplots(figsize=(12, 6))
                for model, ep_list in models_episode_by_seed.items():
                    means, lcis, ucis = aggregate_episode_rewards_across_seeds(ep_list)
                    episodes = np.arange(len(means))
                    if len(episodes) == 0:
                        continue
                    ax.plot(episodes, means, lw=1.8, label=model)
                    ax.fill_between(episodes, lcis, ucis, alpha=0.12)
                ax.set_title(f"{run_id.upper()} — RL overlay: learning curves across models")
                ax.set_xlabel("episode")
                ax.set_ylabel("cumulative reward")
                ax.legend()
                save_fig(fig, os.path.join(out_dir, f"{run_id}_rl_overlay_models.png"))
            except Exception as e:
                log_exception(out_dir, "rl_overlay_models", e)
    except Exception as e:
        log_exception(out_dir, "model_comparison_plots", e)

    try:
        with open(os.path.join(out_dir, f"{run_id}_eda_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
    except Exception as e:
        log_exception(out_dir, "write_summary_json", e)

    print(f"[EDA] Completed run {run_id}. Outputs saved to {out_dir}")

# -----------------------
# CLI
# -----------------------
def parse_seeds(s: Optional[str]) -> List[int]:
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    seeds = []
    for p in parts:
        try:
            seeds.append(int(p))
        except Exception:
            pass
    return seeds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for RL / baseline results (reads CSVs produced by train.py).")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR, help="Base results directory")
    parser.add_argument("--run_ids", type=str, default="rl,b1", help="Comma-separated run ids (subfolders under results/)")
    parser.add_argument("--seeds", type=str, default="", help="Comma-separated integer seeds to include (e.g., 42,43,44). If empty, will auto-detect seed_* folders under each model folder.")
    parser.add_argument("--smooth_window", type=int, default=1, help="Moving-average window for learning curve smoothing (default 1 = no smoothing)")
    args = parser.parse_args()

    results_dir = args.results_dir
    run_ids = [r.strip() for r in args.run_ids.split(",") if r.strip()]
    requested_seeds = parse_seeds(args.seeds)
    out_root = ensure_dir(os.path.join(results_dir, EDA_DIRNAME))

    for run_id in run_ids:
        try:
            run_eda_for_run(
                run_id=run_id,
                results_dir=results_dir,
                requested_seeds=requested_seeds,
                out_root=out_root,
                smooth_window=args.smooth_window,
            )
        except Exception as e:
            run_out_dir = os.path.join(out_root, run_id)
            log_exception(run_out_dir, "run_eda_for_run_top_level", e)
            print(f"[EDA][ERROR] run_eda_for_run failed for {run_id}: {e}")

    print(f"\nAll EDA outputs saved to {out_root}")
