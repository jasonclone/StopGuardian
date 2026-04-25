# rl/tune.py
"""
Ray Tune wrapper (Optuna + ASHA) updated to match modern train.py / TrafficEnv.

This patched version:
- Replaces deprecated tune.get_trial_dir() with train.get_context().get_trial_dir()
- Records per-trial wall-clock and best-effort GPU-seconds
- Ensures parallel coordinates is saved as PNG when possible (kaleido), with robust fallbacks
- Makes trial/result handling robust across Ray versions
- Does NOT call .backward() in the wrapper; train_step_* should handle optimization
- Ensures every tune.report includes the required 'loss' metric (fixes AsyncHyperBandScheduler error)
- Produces required report artifacts:
  - trials_table.csv
  - parallel_coords.png (or fallback PNG/HTML)
  - hyperparam_importance.png (or HTML fallback)
  - tuning_costs.json (num_trials, wall_clock_seconds, gpu_hours_estimate)
  - final_selection.txt
"""

import os
import sys
import time
import json
import math
from typing import Dict, Any, List

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray import tune, train
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import with_resources

import torch
import numpy as np

# Optional plotting/data libs
try:
    import pandas as pd
except Exception as e:
    print("Exception: ", e)
    pd = None

try:
    import plotly.express as px
    import plotly.io as pio
    PLOTLY_AVAILABLE = True
except Exception as e:
    print("Exception: ", e)
    px = None
    pio = None
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except Exception as e:
    print("Exception: ", e)
    plt = None
    sns = None
    MATPLOTLIB_AVAILABLE = False

# Try to import kaleido for static image export
KALEIDO_AVAILABLE = False
if PLOTLY_AVAILABLE:
    try:
        import kaleido  # noqa: F401
        KALEIDO_AVAILABLE = True
    except Exception:
        KALEIDO_AVAILABLE = False

# Local imports (must exist in project)
from config import (
    set_hparams,
    DEVICE,
    GAMMA,
    N_STEPS,
    BUFFER_SIZE,
    BATCH_SIZE,
    NOISY_SIGMA,
    MIN_REPLAY_SIZE,
    WARMUP_STEPS,
    USE_EPSILON,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    ACTIONS,
    NUM_ACTIONS,
    PRIORITY_ALPHA,
    PRIORITY_BETA_START,
    PRIORITY_BETA_END,
    TAU,
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)

from traffic_env import TrafficEnv, make_sumo_config
from train import init_models_and_replay
from action import select_action, apply_action_safe
from learner import train_step_rainbow, train_step_standard

# Ensure SUMO_HOME/tools are on path
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
if tools not in sys.path:
    sys.path.append(tools)

# Tuning output directory
TUNING_DIR = os.path.abspath(os.path.join("results", "tuning"))
os.makedirs(TUNING_DIR, exist_ok=True)

# Optuna search with concurrency limiter
optuna_search = ConcurrencyLimiter(
    OptunaSearch(metric="loss", mode="min"),
    max_concurrent=8,
)

# Train wrapper executed inside each Tune trial
def train_wrapper(config: Dict[str, Any]) -> None:
    """
    config: hyperparameter dict from Tune (contains only the tuned keys).
    This function applies config via set_hparams (best-effort), creates a TrafficEnv,
    computes state_size, builds models via init_models_and_replay, runs a small number
    of episodes, and reports metrics to Ray Tune. It also reports trial_wall_time_s and
    a best-effort gpu_time (seconds) for post-processing.

    IMPORTANT: train_step_rainbow and train_step_standard are expected to perform
    their own backward/optimizer steps and return either a torch.Tensor or numeric loss.
    The wrapper will not call .backward() on the returned value.
    """
    # record trial start time
    trial_start_time = time.time()

    # Attempt to detect assigned GPUs (best-effort)
    try:
        ctx = train.get_context()
        try:
            trial_resources = ctx.get_trial_resources()
            num_gpus_assigned = float(trial_resources.get("GPU", 0.0)) if trial_resources else 0.0
        except Exception:
            num_gpus_assigned = 0.0
    except Exception:
        num_gpus_assigned = 0.0

    # Apply hyperparameters to global config if set_hparams exists
    try:
        set_hparams(**config)
    except Exception as e:
        print("Exception: ", e)
        pass

    # local tau: prefer config value if present, else fallback to TAU from config.py
    tau = float(config.get("TAU", TAU))

    env = TrafficEnv("C")

    # small sleep to reduce SUMO port collisions
    time.sleep(0.1)

    # Determine state size inside a SUMO session owned by this worker
    try:
        env.start(make_sumo_config())
        env.step()
        env.initialize_from_sumo()
        state_size = env.compute_state_size()
    finally:
        try:
            env.close()
        except Exception:
            pass

    # Trial-level defaults (kept outside search space)
    MAX_EPISODES = int(config.get("MAX_EPISODES", 12))
    STEPS_PER_EPISODE = int(config.get("STEPS_PER_EPISODE", 1000))

    # Compute lr decay steps for scheduler (must be positive int)
    total_updates = max(1, int(MAX_EPISODES * STEPS_PER_EPISODE - MIN_REPLAY_SIZE))

    # Build models and replay buffer (no SUMO required)
    model_type = config.get("MODEL_TYPE", "rainbow")
    online_model, target_model, replay_buffer, optimizer = init_models_and_replay(
        state_size=state_size,
        lr_decay_steps=total_updates,
        model_type=model_type,
    )

    try:
        online_model.to(DEVICE)
        if target_model is not None:
            target_model.to(DEVICE)
    except Exception as e:
        print("Exception: ", e)
        pass

    # Beta schedule for prioritized replay (if used)
    beta = float(config.get("PRIORITY_BETA_START", 0.4))
    beta_end = float(config.get("PRIORITY_BETA_END", 0.9))
    total_steps = max(1, MAX_EPISODES * STEPS_PER_EPISODE)
    beta_increment = (beta_end - beta) / float(total_steps)

    last_loss = float("inf")
    last_training_iteration = 0

    for ep in range(MAX_EPISODES):
        # Start SUMO session for this episode
        try:
            env.start(make_sumo_config())
            env.step()
            env.initialize_from_sumo()
            env.reset_tracking()
        except Exception as e:
            # Always report a loss metric so scheduler doesn't crash
            train.report({"loss": float("inf"), "training_iteration": ep + 1})
            break

        cumulative_reward = 0.0
        env_time_total = 0.0
        train_time_total = 0.0
        train_steps = 0

        try:
            for t in range(STEPS_PER_EPISODE):
                t0 = time.perf_counter()
                try:
                    state_raw, _ = env.get_state()
                except Exception as e:
                    # SUMO socket errors can happen; break episode gracefully
                    print("Exception getting state from SUMO:", e)
                    break

                action = select_action(
                    env=env,
                    state_raw=state_raw,
                    global_step=t,
                    mode="train",
                    online_model=online_model,
                    warmup_steps=WARMUP_STEPS,
                    normalize_state_torch=env.normalize_state_torch,
                )

                apply_action_safe(action, env, current_step_global=t)
                try:
                    env.step()
                except Exception as e:
                    print("Exception stepping SUMO:", e)
                    break

                try:
                    env.record_phase_start_if_changed(next_global_step=t + 1)
                except Exception:
                    pass

                try:
                    next_state_raw, _ = env.get_state()
                except Exception as e:
                    print("Exception getting next state from SUMO:", e)
                    break

                reward = env.get_reward(next_state_raw, state_raw, action)
                # sanitize reward/state
                if not np.isfinite(reward):
                    reward = float(np.nan_to_num(reward, nan=0.0, posinf=1e6, neginf=-1e6))
                cumulative_reward += reward

                done = (t == STEPS_PER_EPISODE - 1)
                # sanitize states before adding to replay
                try:
                    state_safe = np.nan_to_num(state_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                    next_state_safe = np.nan_to_num(next_state_raw, nan=0.0, posinf=1e6, neginf=-1e6)
                    replay_buffer.add(state_safe, action, reward, next_state_safe, done)
                except Exception as e:
                    print("Exception adding to replay:", e)
                    pass

                t1 = time.perf_counter()
                env_time_total += (t1 - t0)

                # Training update: train_step_* is expected to perform optimization internally
                t2 = time.perf_counter()
                try:
                    if model_type == "rainbow":
                        loss = train_step_rainbow(
                            online_model=online_model,
                            target_model=target_model,
                            replay_buffer=replay_buffer,
                            optimizer=optimizer,
                            batch_size=int(config.get("BATCH_SIZE", 64)),
                            min_replay_size=int(MIN_REPLAY_SIZE),
                            gamma=float(config.get("GAMMA", 0.99)),
                            n_steps=int(config.get("N_STEPS", 3)),
                            v_min=float(config.get("V_MIN", -80.0)),
                            v_max=float(config.get("V_MAX", 0.0)),
                            delta_z=(float(config.get("V_MAX", 0.0)) - float(config.get("V_MIN", -80.0))) / (int(NUM_ATOMS) - 1),
                            current_step_global=t,
                            priority_beta_start=float(config.get("PRIORITY_BETA_START", 0.4)),
                            priority_beta_end=float(config.get("PRIORITY_BETA_END", 0.9)),
                            total_updates=total_updates,
                            normalize_state_torch=env.normalize_state_torch,
                            tau=float(config.get("TAU", tau)),
                        )
                    else:
                        loss = train_step_standard(
                            online_model=online_model,
                            replay_buffer=replay_buffer,
                            optimizer=optimizer,
                            batch_size=int(config.get("BATCH_SIZE", 64)),
                            min_replay_size=int(MIN_REPLAY_SIZE),
                            gamma=float(config.get("GAMMA", 0.99)),
                            normalize_state_torch=env.normalize_state_torch,
                        )
                except Exception as e:
                    print("Exception during train step:", e)
                    loss = None

                # The wrapper will not call backward/step. Accept tensor or numeric loss for logging.
                try:
                    if loss is None:
                        # keep last_loss unchanged
                        pass
                    elif torch.is_tensor(loss):
                        if not torch.isfinite(loss):
                            print("Non-finite tensor loss detected; logging and continuing. loss=", loss)
                            last_loss = float("inf")
                        else:
                            last_loss = float(loss.detach().cpu().item())
                    else:
                        # numeric scalar (float or numpy) returned by train_step; log and continue
                        try:
                            last_loss = float(loss)
                        except Exception:
                            print("Warning: train_step returned non-numeric, non-tensor loss:", type(loss))
                except Exception as e:
                    print("Exception checking loss:", e)

                t3 = time.perf_counter()
                train_time_total += (t3 - t2)
                train_steps += 1

                beta = min(beta_end, beta + beta_increment)

        finally:
            try:
                env.close()
            except Exception:
                pass

        avg_env_ms = (env_time_total / max(1, STEPS_PER_EPISODE)) * 1000.0
        avg_train_ms = (train_time_total / max(1, train_steps)) * 1000.0
        last_training_iteration = ep + 1

        # Always include 'loss' in reported metrics to satisfy schedulers
        try:
            train.report(
                {
                    "loss": float(last_loss) if last_loss is not None else float("inf"),
                    "training_iteration": last_training_iteration,
                    "avg_env_step_ms": avg_env_ms,
                    "avg_train_step_ms": avg_train_ms,
                    "episode_reward": float(cumulative_reward),
                }
            )
        except Exception as e:
            # If reporting fails, print and continue; ensure at least a minimal report with loss
            print("Exception during train.report():", e)
            try:
                tune.report(loss=float(last_loss) if last_loss is not None else float("inf"), training_iteration=last_training_iteration)
            except Exception:
                pass

    # Save final model (best-effort) into trial dir
    try:
        trial_dir = None
        try:
            trial_dir = train.get_context().get_trial_dir()
        except Exception:
            try:
                import ray.train as _rt
                trial_dir = _rt.get_context().get_trial_dir()
            except Exception:
                trial_dir = None

        if trial_dir:
            model_path = os.path.join(trial_dir, "final_model.pth")
            try:
                torch.save(online_model.state_dict(), model_path)
            except Exception as e:
                print("Exception saving final model:", e)
                pass
    except Exception as e:
        print("Exception: ", e)
        pass

    # report trial-level wall time and best-effort GPU seconds
    trial_end_time = time.time()
    trial_wall_time_s = trial_end_time - trial_start_time
    gpu_seconds_est = trial_wall_time_s * float(num_gpus_assigned or 0.0)

    # Ensure we include 'loss' in this final report as well (some Ray versions treat this as a separate result)
    try:
        train.report({"trial_wall_time_s": trial_wall_time_s, "gpu_time": gpu_seconds_est, "loss": float(last_loss) if last_loss is not None else float("inf"), "training_iteration": last_training_iteration})
    except Exception:
        try:
            tune.report(trial_wall_time_s=trial_wall_time_s, gpu_time=gpu_seconds_est, loss=float(last_loss) if last_loss is not None else float("inf"), training_iteration=last_training_iteration)
        except Exception:
            pass


# -------------------------
# SEARCH SPACE (exactly the keys requested)
# -------------------------
search_space = {
    "LEARNING_RATE": tune.loguniform(1e-4, 1e-3),
    "GAMMA": tune.uniform(0.95, 0.99),
    "N_STEPS": tune.choice([3, 5, 7]),
    "BATCH_SIZE": tune.choice([32, 64, 128, 256]),
    "NOISY_SIGMA": tune.uniform(0.1, 0.5),
    "TAU": tune.loguniform(1e-4, 1e-1),
}

# -------------------------
# Helper: post-processing and reporting
# -------------------------
def _safe_get_trial_name(r):
    """
    Robust extraction of a trial name/id from a Ray Tune Result-like object.
    Tries multiple attributes and falls back to metrics/config fields.
    """
    try:
        for attr in ("logdir", "log_dir", "trial_dir", "trial_id", "trial_name"):
            val = getattr(r, attr, None)
            if val:
                try:
                    return os.path.basename(val) if isinstance(val, str) and (os.sep in val or "/" in val) else str(val)
                except Exception:
                    return str(val)
        metrics = getattr(r, "metrics", {}) or {}
        if isinstance(metrics, dict):
            for key in ("trial_name", "trial_id", "logdir", "log_dir"):
                if key in metrics and metrics[key]:
                    return str(metrics[key])
        cfg = getattr(r, "config", {}) or {}
        if isinstance(cfg, dict):
            for key in ("trial_name", "trial_id"):
                if key in cfg and cfg[key]:
                    return str(cfg[key])
    except Exception:
        pass
    try:
        return getattr(r, "trial_id", None) or getattr(r, "trial_name", None) or str(getattr(r, "experiment_tag", "")) or "unknown_trial"
    except Exception:
        return "unknown_trial"


def _save_trials_table_and_plots(results, run_name: str, out_dir: str, tuner_wall_clock: float = None):
    """
    - Read results (Ray Tune results object) and produce:
      - trials_table.csv
      - parallel_coords.png
      - hyperparam_importance.png (Optuna if available, else heatmap)
      - tuning_costs.json
      - final_selection.txt
    """
    os.makedirs(out_dir, exist_ok=True)
    # 1) Collect trial records from results
    trial_records = []
    gpu_time_sum = 0.0

    # results is a ray.tune.ResultGrid-like object; iterate over results
    try:
        all_results = list(results)
    except Exception as e:
        print("Exception: ", e)
        all_results = results

    for r in all_results:
        rec = {}
        # config
        try:
            cfg = getattr(r, "config", {}) or {}
            if isinstance(cfg, dict):
                rec.update({f"hp_{k}": v for k, v in cfg.items()})
        except Exception as e:
            print("Exception: ", e)
            pass
        # metrics: prefer final reported metrics
        try:
            metrics = getattr(r, "metrics", {}) or {}
            if isinstance(metrics, dict):
                rec["loss"] = float(metrics.get("loss", float("nan")))
                rec["episode_reward"] = float(metrics.get("episode_reward", float("nan")))
                rec["training_iteration"] = int(metrics.get("training_iteration", 0))
                rec["avg_env_step_ms"] = float(metrics.get("avg_env_step_ms", float("nan")))
                rec["avg_train_step_ms"] = float(metrics.get("avg_train_step_ms", float("nan")))
                # trial-level timing/gpu
                if "trial_wall_time_s" in metrics:
                    rec["trial_wall_time_s"] = float(metrics.get("trial_wall_time_s", 0.0) or 0.0)
                if "gpu_time" in metrics:
                    rec["gpu_time"] = float(metrics.get("gpu_time", 0.0) or 0.0)
                else:
                    rec["gpu_time"] = 0.0
        except Exception as e:
            print("Exception: ", e)
            pass
        # trial id/name (robust)
        try:
            rec["trial_name"] = _safe_get_trial_name(r)
        except Exception as e:
            print("Exception: ", e)
            rec["trial_name"] = "unknown_trial"

        # ensure keys exist
        rec.setdefault("trial_wall_time_s", None)
        rec.setdefault("gpu_time", 0.0)
        trial_records.append(rec)
        try:
            gpu_time_sum += float(rec.get("gpu_time", 0.0) or 0.0)
        except Exception:
            pass

    # Save all_trials.jsonl already done earlier; create CSV table
    csv_path = os.path.join(out_dir, "trials_table.csv")
    if pd is not None:
        df = pd.DataFrame(trial_records)
        df.to_csv(csv_path, index=False)
    else:
        keys = set()
        for rec in trial_records:
            keys.update(rec.keys())
        keys = sorted(keys)
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(keys) + "\n")
            for rec in trial_records:
                row = []
                for k in keys:
                    v = rec.get(k, "")
                    row.append(str(v).replace(",", ";"))
                f.write(",".join(row) + "\n")

    # Create parallel coordinates plot (hyperparams vs loss)
    pcoord_png_path = os.path.join(out_dir, "parallel_coords.png")
    pcoord_html_path = os.path.join(out_dir, "parallel_coords.html")
    try:
        if pd is not None:
            df_plot = df.copy()
            hp_cols = [c for c in df_plot.columns if c.startswith("hp_")]
            if "loss" in df_plot.columns and hp_cols:
                plot_df = df_plot[hp_cols + ["loss"]].dropna(how="all")
                for c in hp_cols:
                    try:
                        plot_df[c] = pd.to_numeric(plot_df[c], errors="coerce")
                    except Exception:
                        pass
                if PLOTLY_AVAILABLE:
                    fig = px.parallel_coordinates(plot_df, color="loss", labels={c: c.replace("hp_", "") for c in plot_df.columns})
                    # Try to write PNG using kaleido; fallback to HTML then matplotlib PNG
                    wrote_png = False
                    if KALEIDO_AVAILABLE:
                        try:
                            pio.write_image(fig, pcoord_png_path, width=1400, height=600, engine="kaleido")
                            wrote_png = True
                        except Exception as e:
                            print("Exception writing parallel coords PNG with kaleido:", e)
                            wrote_png = False
                    if not wrote_png:
                        try:
                            img_bytes = fig.to_image(format="png")
                            with open(pcoord_png_path, "wb") as fh:
                                fh.write(img_bytes)
                            wrote_png = True
                        except Exception as e:
                            print("Exception exporting fig to image bytes:", e)
                            wrote_png = False
                    if not wrote_png:
                        try:
                            fig.write_html(pcoord_html_path)
                        except Exception as e:
                            print("Exception writing parallel coords HTML:", e)
                            pass
                        # matplotlib fallback: correlation heatmap PNG
                        if MATPLOTLIB_AVAILABLE:
                            try:
                                plt.figure(figsize=(12, 8))
                                sns.heatmap(plot_df.corr(), annot=True, fmt=".2f", cmap="vlag")
                                plt.title("Hyperparam correlations (fallback)")
                                plt.tight_layout()
                                plt.savefig(os.path.join(out_dir, "parallel_coords_fallback_corr.png"))
                                plt.close()
                            except Exception as e:
                                print("Exception creating matplotlib fallback:", e)
                elif MATPLOTLIB_AVAILABLE:
                    try:
                        plt.figure(figsize=(12, 8))
                        sns.heatmap(plot_df.corr(), annot=True, fmt=".2f", cmap="vlag")
                        plt.title("Hyperparam correlations (fallback)")
                        plt.tight_layout()
                        plt.savefig(os.path.join(out_dir, "parallel_coords_fallback_corr.png"))
                        plt.close()
                    except Exception as e:
                        print("Exception creating matplotlib fallback:", e)
    except Exception as e:
        print("Exception: ", e)
        pass

    # Hyperparameter importance: try Optuna if available and if we can access the study
    importance_path = os.path.join(out_dir, "hyperparam_importance.png")
    importance_html_path = os.path.join(out_dir, "hyperparam_importance.html")
    study = None
    try:
        import optuna  # type: ignore
        try:
            search_alg = optuna_search.search_alg if hasattr(optuna_search, "search_alg") else optuna_search
            study = getattr(search_alg, "_optuna_study", None) or getattr(search_alg, "_study", None)
        except Exception as e:
            print("Exception accessing optuna study:", e)
            study = None

        if study is not None:
            try:
                fig = optuna.visualization.plot_param_importances(study)
                wrote_imp = False
                if PLOTLY_AVAILABLE and KALEIDO_AVAILABLE:
                    try:
                        pio.write_image(fig, importance_path, width=900, height=600, engine="kaleido")
                        wrote_imp = True
                    except Exception as e:
                        print("Exception writing optuna importance PNG:", e)
                        wrote_imp = False
                if not wrote_imp:
                    try:
                        fig.write_html(importance_html_path)
                    except Exception as e:
                        print("Exception writing optuna importance HTML:", e)
            except Exception as e:
                print("Exception plotting optuna importance:", e)
                study = None
    except Exception as e:
        print("Exception importing optuna:", e)
        study = None

    # If Optuna study not available, fallback to simple importance via correlation with loss
    if study is None and pd is not None:
        try:
            df_corr = df.copy()
            hp_cols = [c for c in df_corr.columns if c.startswith("hp_")]
            if "loss" in df_corr.columns and hp_cols:
                corr = df_corr[hp_cols + ["loss"]].corr()["loss"].drop("loss").abs().sort_values(ascending=False)
                if MATPLOTLIB_AVAILABLE:
                    try:
                        plt.figure(figsize=(8, 4))
                        sns.barplot(x=corr.values, y=[c.replace("hp_", "") for c in corr.index])
                        plt.xlabel("abs(correlation with loss)")
                        plt.tight_layout()
                        plt.savefig(importance_path)
                        plt.close()
                    except Exception as e:
                        print("Exception saving importance barplot:", e)
                elif PLOTLY_AVAILABLE:
                    try:
                        fig = px.bar(x=corr.values, y=[c.replace("hp_", "") for c in corr.index], orientation="h")
                        if KALEIDO_AVAILABLE:
                            try:
                                pio.write_image(fig, importance_path, width=800, height=400, engine="kaleido")
                            except Exception:
                                fig.write_html(importance_html_path)
                        else:
                            fig.write_html(importance_html_path)
                    except Exception as e:
                        print("Exception creating plotly importance:", e)
        except Exception as e:
            print("Exception: ", e)
            pass

    # Compute tuning costs: wall-clock, #trials, GPU-hours (best-effort)
    try:
        trial_times = [rec.get("trial_wall_time_s") for rec in trial_records if rec.get("trial_wall_time_s") is not None]
        if trial_times:
            if tuner_wall_clock is not None:
                wall_clock_s = float(tuner_wall_clock)
            else:
                wall_clock_s = float(max(trial_times))
        else:
            wall_clock_s = float(tuner_wall_clock) if tuner_wall_clock is not None else None
    except Exception as e:
        print("Exception computing wall clock:", e)
        wall_clock_s = None

    try:
        total_gpu_seconds = sum(float(rec.get("gpu_time", 0.0) or 0.0) for rec in trial_records)
        gpu_hours = total_gpu_seconds / 3600.0 if total_gpu_seconds is not None else None
    except Exception as e:
        print("Exception computing gpu hours:", e)
        gpu_hours = None

    tuning_costs = {
        "num_trials": len(trial_records),
        "wall_clock_seconds": wall_clock_s,
        "gpu_hours_estimate": gpu_hours,
    }
    with open(os.path.join(out_dir, "tuning_costs.json"), "w") as f:
        json.dump(tuning_costs, f, indent=2)

    # Final selection justification: best config and short justification
    try:
        best = results.get_best_result(metric="loss", mode="min")
        best_config = best.config if best is not None else None
        best_metrics = best.metrics if best is not None else {}
    except Exception as e:
        print("Exception getting best result:", e)
        best_config = None
        best_metrics = {}

    justification_lines = []
    justification_lines.append(f"Run name: {run_name}")
    justification_lines.append(f"Number of trials: {len(trial_records)}")
    if best_config is not None:
        justification_lines.append("Selected hyperparameters (best by 'loss'):")
        justification_lines.append(json.dumps(best_config, indent=2))
        justification_lines.append("Selected trial metrics:")
        justification_lines.append(json.dumps(best_metrics, indent=2))
        try:
            if pd is not None and "loss" in df.columns:
                topk = df.nsmallest(min(5, len(df)), "loss")["loss"].tolist()
                justification_lines.append(f"Top-{min(5, len(df))} losses: {topk}")
                justification_lines.append(f"Median of top-{min(5, len(df))}: {float(np.median(topk)) if topk else 'N/A'}")
        except Exception as e:
            print("Exception computing top-k stats:", e)
            pass
    else:
        justification_lines.append("No best config found in results object.")

    justification_lines.append("")
    justification_lines.append("Justification:")
    justification_lines.append("- Selection metric: 'loss' (minimize).")
    justification_lines.append("- Best config chosen because it achieved the lowest reported 'loss' across trials.")
    justification_lines.append("- For robustness, examine top-k trials and consider retraining the selected config with multiple seeds.")
    with open(os.path.join(out_dir, "final_selection.txt"), "w") as f:
        f.write("\n".join(justification_lines))

    # verify required artifacts exist and log missing ones
    required = {
        "trials_csv": csv_path,
        "parallel_coords_png": pcoord_png_path,
        "importance_plot": importance_path,
        "tuning_costs": os.path.join(out_dir, "tuning_costs.json"),
        "final_selection": os.path.join(out_dir, "final_selection.txt"),
    }
    missing = [k for k, p in required.items() if not os.path.exists(p)]
    if missing:
        print("Warning: missing report artifacts:", missing)

    return {
        "trials_csv": csv_path,
        "parallel_coords": pcoord_png_path if os.path.exists(pcoord_png_path) else pcoord_html_path,
        "importance_plot": importance_path if os.path.exists(importance_path) else importance_html_path,
        "tuning_costs": os.path.join(out_dir, "tuning_costs.json"),
        "final_selection": os.path.join(out_dir, "final_selection.txt"),
    }


# -------------------------
# Entrypoint
# -------------------------
def main():
    run_name = "rainbow_tuning"
    run_dir = os.path.join(TUNING_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ray.init(ignore_reinit_error=True)

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=20,
        grace_period=1,
        reduction_factor=2,
        time_attr="training_iteration",
    )

    tuner = tune.Tuner(
        with_resources(train_wrapper, resources={"cpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=100,
        ),
        run_config=RunConfig(
            name=run_name,
            storage_path=TUNING_DIR,
        ),
        param_space=search_space,
    )

    # record start time
    tuning_start = time.time()
    results = tuner.fit()
    tuning_end = time.time()

    # best result
    try:
        best = results.get_best_result(metric="loss", mode="min")
    except Exception as e:
        print("Exception: ", e)
        best = None

    # Save best config and trial summaries (existing behavior)
    best_config_path = os.path.join(run_dir, "best_config.json")
    try:
        with open(best_config_path, "w") as f:
            json.dump(best.config if best is not None else {}, f, indent=2)
    except Exception as e:
        print("Exception: ", e)
        pass

    # Save all trials JSONL (existing behavior) - results is iterable of result objects
    all_trials_path = os.path.join(run_dir, "all_trials.jsonl")
    try:
        with open(all_trials_path, "w") as f:
            for r in results:
                try:
                    cfg = getattr(r, "config", {}) or {}
                    metrics = getattr(r, "metrics", {}) or {}
                    record = {"config": cfg, "metrics": metrics}
                    f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print("Exception writing trial record:", e)
                    pass
    except Exception as e:
        print("Exception: ", e)
        pass

    # Post-processing: create CSV, plots, cost summary, final selection justification
    report_dir = os.path.join(run_dir, "report")
    os.makedirs(report_dir, exist_ok=True)
    artifacts = _save_trials_table_and_plots(results, run_name, report_dir, tuner_wall_clock=(tuning_end - tuning_start))

    # Save overall tuning summary (wall-clock, #trials)
    try:
        num_trials_recorded = 0
        if artifacts and os.path.exists(artifacts["trials_csv"]):
            num_trials_recorded = max(0, len(open(artifacts["trials_csv"]).read().splitlines()) - 1)
    except Exception:
        num_trials_recorded = None

    tuning_summary = {
        "run_name": run_name,
        "tuning_start_epoch": tuning_start,
        "tuning_end_epoch": tuning_end,
        "wall_clock_seconds": tuning_end - tuning_start,
        "num_samples": 3,
        "num_trials_recorded": num_trials_recorded,
        "artifacts": artifacts,
    }
    with open(os.path.join(run_dir, "tuning_summary.json"), "w") as f:
        json.dump(tuning_summary, f, indent=2)

    print("Tuning complete.")
    print("Artifacts saved under:", run_dir)
    print("Report artifacts:", artifacts)


if __name__ == "__main__":
    main()
