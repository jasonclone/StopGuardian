# train.py
"""
Updated training script that writes results into model-separated folders:
results/<run_id>/<model>/seed_<seed>/...

Changes:
- Each seed run writes its CSVs and checkpoints into results/<run_id>/<model>/seed_<seed>
- Aggregated outputs (aggregated_episode_metrics.csv, sample_efficiency_summary.csv, policy_stability.csv)
  are written into results/<run_id>/<model> (model-level folder)
- Robust exception logging per-seed and per-run (train_errors.log)
- Keeps previous behavior (episode_start_step, sample-efficiency detection, checkpointing)
- Minimal API changes: CLI still accepts --model; results_dir now contains run_id/model/seed_*
"""

import os
import sys
import argparse
import traceback
from typing import List, Dict, Any, Tuple
import random
import numpy as np
import torch
import csv
from collections import defaultdict

# Stats
from scipy import stats

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from config import (
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

# local imports (unchanged)
from traffic_env import (
    TrafficEnv,
    save_step_csv,
    save_episode_csv,
    plot_metrics,
    make_sumo_config,
)

from model import build_rainbow_model, build_standard_model
from replay import PrioritizedReplayBuffer, UniformReplayBuffer

from learner import train_step_rainbow, train_step_standard
from action import apply_action_safe, select_action

# -----------------------
# Logging helpers
# -----------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
    return path

def log_exception(path: str, context: str, exc: Exception):
    """
    Append exception traceback to a log file at path.
    path should be a directory where train_errors.log will be created/appended.
    """
    try:
        ensure_dir(path)
        log_path = os.path.join(path, "train_errors.log")
        with open(log_path, "a") as f:
            f.write(f"\n--- Exception in {context} ---\n")
            traceback.print_exc(file=f)
            f.write("\n")
    except Exception:
        # best-effort: print to console if logging fails
        print("[TRAIN][WARN] Failed to write exception log")

# ================================================================
# Utilities: seeding and checkpointing
# ================================================================
def set_seed(seed: int = 42):
    import random as _random
    import numpy as _np
    import torch as _torch

    _random.seed(seed)
    _np.random.seed(seed)
    _torch.manual_seed(seed)
    try:
        _torch.use_deterministic_algorithms(True)
    except Exception as e:
        print("Exception", e)
        pass

    rng_states = {
        "python": _random.getstate(),
        "numpy": _np.random.get_state(),
        "torch": _torch.get_rng_state(),
        "device": DEVICE,
    }

    print(f"[INIT] Training Seeds set to {seed} | DEVICE={DEVICE}")
    return rng_states

def _capture_rng_states():
    states = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        states["torch_cuda"] = torch.cuda.get_rng_state_all()
    return states

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, replay_buffer=None):
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng_states": _capture_rng_states(),
    }
    if hasattr(optimizer, "scheduler") and optimizer.scheduler is not None:
        try:
            payload["scheduler"] = optimizer.scheduler.state_dict()
        except Exception as e:
            print("Exception", e)
            pass
    if replay_buffer is not None:
        try:
            payload["replay"] = replay_buffer.save_to_bytes()
        except Exception as e:
            print("Exception", e)
            pass
    torch.save(payload, path)

# ================================================================
# Init models & replay
# ================================================================
def init_models_and_replay(state_size, lr_decay_steps: int = None, model_type: str = "standard"):
    if model_type == "rainbow":
        online_model, online_optimizer = build_rainbow_model(
            state_size=state_size,
            lr_decay_steps=lr_decay_steps,
        )
        target_model, _ = build_rainbow_model(
            state_size=state_size,
            lr_decay_steps=lr_decay_steps,
        )
        replay_buffer = PrioritizedReplayBuffer()
        target_model.load_state_dict(online_model.state_dict())
    elif model_type == "standard":
        online_model, online_optimizer = build_standard_model(
            state_size=state_size,
            lr_decay_steps=lr_decay_steps,
        )
        target_model = None
        replay_buffer = UniformReplayBuffer()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    return online_model, target_model, replay_buffer, online_optimizer

# ================================================================
# Aggregation helpers
# ================================================================
def mean_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Returns (mean, lower_ci, upper_ci) using t-distribution.
    If len(data) == 1, returns (x, x, x).
    """
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

# ================================================================
# Main single-seed run (refactored, eval/sample-efficiency fixes)
# ================================================================
def run_single_seed(
    seed: int,
    seed_run_dir: str,
    model_type: str,
    mode: str,
    num_episodes: int,
    steps_per_episode: int,
    performance_threshold: float = None,
    sliding_window: int = 5,
) -> Dict[str, Any]:
    """
    Runs training/eval for a single seed and returns a dictionary with:
    - 'episode_rows': list of per-episode dicts
    - 'step_rows': list of per-step dicts
    - 'sample_efficiency_step': first global step where eval mean >= threshold (or None)
    - 'seed': seed
    """
    # ensure seed directory exists
    ensure_dir(seed_run_dir)

    # per-seed logging path
    try:
        log_dir = seed_run_dir
    except Exception:
        log_dir = "."

    try:
        # set seed
        set_seed(seed)

        # Prepare directories for this seed
        checkpoint_dir = os.path.join(seed_run_dir, "checkpoints")
        ensure_dir(checkpoint_dir)
        replay_dir = os.path.join(seed_run_dir, "replay")
        ensure_dir(replay_dir)

        RL_STEP_CSV = os.path.join(seed_run_dir, "rl_step_metrics.csv")
        RL_EPISODE_CSV = os.path.join(seed_run_dir, "rl_episode_metrics.csv")
        PLOT_PATH = os.path.join(seed_run_dir, "rl_plot_reward.png")
        BEST_MODEL_PATH = os.path.join(checkpoint_dir, "best_model.pth")
        LAST_MODEL_PATH = os.path.join(checkpoint_dir, "last_model.pth")
        REPLAY_PATH = os.path.join(replay_dir, "replay.pkl")
        SAMPLE_EFF_CSV = os.path.join(seed_run_dir, "sample_efficiency_seed.csv")

        # create environment
        env = TrafficEnv("C")

        # SUMO check
        if "SUMO_HOME" not in os.environ:
            raise EnvironmentError("Please declare environment variable 'SUMO_HOME'")

        tools = os.path.join(os.environ["SUMO_HOME"], "tools")
        if tools not in sys.path:
            sys.path.append(tools)

        # Start SUMO once to compute state size
        sumocfg = make_sumo_config()
        env.start(sumocfg)
        env.step()
        env.initialize_from_sumo()
        state_size = env.compute_state_size()
        env.close()

        total_updates = max((0.8 * num_episodes * steps_per_episode) - MIN_REPLAY_SIZE, -1)
        total_updates = max(1, int(total_updates))

        online_model, target_model, replay_buffer, optimizer = init_models_and_replay(
            state_size=state_size, lr_decay_steps=total_updates, model_type=model_type
        )

        # Tensorboard writer per-seed (if available)
        tb_dir = os.path.join(seed_run_dir, "tb")
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir=tb_dir)
        except Exception as e:
            print("Exception", e)
            writer = None

        step_rows: List[Dict[str, Any]] = []
        episode_rows: List[Dict[str, Any]] = []

        total_steps_global = 0
        total_grad_steps_global = 0

        best_metric = None
        sample_efficiency_step = None

        TRAIN_EPISODES = max(1, int(num_episodes * 0.8))
        EVAL_EPISODES = num_episodes - TRAIN_EPISODES

        # ensure sliding_window is sensible
        SLIDING_WINDOW = max(1, int(sliding_window))

        print(f"\n=== Starting seed={seed} (model= {model_type.upper()} | mode= {mode.upper()} | seed_dir={seed_run_dir}) ===")
        print(f"Episodes: {num_episodes} (Train: {TRAIN_EPISODES}, Eval: {EVAL_EPISODES}), Steps per episode: {steps_per_episode}")

        try:
            for ep in range(num_episodes):
                # record episode start step explicitly for correct attribution later
                episode_start_step = total_steps_global

                if mode == "train":
                    ep_mode = "train" if ep < TRAIN_EPISODES else "eval"
                else:
                    ep_mode = mode

                # Ensure model is in the correct PyTorch mode for train/eval episodes
                try:
                    if ep_mode == "eval":
                        online_model.eval()
                    else:
                        online_model.train()
                except Exception as e:
                    print("Exception", e)
                    pass

                sumocfg = make_sumo_config()
                env.start(sumocfg)
                env.step()
                env.initialize_from_sumo()
                env.reset_tracking()

                cumulative_reward = 0.0
                episode_length = 0

                vehicle_queue_hist = []
                ped_queue_hist = []
                vehicle_wait_hist = []
                ped_wait_hist = []

                vehicle_throughput = 0
                ped_throughput = 0
                switch_count = 0
                prev_phase = env.get_phase()
                episode_ok = False

                try:
                    for t in range(steps_per_episode):
                        state_raw, _ = env.get_state()

                        # use_epsilon only during training episodes and if use_epsilon is true or it is standard dqn
                        use_eps = ep_mode == "train" and (USE_EPSILON or model_type == "standard")

                        action = select_action(
                            env=env,
                            state_raw=state_raw,
                            global_step=total_steps_global,
                            mode=ep_mode,
                            online_model=online_model,
                            use_epsilon=use_eps,
                            normalize_state_torch=env.normalize_state_torch,
                        )

                        apply_action_safe(action, env, current_step_global=total_steps_global)

                        cur_phase = env.get_phase()
                        if cur_phase != prev_phase:
                            switch_count += 1
                            prev_phase = cur_phase

                        env.step()

                        try:
                            env.record_phase_start_if_changed(next_global_step=total_steps_global + 1)
                        except Exception as e:
                            print("Exception", e)
                            pass

                        phase = env.get_phase()
                        next_state_raw, next_info = env.get_state()

                        q_list = next_info.get("veh_q_list", [])
                        w_list = next_info.get("veh_w_list", [])
                        veh_thru = next_info.get("veh_thru", 0)

                        ped_queue = next_info.get("ped_q", 0)
                        ped_wait = next_info.get("ped_w", 0)
                        ped_thru = next_info.get("ped_thru", 0)

                        vehicle_throughput += veh_thru
                        ped_throughput += ped_thru

                        vehicle_queue = int(sum(q_list)) if q_list else 0
                        vehicle_wait = float(sum(w_list)) if w_list else 0.0

                        reward = env.get_reward(next_state_raw, state_raw, action)
                        cumulative_reward += reward

                        vehicle_queue_hist.append(vehicle_queue)
                        ped_queue_hist.append(ped_queue)
                        vehicle_wait_hist.append(vehicle_wait)
                        ped_wait_hist.append(ped_wait)

                        done = (t == steps_per_episode - 1)
                        episode_length += 1

                        step_rows.append(
                            {
                                "global_step": total_steps_global,
                                "episode": ep,
                                "mode": ep_mode,
                                "step_in_episode": t,
                                "vehicle_queue": vehicle_queue,
                                "vehicle_wait": vehicle_wait,
                                "ped_queue": ped_queue,
                                "ped_wait": ped_wait,
                                "ped_thru_step": ped_thru,
                                "veh_thru_step": veh_thru,
                                "phase": phase,
                                "reward": reward,
                                "loss": None,
                                "action": None,
                            }
                        )

                        if writer is not None:
                            writer.add_scalar("env/queue_vehicle", vehicle_queue, total_steps_global)
                            writer.add_scalar("env/reward_step", reward, total_steps_global)

                        # Only add to replay during training episodes
                        if ep_mode == "train":
                            replay_buffer.add(state_raw, action, reward, next_state_raw, done)

                        # Only perform training updates during training episodes
                        if ep_mode == "train":
                            if model_type == "rainbow":
                                loss = train_step_rainbow(
                                    online_model=online_model,
                                    target_model=target_model,
                                    replay_buffer=replay_buffer,
                                    optimizer=optimizer,
                                    current_step_global=total_steps_global,
                                    total_updates=total_updates,
                                    normalize_state_torch=env.normalize_state_torch,
                                )
                            else:
                                loss = train_step_standard(
                                    online_model=online_model,
                                    replay_buffer=replay_buffer,
                                    optimizer=optimizer,
                                    normalize_state_torch=env.normalize_state_torch,
                                )
                        else:
                            loss = None

                        if loss is not None:
                            total_grad_steps_global += 1
                            step_rows[-1]["loss"] = loss
                            step_rows[-1]["action"] = action
                            if writer is not None:
                                writer.add_scalar("train/loss", loss, total_grad_steps_global)
                                current_lr = optimizer.param_groups[0]["lr"]
                                writer.add_scalar("train/learning_rate", current_lr, total_grad_steps_global)
                                writer.add_scalar("train/replay_size", len(replay_buffer), total_grad_steps_global)

                        total_steps_global += 1

                    episode_ok = True

                except Exception as e:
                    print(f"[Seed {seed} Episode {ep}] Exception: {e}")
                    log_exception(log_dir, f"seed_{seed}_episode_{ep}", e)
                finally:
                    try:
                        env.close()
                    except Exception as e:
                        print("Exception", e)
                        pass

                if ep_mode == "train":
                    try:
                        replay_buffer.flush()
                    except Exception as e:
                        print("Exception", e)
                        pass

                avg_vehicle_queue = float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else -1.0
                avg_ped_queue = float(np.mean(ped_queue_hist)) if ped_queue_hist else -1.0
                avg_vehicle_wait = float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else -1.0
                avg_ped_wait = float(np.mean(ped_wait_hist)) if ped_wait_hist else -1.0

                episode_rows.append(
                    {
                        "episode": ep,
                        "mode": ep_mode,
                        "cumulative_reward": cumulative_reward,
                        "episode_length": episode_length,
                        "avg_vehicle_queue": avg_vehicle_queue,
                        "avg_ped_queue": avg_ped_queue,
                        "avg_vehicle_wait": avg_vehicle_wait,
                        "avg_ped_wait": avg_ped_wait,
                        "vehicle_total_throughput": vehicle_throughput,
                        "ped_total_throughput": ped_throughput,
                        "switch_count": switch_count,
                        "episode_ok": int(episode_ok),
                        "episode_start_step": int(episode_start_step),
                    }
                )

                # Logging
                if writer is not None:
                    writer.add_scalar("episode/cumulative_reward", cumulative_reward, ep)
                    writer.add_scalar("episode/episode_length", episode_length, ep)

                # Save per-episode CSV incrementally
                try:
                    save_episode_csv(episode_rows, RL_EPISODE_CSV)
                except Exception as e:
                    log_exception(log_dir, "save_episode_csv", e)
                try:
                    save_step_csv(step_rows, RL_STEP_CSV)
                except Exception as e:
                    log_exception(log_dir, "save_step_csv", e)

                try:
                    plot_metrics(episode_rows, PLOT_PATH, RUN_ID=seed_run_dir)
                except Exception as e:
                    # non-fatal plotting error
                    log_exception(log_dir, "plot_metrics", e)

                # Save best model and last model
                if ep_mode == "train" and episode_ok:
                    metric = cumulative_reward
                    if best_metric is None or metric > best_metric:
                        best_metric = metric
                        try:
                            save_checkpoint(BEST_MODEL_PATH, online_model, optimizer, replay_buffer=replay_buffer)
                        except Exception as e:
                            print("Exception", e)
                            log_exception(log_dir, "save_best_checkpoint", e)
                            try:
                                torch.save(online_model.state_dict(), BEST_MODEL_PATH)
                            except Exception as e2:
                                log_exception(log_dir, "torch_save_best", e2)

                    if ep % CHECKPOINT_EVERY_EPISODES == 0:
                        try:
                            save_checkpoint(LAST_MODEL_PATH, online_model, optimizer, replay_buffer=replay_buffer)
                        except Exception as e:
                            print("Exception", e)
                            log_exception(log_dir, "save_last_checkpoint", e)
                            try:
                                torch.save(online_model.state_dict(), LAST_MODEL_PATH)
                            except Exception as e2:
                                log_exception(log_dir, "torch_save_last", e2)
                            try:
                                replay_buffer.save(REPLAY_PATH)
                            except Exception as e2:
                                log_exception(log_dir, "replay_save", e2)

                # If we are in evaluation episodes and a performance threshold is provided,
                # check sample efficiency: first global step where mean eval reward over a recent sliding window >= threshold.
                if performance_threshold is not None and ep_mode == "eval":
                    try:
                        eval_rewards_all = [r["cumulative_reward"] for r in episode_rows if r["mode"] == "eval"]
                        if eval_rewards_all:
                            window_size = min(SLIDING_WINDOW, len(eval_rewards_all))
                            recent_rewards = eval_rewards_all[-window_size:]
                            mean_eval = float(np.mean(recent_rewards))
                            eval_start_step = int(max(0, episode_start_step))
                            if sample_efficiency_step is None and mean_eval >= performance_threshold:
                                sample_efficiency_step = eval_start_step
                                # persist immediately for robustness
                                try:
                                    with open(SAMPLE_EFF_CSV, "w", newline="") as f:
                                        w = csv.writer(f)
                                        w.writerow(["seed", "sample_efficiency_step", "mean_eval", "window_size"])
                                        w.writerow([seed, sample_efficiency_step, mean_eval, window_size])
                                except Exception as e:
                                    log_exception(log_dir, "write_sample_eff_seed", e)
                                print(f"[Seed {seed}] Sample efficiency reached at global step {sample_efficiency_step} (mean_eval={mean_eval:.3f} over last {window_size} evals)")
                    except Exception as e:
                        log_exception(log_dir, f"sample_eff_check_seed_{seed}_ep_{ep}", e)

            # final save
            try:
                save_checkpoint(LAST_MODEL_PATH, online_model, optimizer, replay_buffer=replay_buffer)
            except Exception as e:
                log_exception(log_dir, "final_save_checkpoint", e)
                try:
                    torch.save(online_model.state_dict(), LAST_MODEL_PATH)
                except Exception as e2:
                    log_exception(log_dir, "torch_save_last_final", e2)
                try:
                    replay_buffer.save(REPLAY_PATH)
                except Exception as e2:
                    log_exception(log_dir, "replay_save_final", e2)

        finally:
            try:
                env.close()
            except Exception as e:
                log_exception(log_dir, "env_close_final", e)
            if writer is not None:
                try:
                    writer.close()
                except Exception as e:
                    log_exception(log_dir, "writer_close", e)

        return {
            "seed": seed,
            "episode_rows": episode_rows,
            "step_rows": step_rows,
            "sample_efficiency_step": sample_efficiency_step,
            "run_dir": seed_run_dir,
        }

    except Exception as e:
        # top-level seed failure: log and re-raise so caller can decide
        log_exception(log_dir, f"run_single_seed_seed_{seed}_fatal", e)
        raise

# ================================================================
# Multi-seed orchestration and aggregation
# ================================================================
def run_multi_seed(
    seeds: List[int],
    base_model_dir: str,
    model_type: str,
    mode: str,
    num_episodes: int,
    steps_per_episode: int,
    performance_threshold: float = None,
    sliding_window: int = 5,
):
    """
    base_model_dir: results/<run_id>/<model>  (model-level folder)
    Each seed will be written to base_model_dir/seed_<seed>
    """
    ensure_dir(base_model_dir)
    per_seed_results = []
    for seed in seeds:
        seed_dir = os.path.join(base_model_dir, f"seed_{seed}")
        ensure_dir(seed_dir)
        try:
            res = run_single_seed(
                seed=seed,
                seed_run_dir=seed_dir,
                model_type=model_type,
                mode=mode,
                num_episodes=num_episodes,
                steps_per_episode=steps_per_episode,
                performance_threshold=performance_threshold,
                sliding_window=sliding_window,
            )
            per_seed_results.append(res)
        except Exception as e:
            # log and continue with other seeds
            log_exception(base_model_dir, f"run_multi_seed_seed_{seed}_failed", e)
            print(f"[WARN] Seed {seed} failed; continuing with other seeds. See train_errors.log in {base_model_dir}")

    # Aggregate per-episode across seeds
    episodes = list(range(num_episodes))
    agg_rows = []
    for ep in episodes:
        rewards = []
        lengths = []
        for res in per_seed_results:
            ep_rows = res["episode_rows"]
            if ep < len(ep_rows):
                rewards.append(ep_rows[ep]["cumulative_reward"])
                lengths.append(ep_rows[ep].get("episode_length", steps_per_episode))
        if len(rewards) == 0:
            continue
        mean_r, lower_ci, upper_ci = mean_confidence_interval(rewards, confidence=0.95)
        mean_len, len_lci, len_uci = mean_confidence_interval(lengths, confidence=0.95)
        agg_rows.append(
            {
                "episode": ep,
                "mean_cumulative_reward": mean_r,
                "ci_lower": lower_ci,
                "ci_upper": upper_ci,
                "mean_episode_length": mean_len,
                "length_ci_lower": len_lci,
                "length_ci_upper": len_uci,
                "num_seeds": len(rewards),
            }
        )

    # Sample efficiency: compute median (or mean) of sample_efficiency_step across seeds (first step where threshold reached)
    sample_steps = [r["sample_efficiency_step"] for r in per_seed_results if r["sample_efficiency_step"] is not None]
    sample_efficiency_summary = {
        "per_seed_sample_steps": {r["seed"]: r["sample_efficiency_step"] for r in per_seed_results},
        "median_sample_step": int(np.median(sample_steps)) if sample_steps else None,
        "mean_sample_step": float(np.mean(sample_steps)) if sample_steps else None,
    }

    # Policy stability: variance of returns across evaluation episodes (aggregate)
    eval_returns_by_episode = defaultdict(list)
    for res in per_seed_results:
        eval_count = 0
        for ep_row in res["episode_rows"]:
            if ep_row.get("mode") == "eval":
                eval_returns_by_episode[eval_count].append(ep_row["cumulative_reward"])
                eval_count += 1

    policy_stability = {}
    for eval_idx, vals in eval_returns_by_episode.items():
        if len(vals) >= 2:
            policy_stability[eval_idx] = {
                "variance": float(np.var(vals, ddof=1)),
                "std": float(np.std(vals, ddof=1)),
                "n": len(vals),
            }
        else:
            policy_stability[eval_idx] = {"variance": float(0.0), "std": float(0.0), "n": len(vals)}

    # Save aggregated CSVs into base_model_dir
    try:
        agg_csv = os.path.join(base_model_dir, "aggregated_episode_metrics.csv")
        with open(agg_csv, "w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "episode",
                    "mean_cumulative_reward",
                    "ci_lower",
                    "ci_upper",
                    "mean_episode_length",
                    "length_ci_lower",
                    "length_ci_upper",
                    "num_seeds",
                ],
            )
            writer.writeheader()
            for row in agg_rows:
                writer.writerow(row)
    except Exception as e:
        log_exception(base_model_dir, "write_aggregated_episode_metrics", e)

    # Save sample efficiency summary
    try:
        sample_csv = os.path.join(base_model_dir, "sample_efficiency_summary.csv")
        with open(sample_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["seed", "sample_efficiency_step"])
            for seed, step in sample_efficiency_summary["per_seed_sample_steps"].items():
                writer.writerow([seed, step])
            writer.writerow([])
            writer.writerow(["median_sample_step", sample_efficiency_summary["median_sample_step"]])
            writer.writerow(["mean_sample_step", sample_efficiency_summary["mean_sample_step"]])
    except Exception as e:
        log_exception(base_model_dir, "write_sample_efficiency_summary", e)

    # Save policy stability
    try:
        stability_csv = os.path.join(base_model_dir, "policy_stability.csv")
        with open(stability_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["episode", "variance", "std", "n"])
            writer.writeheader()
            for ep_idx, statsd in sorted(policy_stability.items()):
                writer.writerow({"episode": ep_idx, **statsd})
    except Exception as e:
        log_exception(base_model_dir, "write_policy_stability", e)

    # Plot aggregated reward curve with CI if plot_metrics supports it; otherwise leave CSV for plotting externally
    try:
        plot_metrics([r for r in agg_rows], os.path.join(base_model_dir, "agg_reward_plot.png"), RUN_ID=base_model_dir)
    except Exception as e:
        log_exception(base_model_dir, "plot_agg_reward", e)

    return {
        "per_seed_results": per_seed_results,
        "aggregated_episode_rows": agg_rows,
        "sample_efficiency_summary": sample_efficiency_summary,
        "policy_stability": policy_stability,
    }

# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "infer"])
    parser.add_argument("--run_id", type=str, default="rl")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--model", type=str, default="standard", choices=["rainbow", "standard"])
    parser.add_argument("--seeds", type=str, default="42,43", help="Comma-separated list of integer seeds, e.g. 42,43,44")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results directory")
    parser.add_argument("--performance_threshold", type=float, default=-400, help="performance threshold for sample efficiency (mean eval reward)")
    parser.add_argument("--eval_window", type=int, default=5, help="Sliding window size (number of recent eval episodes) used to compute mean for sample efficiency detection")
    args = parser.parse_args()

    MODEL = args.model
    MODE = args.mode
    RUN_ID = args.run_id
    NUM_EPISODES = args.episodes
    STEPS_PER_EPISODE = args.steps_per_episode

    # parse seeds
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # New layout: results/<run_id>/<model>/seed_<seed>
    base_run_dir = os.path.join(args.results_dir, RUN_ID, MODEL)
    ensure_dir(base_run_dir)

    # Run multi-seed experiment
    try:
        summary = run_multi_seed(
            seeds=seeds,
            base_model_dir=base_run_dir,
            model_type=MODEL,
            mode=MODE,
            num_episodes=NUM_EPISODES,
            steps_per_episode=STEPS_PER_EPISODE,
            performance_threshold=args.performance_threshold,
            sliding_window=args.eval_window,
        )
    except Exception as e:
        log_exception(base_run_dir, "run_multi_seed_top_level", e)
        raise

    # Print concise summary
    print("\n=== Multi-seed summary ===")
    print(f"Model: {MODEL}")
    print(f"Seeds: {seeds}")
    print(f"Model-level outputs saved to: {base_run_dir}")
    print(f"Aggregated episodes saved to: {os.path.join(base_run_dir, 'aggregated_episode_metrics.csv')}")
    print(f"Sample efficiency summary saved to: {os.path.join(base_run_dir, 'sample_efficiency_summary.csv')}")
    print(f"Policy stability saved to: {os.path.join(base_run_dir, 'policy_stability.csv')}")
    print("Done.")
