#!/usr/bin/env python3
"""
demo.py

Per-step synchronized SUMO runner (baseline + rainbow) with robust OS-level window titling.

Run:
    python demo.py --show_gui --sumo_base_port 8813 --train_seed 42 --steps_per_episode 500
"""
import os
import sys
import argparse
import glob
import multiprocessing as mp
import shutil
import time
import subprocess
import random
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
RL_DIR = os.path.join(PROJECT_ROOT, "rl")


# ---------- SUMO helpers ----------
def resolve_sumo_binary(gui: bool) -> str:
    name = "sumo-gui" if gui else "sumo"
    sumo_home = os.environ.get("SUMO_HOME")
    if sumo_home:
        bin_dir = os.path.join(sumo_home, "bin")
        candidate = os.path.join(bin_dir, f"{name}.exe" if os.name == "nt" else name)
        if os.path.exists(candidate):
            return os.path.normpath(candidate)
    found = shutil.which(name)
    if found:
        return os.path.normpath(found)
    raise RuntimeError(f"Could not locate SUMO binary '{name}'. Ensure SUMO_HOME is set or '{name}' is on PATH.")


def find_default_sumocfg() -> str:
    cur = os.path.abspath(os.path.dirname(__file__))
    while True:
        candidate = os.path.join(cur, "simulation", "sumo", "test.sumocfg")
        if os.path.exists(candidate):
            return os.path.normpath(os.path.abspath(candidate))
        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            raise RuntimeError("Could not locate simulation/sumo/test.sumocfg")
        cur = parent


def build_sumocfg_cmd(
    sumocfg_path: str,
    seed: int = 42,
    step_length: float = 0.5,
    lateral_res: int = 0,
    gui: bool = True,
) -> List[str]:
    binary = resolve_sumo_binary(gui=gui)
    cmd = [
        binary,
        "-c", str(sumocfg_path),
        "--step-length", str(step_length),
        "--lateral-resolution", str(lateral_res),
        "--random",
        "--seed", str(seed),
        "--start",
    ]
    return [str(x) for x in cmd]


# ---------- Utilities ----------
def checkpoint_for_model_seed(results_dir: str, model_name: str, seed: int) -> Optional[str]:
    """
    Locate a checkpoint for the given model and seed.

    Preference order:
      1. If a 'last_model.pth' exists in the checkpoints directory, return it.
      2. Else if a 'best_model.pth' (or common best variants) exists, return it.
      3. Else return the most recently modified .pth file.
    """
    pattern = os.path.join(results_dir, "*", model_name, f"seed_{seed}", "checkpoints", "*.pth")
    cands = glob.glob(pattern)
    if not cands:
        return None

    ck_dir = os.path.dirname(cands[0])

    last_path = os.path.join(ck_dir, "last_model.pth")
    if os.path.exists(last_path):
        return os.path.normpath(last_path)

    for name in ("best_model.pth", "best.pth", "best.pt"):
        best_path = os.path.join(ck_dir, name)
        if os.path.exists(best_path):
            return os.path.normpath(best_path)

    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return os.path.normpath(cands[0])


def compute_lr_decay_steps(num_episodes: int, steps_per_episode: int, min_replay_size: int) -> int:
    total_updates = max((0.8 * num_episodes * steps_per_episode) - min_replay_size, -1)
    total_updates = max(1, int(total_updates))
    return total_updates


def set_training_seed(seed: int):
    import random as _random
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch as _torch
        _torch.manual_seed(seed)
        try:
            _torch.use_deterministic_algorithms(True)
        except Exception:
            pass
    except Exception:
        pass


# ---------- OS window titling helpers ----------
def subprocess_check_output_safe(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode(errors="replace")
    except Exception:
        return ""


def subprocess_run_safe(cmd: List[str]):
    try:
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass


def _set_window_title_windows(pid: int, title: str, timeout: float = 5.0) -> bool:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        EnumWindows = user32.EnumWindows
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
        GetWindowThreadProcessId = user32.GetWindowThreadProcessId
        SetWindowTextW = user32.SetWindowTextW
        IsWindowVisible = user32.IsWindowVisible

        found = []

        def _enum_proc(hwnd, lParam):
            try:
                if not IsWindowVisible(hwnd):
                    return True
                pid_buf = ctypes.c_ulong()
                GetWindowThreadProcessId(hwnd, ctypes.byref(pid_buf))
                if pid_buf.value == pid:
                    found.append(hwnd)
            except Exception:
                pass
            return True

        proc_cb = EnumWindowsProc(_enum_proc)
        start = time.time()
        while time.time() - start < timeout:
            found.clear()
            EnumWindows(proc_cb, 0)
            if found:
                success = False
                for hwnd in found:
                    try:
                        SetWindowTextW(hwnd, ctypes.c_wchar_p(title))
                        success = True
                    except Exception:
                        pass
                if success:
                    return True
            time.sleep(0.1)
    except Exception:
        pass
    return False


def _set_window_title_unix(pid: int, title: str, timeout: float = 5.0) -> bool:
    try:
        out = subprocess_check_output_safe(["wmctrl", "-lp"])
        if out:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 3:
                    win_id = parts[0]
                    win_pid = parts[2]
                    try:
                        if int(win_pid) == pid:
                            subprocess_run_safe(["wmctrl", "-i", "-r", win_id, "-T", title])
                    except Exception:
                        pass
    except Exception:
        pass
    try:
        out = subprocess_check_output_safe(["xdotool", "search", "--pid", str(pid)])
        if out:
            for w in out.split():
                try:
                    subprocess_run_safe(["xdotool", "set_window", "--name", title, w])
                except Exception:
                    pass
            return True
    except Exception:
        pass
    return False


def set_window_title_for_pid(pid: int, title: str, timeout: float = 5.0) -> bool:
    try:
        if os.name == "nt":
            return _set_window_title_windows(pid, title, timeout=timeout)
        else:
            return _set_window_title_unix(pid, title, timeout=timeout)
    except Exception:
        return False


# ---------- Worker with per-step synchronization ----------
def worker_run(
    label: str,
    kind: str,  # "baseline" | "rainbow"
    ckpt_path: Optional[str],
    out_dir: str,
    steps_per_episode: int,
    sumo_port: int,
    sumo_seed: int,
    show_gui: bool,
    train_seed: Optional[int],
    step_barrier: mp.Barrier,
):
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    if os.path.isdir(RL_DIR) and RL_DIR not in sys.path:
        sys.path.insert(0, RL_DIR)

    try:
        import traci
        from traffic_env import TrafficEnv, save_step_csv, save_episode_csv
        from action import apply_action_safe, select_action
        from model import build_rainbow_model
        try:
            from config import MIN_REPLAY_SIZE
        except Exception:
            MIN_REPLAY_SIZE = 1000
    except Exception as e:
        print(f"[{label}] Import error: {e}")
        raise

    if kind == "rainbow" and train_seed is not None:
        set_training_seed(train_seed)
        print(f"[{label}] Training seed set to {train_seed}")
    else:
        print(f"[{label}] No training seed applied (baseline or train_seed=None)")

    os.makedirs(out_dir, exist_ok=True)
    step_csv = os.path.join(out_dir, f"{label}_step_metrics.csv")
    ep_csv = os.path.join(out_dir, f"{label}_episode_summary.csv")

    sumocfg_path = find_default_sumocfg()
    sumocfg_cmd = build_sumocfg_cmd(sumocfg_path, seed=sumo_seed, step_length=0.5, lateral_res=0, gui=show_gui)

    print(f"[{label}] SUMO command: {sumocfg_cmd}")
    print(f"[{label}] traci.start will use port={sumo_port}, show_gui={show_gui}")

    env = TrafficEnv("C")
    try:
        # Start SUMO via traci.start (TraCI will append --remote-port)
        traci.start(sumocfg_cmd, port=sumo_port)

        # Robust titling helper
        def _title_sumo_gui_with_retries_all(label_title: str, sumocfg_path: str, sumo_port: int, max_attempts: int = 120, delay: float = 0.25):
            # small randomized stagger to reduce race conditions when multiple workers start simultaneously
            time.sleep(random.uniform(0.0, 0.35))

            # 1) Try TraCI GUI API FIRST (best method)
            try:
                for attempt in range(max_attempts):
                    try:
                        gui_ids = []
                        try:
                            gui_ids = traci.gui.getIDList()
                        except Exception:
                            gui_ids = []

                        if gui_ids:
                            success_any = False
                            for gid in gui_ids:
                                try:
                                    traci.gui.setWindowTitle(gid, label_title)
                                    success_any = True
                                except Exception:
                                    pass

                            if success_any:
                                print(f"[{label}] set title via traci.gui for ALL gui_ids={gui_ids} on attempt {attempt+1}")
                                return True
                    except Exception:
                        pass

                    time.sleep(delay)
            except Exception:
                pass

            # Small delay before fallback (important)
            time.sleep(1.0)

            # 2) Fallback: psutil-based PID matching (require port or sumocfg path)
            try:
                import psutil
                now = time.time()
                candidates = []

                for p in psutil.process_iter(attrs=["pid", "name", "cmdline", "create_time"]):
                    try:
                        name = (p.info.get("name") or "").lower()
                        if "sumo-gui" not in name and "sumo" not in name:
                            continue

                        age = now - float(p.info.get("create_time", now))
                        cmdline_list = p.info.get("cmdline") or []
                        cmdline = " ".join(cmdline_list)

                        score = 0
                        if sumocfg_path and sumocfg_path in cmdline:
                            score += 20
                        port_tokens = [f"--remote-port {sumo_port}", f"--remote-port={sumo_port}", f"remote-port={sumo_port}", f"-p {sumo_port}"]
                        if any(tok in cmdline for tok in port_tokens):
                            score += 50
                        if str(sumo_port) in cmdline_list:
                            score += 30
                        if age < 60.0:
                            score += 5

                        if score >= 20:
                            candidates.append((score, int(p.info["pid"]), cmdline, age))
                    except Exception:
                        pass

                candidates.sort(key=lambda x: (-x[0], x[3]))

                success_any = False

                for score, pid_candidate, cmdline, age in candidates:
                    try:
                        if set_window_title_for_pid(pid_candidate, label_title, timeout=2.0):
                            print(f"[{label}] set title via psutil PID={pid_candidate} score={score} age={age} cmd={cmdline}")
                            success_any = True
                            break
                    except Exception:
                        pass

                if success_any:
                    return True

            except Exception:
                pass

            # 3) Final fallback: platform scan (use pgrep with port or tasklist)
            try:
                success_any = False

                if os.name == "nt":
                    out = subprocess_check_output_safe(["tasklist", "/FI", "IMAGENAME eq sumo-gui.exe"])
                    for line in out.splitlines():
                        if "sumo-gui" in line.lower():
                            parts = line.split()
                            try:
                                pid_candidate = int(parts[1])
                                cmdline = ""
                                try:
                                    cmdline = subprocess_check_output_safe(["wmic", "process", "where", f"ProcessId={pid_candidate}", "get", "CommandLine"]).strip()
                                except Exception:
                                    cmdline = ""
                                if str(sumo_port) in cmdline or (sumocfg_path and sumocfg_path in cmdline):
                                    if set_window_title_for_pid(pid_candidate, label_title, timeout=2.0):
                                        print(f"[{label}] set title via tasklist PID={pid_candidate} cmd={cmdline}")
                                        success_any = True
                                        break
                            except Exception:
                                pass

                else:
                    out = subprocess_check_output_safe(["pgrep", "-a", "-f", "sumo-gui"])
                    for line in out.splitlines():
                        try:
                            parts = line.split(None, 1)
                            pid_str = parts[0]
                            cmd = parts[1] if len(parts) > 1 else ""
                            if str(sumo_port) in cmd or (sumocfg_path and sumocfg_path in cmd):
                                pid_candidate = int(pid_str)
                                if set_window_title_for_pid(pid_candidate, label_title, timeout=2.0):
                                    print(f"[{label}] set title via pgrep PID={pid_candidate} cmd={cmd}")
                                    success_any = True
                                    break
                        except Exception:
                            pass

                if success_any:
                    return True

            except Exception:
                pass

            print(f"[{label}] set_window_title attempt result: False")
            return False

        # Proper titles for each tab (include label, port, SUMO seed and training seed for rainbow)
        if kind == "rainbow":
            seed_part = f", train_seed {train_seed}" if train_seed is not None else ""
            run_title = f"Rainbow DQN — {label} (port {sumo_port}, SUMO Seed {sumo_seed}{seed_part})"
        else:
            run_title = f"Fixed-Time Baseline — {label} (port {sumo_port}, SUMO Seed {sumo_seed})"

        try:
            _title_sumo_gui_with_retries_all(run_title, sumocfg_path, sumo_port, max_attempts=80, delay=0.25)
        except Exception as e:
            print(f"[{label}] titling helper failed: {e}")

        # One simulation step to populate SUMO state
        traci.simulationStep()
        env.initialize_from_sumo()
        env.reset_tracking()

        online_model = None
        if kind == "rainbow" and ckpt_path:
            # Evaluation only: do not compute LR decay or train.
            lr_decay_steps = 1
            online_model, _ = build_rainbow_model(state_size=env.compute_state_size(), lr_decay_steps=lr_decay_steps)
            try:
                import torch as _torch
                ckpt = _torch.load(ckpt_path, map_location="cpu")
                if isinstance(ckpt, dict) and "model" in ckpt:
                    online_model.load_state_dict(ckpt["model"])
                else:
                    online_model.load_state_dict(ckpt)
            except Exception as e:
                try:
                    online_model.load_state_dict(__import__("torch").load(ckpt_path, map_location="cpu"))
                except Exception as e2:
                    raise RuntimeError(f"[{label}] Failed to load checkpoint {ckpt_path}: {e}; {e2}")
            online_model.eval()
            if hasattr(online_model, "disable_noise"):
                try:
                    online_model.disable_noise()
                except Exception:
                    pass
            print(f"[{label}] Loaded model from {ckpt_path}")

        print(f"[{label}] GUI started and environment initialized — waiting at start barrier.")
        try:
            step_barrier.wait()
        except Exception as e:
            print(f"[{label}] start barrier error: {e}")

        step_rows = []
        cumulative_reward = 0.0
        vehicle_throughput = 0
        ped_throughput = 0
        vehicle_queue_hist = []
        ped_queue_hist = []
        vehicle_wait_hist = []
        ped_wait_hist = []
        switch_count = 0
        prev_phase = env.get_phase()
        total_steps = 0

        for t in range(steps_per_episode):
            state_raw, _ = env.get_state()

            if kind == "baseline":
                action = None
            else:
                action = select_action(
                    env=env,
                    state_raw=state_raw,
                    global_step=total_steps,
                    mode="eval",
                    online_model=online_model,
                    use_epsilon=False,
                    normalize_state_torch=env.normalize_state_torch,
                )
                apply_action_safe(action, env, current_step_global=total_steps)

            try:
                step_barrier.wait()
            except Exception as e:
                print(f"[{label}] step barrier broken at step {t}: {e}")
                break

            try:
                traci.simulationStep()
            except Exception as e:
                print(f"[{label}] traci.simulationStep() failed at step {t}: {e}")
                break

            try:
                env.record_phase_start_if_changed(next_global_step=total_steps + 1)
            except Exception:
                pass

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

            step_rows.append({
                "step": t,
                "phase": env.get_phase(),
                "action": int(action) if action is not None else None,
                "vehicle_queue": vehicle_queue,
                "ped_queue": ped_queue,
                "vehicle_wait": vehicle_wait,
                "ped_wait": ped_wait,
                "veh_thru_step": veh_thru,
                "ped_thru_step": ped_thru,
                "reward": reward,
            })

            total_steps += 1

        episode_summary = {
            "label": label,
            "cumulative_reward": cumulative_reward,
            "avg_vehicle_queue": float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else -1.0,
            "avg_ped_queue": float(np.mean(ped_queue_hist)) if ped_queue_hist else -1.0,
            "avg_vehicle_wait": float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else -1.0,
            "avg_ped_wait": float(np.mean(ped_wait_hist)) if ped_wait_hist else -1.0,
            "vehicle_total_throughput": int(vehicle_throughput),
            "ped_total_throughput": int(ped_throughput),
            "switch_count": int(switch_count),
            "episode_length": total_steps,
        }

        try:
            save_step_csv(step_rows, step_csv)
        except Exception:
            pd.DataFrame(step_rows).to_csv(step_csv, index=False)
        try:
            save_episode_csv([episode_summary], ep_csv)
        except Exception:
            pd.DataFrame([episode_summary]).to_csv(ep_csv, index=False)

        print(f"[{label}] Finished. Saved: {step_csv}, {ep_csv}")

    except Exception as e:
        print(f"[{label}] Worker error: {e}")
        raise
    finally:
        try:
            traci.close()
        except Exception:
            pass


# ---------- Plotting ----------
def plot_comparison_from_files(out_base: str, run_id: str, labels: List[str]) -> List[str]:
    all_step_rows = {}
    all_episode_summaries = {}
    for label in labels:
        step_csv = os.path.join(out_base, f"{label}_step_metrics.csv")
        ep_csv = os.path.join(out_base, f"{label}_episode_summary.csv")
        if os.path.exists(step_csv):
            df = pd.read_csv(step_csv)
            all_step_rows[label] = df.to_dict(orient="records")
        if os.path.exists(ep_csv):
            df2 = pd.read_csv(ep_csv)
            if not df2.empty:
                all_episode_summaries[label] = df2.iloc[0].to_dict()

    os.makedirs(out_base, exist_ok=True)
    labels_present = list(all_episode_summaries.keys())

    # Summary bar plots: cumulative reward, throughput (vehicle vs ped), avg queues (vehicle vs ped)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if labels_present:
        rewards = [all_episode_summaries[l]["cumulative_reward"] for l in labels_present]
        vehicle_throughputs = [all_episode_summaries[l].get("vehicle_total_throughput", 0) for l in labels_present]
        ped_throughputs = [all_episode_summaries[l].get("ped_total_throughput", 0) for l in labels_present]
        avg_vehicle_queues = [all_episode_summaries[l].get("avg_vehicle_queue", 0.0) for l in labels_present]
        avg_ped_queues = [all_episode_summaries[l].get("avg_ped_queue", 0.0) for l in labels_present]
    else:
        rewards = vehicle_throughputs = ped_throughputs = avg_vehicle_queues = avg_ped_queues = []

    # Cumulative reward
    axs[0].bar(labels_present, rewards, color=["C0", "C1", "C2"][:len(labels_present)])
    axs[0].set_title("Cumulative Reward (one episode)")
    axs[0].set_ylabel("Cumulative reward")

    # Throughput: grouped bars vehicle vs pedestrian
    x = np.arange(len(labels_present))
    width = 0.35
    if labels_present:
        axs[1].bar(x - width/2, vehicle_throughputs, width, label="Vehicle", color="C0")
        axs[1].bar(x + width/2, ped_throughputs, width, label="Pedestrian", color="C1")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(labels_present)
    axs[1].set_title("Throughput (one episode)")
    axs[1].set_ylabel("Passed")
    axs[1].legend()

    # Average queue lengths: grouped bars vehicle vs pedestrian
    if labels_present:
        axs[2].bar(x - width/2, avg_vehicle_queues, width, label="Avg vehicle queue", color="C0")
        axs[2].bar(x + width/2, avg_ped_queues, width, label="Avg ped queue", color="C1")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(labels_present)
    axs[2].set_title("Average Queue (one episode)")
    axs[2].set_ylabel("Avg queue length")
    axs[2].legend()

    plt.suptitle(f"Agent Comparison — {run_id}")
    out_path = os.path.join(out_base, f"{run_id}_comparison_summary.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Vehicle and pedestrian queue timeseries (single plot)
    out_ts = os.path.join(out_base, f"{run_id}_vehicle_ped_queue_timeseries.png")
    plt.figure(figsize=(10, 5))
    for label, rows in all_step_rows.items():
        df = pd.DataFrame(rows)
        if "vehicle_queue" in df.columns:
            plt.plot(df["step"].values, df["vehicle_queue"].values, label=f"{label} vehicle_queue", alpha=0.8)
        if "ped_queue" in df.columns:
            plt.plot(df["step"].values, df["ped_queue"].values, label=f"{label} ped_queue", alpha=0.8, linestyle="--")
    plt.xlabel("step")
    plt.ylabel("queue length")
    plt.title("Vehicle and Pedestrian queue over time (one episode)")
    if all_step_rows:
        plt.legend()
    plt.savefig(out_ts, dpi=150)
    plt.close()

    # Cumulative reward timeseries
    out_rts = os.path.join(out_base, f"{run_id}_cumulative_reward_timeseries.png")
    plt.figure(figsize=(10, 5))
    for label, rows in all_step_rows.items():
        df = pd.DataFrame(rows)
        if "reward" in df.columns:
            plt.plot(df["step"].values, np.cumsum(df["reward"].values), label=label, alpha=0.8)
    plt.xlabel("step")
    plt.ylabel("cumulative reward")
    plt.title("Cumulative reward over time (one episode)")
    if all_step_rows:
        plt.legend()
    plt.savefig(out_rts, dpi=150)
    plt.close()

    return [out_path, out_ts, out_rts]


def plot_combined_across_episodes(results_dir: str, run_id: str, num_episodes: int, labels: List[str]) -> List[str]:
    """
    Create combined visualizations across all episodes:
      - Aggregated summary (sum throughput, mean avg queues, mean cumulative reward)
      - Timeseries: mean +/- std across episodes for vehicle_queue and ped_queue per label
    """
    combined_dir = os.path.join(results_dir, run_id, "combined")
    os.makedirs(combined_dir, exist_ok=True)

    # Collect per-episode summaries and step series
    per_label_episode_summaries: Dict[str, List[Dict]] = {l: [] for l in labels}
    per_label_step_dfs: Dict[str, List[pd.DataFrame]] = {l: [] for l in labels}

    for ep_idx in range(num_episodes):
        ep_folder = os.path.join(results_dir, run_id, f"comparison_ep{ep_idx}")
        for label in labels:
            ep_csv = os.path.join(ep_folder, f"{label}_episode_summary.csv")
            step_csv = os.path.join(ep_folder, f"{label}_step_metrics.csv")
            if os.path.exists(ep_csv):
                try:
                    df_ep = pd.read_csv(ep_csv)
                    if not df_ep.empty:
                        per_label_episode_summaries[label].append(df_ep.iloc[0].to_dict())
                except Exception:
                    pass
            if os.path.exists(step_csv):
                try:
                    df_steps = pd.read_csv(step_csv)
                    per_label_step_dfs[label].append(df_steps)
                except Exception:
                    pass

    # Build aggregated summary table
    agg_labels = []
    agg_cum_rewards = []
    agg_vehicle_thru = []
    agg_ped_thru = []
    agg_avg_vehicle_queue = []
    agg_avg_ped_queue = []
    for label in labels:
        summaries = per_label_episode_summaries.get(label, [])
        if not summaries:
            continue
        agg_labels.append(label)
        # cumulative reward: mean across episodes
        cum_rewards = [s.get("cumulative_reward", 0.0) for s in summaries]
        agg_cum_rewards.append(float(np.mean(cum_rewards)))
        # throughput: sum across episodes (total passed across episodes)
        vehicle_thrus = [s.get("vehicle_total_throughput", 0) for s in summaries]
        ped_thrus = [s.get("ped_total_throughput", 0) for s in summaries]
        agg_vehicle_thru.append(int(np.sum(vehicle_thrus)))
        agg_ped_thru.append(int(np.sum(ped_thrus)))
        # avg queues: mean of per-episode averages
        avg_vq = [s.get("avg_vehicle_queue", 0.0) for s in summaries]
        avg_pq = [s.get("avg_ped_queue", 0.0) for s in summaries]
        agg_avg_vehicle_queue.append(float(np.mean(avg_vq)))
        agg_avg_ped_queue.append(float(np.mean(avg_pq)))

    # Summary figure
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    if agg_labels:
        axs[0].bar(agg_labels, agg_cum_rewards, color=["C0", "C1", "C2"][:len(agg_labels)])
    axs[0].set_title("Mean Cumulative Reward (across episodes)")
    axs[0].set_ylabel("Mean cumulative reward")

    x = np.arange(len(agg_labels))
    width = 0.35
    if agg_labels:
        axs[1].bar(x - width/2, agg_vehicle_thru, width, label="Vehicle total (sum episodes)", color="C0")
        axs[1].bar(x + width/2, agg_ped_thru, width, label="Ped total (sum episodes)", color="C1")
        axs[1].set_xticks(x)
        axs[1].set_xticklabels(agg_labels)
    axs[1].set_title("Total Throughput (sum across episodes)")
    axs[1].set_ylabel("Passed")
    axs[1].legend()

    if agg_labels:
        axs[2].bar(x - width/2, agg_avg_vehicle_queue, width, label="Avg vehicle queue (mean)", color="C0")
        axs[2].bar(x + width/2, agg_avg_ped_queue, width, label="Avg ped queue (mean)", color="C1")
        axs[2].set_xticks(x)
        axs[2].set_xticklabels(agg_labels)
    axs[2].set_title("Average Queue (mean across episodes)")
    axs[2].set_ylabel("Avg queue length")
    axs[2].legend()

    plt.suptitle(f"Combined Agent Comparison — {run_id} (across {num_episodes} episodes)")
    out_summary = os.path.join(combined_dir, f"{run_id}_combined_summary.png")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_summary, dpi=150)
    plt.close(fig)

    # Timeseries: for each label, compute mean and std across episodes (align by step index)
    out_ts = os.path.join(combined_dir, f"{run_id}_combined_queue_timeseries.png")
    plt.figure(figsize=(12, 6))
    any_plotted = False
    for label in labels:
        dfs = per_label_step_dfs.get(label, [])
        if not dfs:
            continue
        # align by minimum length
        min_len = min((len(df) for df in dfs), default=0)
        if min_len == 0:
            continue
        veh_arrays = np.vstack([df["vehicle_queue"].values[:min_len] for df in dfs])
        ped_arrays = np.vstack([df["ped_queue"].values[:min_len] for df in dfs]) if "ped_queue" in dfs[0].columns else None

        steps = np.arange(min_len)
        veh_mean = np.mean(veh_arrays, axis=0)
        veh_std = np.std(veh_arrays, axis=0)
        plt.plot(steps, veh_mean, label=f"{label} vehicle mean", alpha=0.9)
        plt.fill_between(steps, veh_mean - veh_std, veh_mean + veh_std, alpha=0.2)

        if ped_arrays is not None:
            ped_mean = np.mean(ped_arrays, axis=0)
            ped_std = np.std(ped_arrays, axis=0)
            plt.plot(steps, ped_mean, label=f"{label} ped mean", alpha=0.9, linestyle="--")
            plt.fill_between(steps, ped_mean - ped_std, ped_mean + ped_std, alpha=0.15)

        any_plotted = True

    if any_plotted:
        plt.xlabel("step")
        plt.ylabel("queue length")
        plt.title(f"Mean Vehicle and Pedestrian queue over time (across {num_episodes} episodes)")
        plt.legend()
        plt.savefig(out_ts, dpi=150)
    else:
        # create an empty placeholder image to indicate no data
        plt.text(0.5, 0.5, "No timeseries data available across episodes", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_ts, dpi=150)
    plt.close()

    # Combined cumulative reward timeseries (mean across episodes)
    out_rts = os.path.join(combined_dir, f"{run_id}_combined_cumulative_reward_timeseries.png")
    plt.figure(figsize=(12, 6))
    any_plotted = False
    for label in labels:
        dfs = per_label_step_dfs.get(label, [])
        if not dfs:
            continue
        min_len = min((len(df) for df in dfs), default=0)
        if min_len == 0:
            continue
        reward_arrays = np.vstack([np.cumsum(df["reward"].values[:min_len]) for df in dfs])
        mean_reward = np.mean(reward_arrays, axis=0)
        std_reward = np.std(reward_arrays, axis=0)
        steps = np.arange(min_len)
        plt.plot(steps, mean_reward, label=f"{label} mean cumulative reward", alpha=0.9)
        plt.fill_between(steps, mean_reward - std_reward, mean_reward + std_reward, alpha=0.2)
        any_plotted = True

    if any_plotted:
        plt.xlabel("step")
        plt.ylabel("cumulative reward")
        plt.title(f"Mean cumulative reward over time (across {num_episodes} episodes)")
        plt.legend()
        plt.savefig(out_rts, dpi=150)
    else:
        plt.text(0.5, 0.5, "No reward timeseries data available across episodes", ha="center", va="center")
        plt.axis("off")
        plt.savefig(out_rts, dpi=150)
    plt.close()

    return [out_summary, out_ts, out_rts]


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--run_id", type=str, default="parallel_compare")
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    parser.add_argument("--show_gui", action="store_true", default=True)
    parser.add_argument("--sumo_base_port", type=int, default=8813)
    parser.add_argument("--train_seed", type=int, default=42)
    parser.add_argument("--num_episodes", type=int, default=2, help="Number of demo episodes to run (each uses a different SUMO seed)")
    parser.add_argument("--sumo_seed_base", type=int, default=42, help="Base SUMO seed; each episode will use sumo_seed_base + episode_index")
    args = parser.parse_args()

    results_dir = args.results_dir
    run_id = args.run_id
    steps_per_episode = args.steps_per_episode
    show_gui = args.show_gui
    base_port = args.sumo_base_port
    train_seed = args.train_seed
    num_episodes = max(1, int(args.num_episodes))
    sumo_seed_base = args.sumo_seed_base

    print(f"[MAIN] Python: {sys.executable}")
    print(f"[MAIN] SUMO_HOME: {os.environ.get('SUMO_HOME')}")
    print(f"[MAIN] PATH contains sumo-gui: {shutil.which('sumo-gui')}")
    print(f"[MAIN] Working dir: {os.getcwd()}")

    if "SUMO_HOME" not in os.environ:
        raise EnvironmentError("Please declare environment variable 'SUMO_HOME' before running this script.")

    seed = 42
    ckpt_rainbow = checkpoint_for_model_seed(results_dir, "rainbow", seed)
    if ckpt_rainbow:
        print("[INFO] Found rainbow checkpoint:", ckpt_rainbow)
    else:
        print("[WARN] No rainbow checkpoint found under seed_42; rainbow will be skipped.")

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass

    # Run episodes
    for ep_idx in range(num_episodes):
        sumo_seed = sumo_seed_base + ep_idx
        out_base = os.path.join(results_dir, run_id, f"comparison_ep{ep_idx}")
        os.makedirs(out_base, exist_ok=True)

        print(f"[MAIN] Starting episode {ep_idx} with SUMO seed {sumo_seed} -> outputs: {out_base}")

        workers = []
        workers.append(("baseline_fixed", "baseline", None, base_port))
        if ckpt_rainbow:
            workers.append(("rainbow", "rainbow", ckpt_rainbow, base_port + 1))

        num_workers = len(workers)
        if num_workers == 0:
            print("No workers configured. Skipping episode.")
            continue

        step_barrier = mp.Barrier(num_workers)

        procs = []
        for label, kind, ckpt, port in workers:
            p = mp.Process(
                target=worker_run,
                args=(
                    label,
                    kind,
                    ckpt,
                    out_base,
                    steps_per_episode,
                    port,
                    sumo_seed,
                    show_gui,
                    train_seed if kind == "rainbow" else None,
                    step_barrier,
                ),
                daemon=False,
            )
            p.start()
            print(f"[MAIN] Started worker {label} (kind={kind}) on SUMO port {port} (pid={p.pid})")
            procs.append((label, p))

        time.sleep(1.0)
        print(f"[MAIN] Episode {ep_idx}: All workers started. Parent not participating in per-step barrier; workers will synchronize themselves.")

        for label, p in procs:
            p.join()
            print(f"[MAIN] Episode {ep_idx}: Worker {label} finished with exitcode={p.exitcode}")

        labels = [label for label, _ in procs]
        print(f"[MAIN] Episode {ep_idx}: Generating comparison plots")
        plot_files = plot_comparison_from_files(out_base, f"{run_id}_ep{ep_idx}", labels)
        print(f"[MAIN] Episode {ep_idx}: Plots saved to:", out_base)
        for f in plot_files:
            print(" -", f)

    # After all episodes, create combined visualizations
    # Determine labels from the last episode folder that exists (fallback)
    sample_ep_folder = os.path.join(results_dir, run_id, "comparison_ep0")
    labels_for_combined = []
    if os.path.exists(sample_ep_folder):
        for f in glob.glob(os.path.join(sample_ep_folder, "*_episode_summary.csv")):
            lbl = os.path.basename(f).replace("_episode_summary.csv", "")
            labels_for_combined.append(lbl)

    if not labels_for_combined:
        # fallback: use default names
        labels_for_combined = ["baseline_fixed"]
        if ckpt_rainbow:
            labels_for_combined.append("rainbow")

    print("[MAIN] Generating combined plots across episodes")
    combined_files = plot_combined_across_episodes(results_dir, run_id, num_episodes, labels_for_combined)
    print("[MAIN] Combined plots saved to:", os.path.join(results_dir, run_id, "combined"))
    for f in combined_files:
        print(" -", f)

    print("[MAIN] All episodes completed.")


if __name__ == "__main__":
    main()
