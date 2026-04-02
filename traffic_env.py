# traffic_env.py

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import traci


# ================================================================
# Traffic Environment Class
# ================================================================

class TrafficEnv:


    def __init__(self, tls_id="C"):
        self.tls_id = tls_id
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()
        
        # Track last switch step to enforce minimum green time if needed in future
        self.last_switch_step = 0
        
        # reward normalization constants (based on baseline observations)
        self.MAX_VEH_QUEUE = 100
        self.MAX_PED_QUEUE = 20
        self.MAX_VEH_THRU = 6.0 
        self.MAX_PED_THRU = 2.0
        self.MAX_VEH_WAIT = 6000.0
        self.MAX_PED_WAIT = 1500.0
        # reward weights
        self.REWARD_ABS_QUEUE_PENALTY_WEIGHT = 0.05
        self.VEH_Q_WEIGHT = 3.0
        self.PED_Q_WEIGHT = 2.5
        self.VEH_WAIT_WEIGHT = 0.01
        self.PED_WAIT_WEIGHT = 0.002
        self.VEH_THRU_WEIGHT = 0.3
        self.PED_THRU_WEIGHT = 0.1
        self.REWARD_CLIP = 3.0
        # initialized from get_state() but set here for clarity (tracking vars)
        self.veh_thru = 0
        self.ped_thru = 0
        self.veh_wait = 0
        self.ped_wait = 0
        
    def initialize_from_sumo(self):
        """
        Call this after traci.start() and traci.simulationStep().
        It sets a deterministic lane order from SUMO lanes.
        """
        try:
            lanes = []
            

            for lane in traci.lane.getIDList():
                edge = traci.lane.getEdgeID(lane)
                
                # keep only edges that END at the intersection (incoming)
                if edge.endswith("C"):
                    # remove pedestrian + internal lanes
                    if not lane.startswith(":") and not lane.endswith("_0"):
                        lanes.append(lane)

            self.veh_lane_order = sorted(lanes)
            
            print(f"[INIT] Vehicle Lane count: {len(self.veh_lane_order)}")
        except Exception:
            # fallback: keep existing _lane_order if any, or leave None
            raise RuntimeError("Failed to initialize lane order from SUMO. Ensure traci is connected and simulation has started.")

    # ------------------------------------------------------------
    # Reset persistent tracking
    # ------------------------------------------------------------
    def reset_tracking(self):
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()
        
        
        
    #* Auto-tuning functions to calibrate reward normalization, and weights based on observed data
    def calibrate_env_params(self, csv_path):
        #* call tune max vals before weights since weights depend on reward deltas which depend on max vals for normalization
        auto_tune_max_vals(self, csv_path)
        auto_tune_weights(self, csv_path)
        
        
    def compute_state_size(self):
        # must be called after traci.start() and initialize_from_sumo()
        dummy_state, _ = self.get_state()
        dummy_tensor = self.normalize_state_torch(dummy_state)
        self.state_size = dummy_tensor.shape[0]
        return self.state_size


    # ------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------
    def get_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def get_num_phases(self):
        program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        return len(program.phases)


    def get_state(self):
        # --- raw data ---
        veh = self.get_vehicle_state()
        ped_q, ped_w, ped_thru,= self.get_ped_state()

        veh_thru = veh[-1]

        # save throughput for reward
        self.veh_thru = veh_thru
        self.ped_thru = ped_thru

        phase = self.get_phase()

        # --- unpack vehicle ---
        if len(veh) >= 1:
            n = (len(veh) - 1) // 2
            q_list = veh[:n]
            w_list = veh[n:2*n]
        else:
            q_list, w_list = [], []

        # save wait times for reward
        self.veh_wait = sum(w_list)
        self.ped_wait = ped_w

        # --- RL STATE ---
        state = q_list + [ped_q, phase]

        # --- EXTRA INFO (for logging) ---
        info = {
            "veh_q_list": q_list,
            "veh_w_list": w_list,
            "veh_thru": veh_thru,
            "ped_q": ped_q,
            "ped_w": ped_w,
            "ped_thru": ped_thru,
            "phase": phase,
        }

        return state, info


    def get_vehicle_state(self):
        current = set(traci.vehicle.getIDList())

        new = current - self.prev_veh_ids
        left = self.prev_veh_ids - current
        throughput = len(left)

        lane_q = {}
        lane_w = {}

        for vid in current:
            if vid in new:
                continue

            try:
                lane = traci.vehicle.getLaneID(vid)
                wt   = traci.vehicle.getWaitingTime(vid)
                speed = traci.vehicle.getSpeed(vid)

                if lane.endswith("_0"):
                    continue
                if wt <= 1:
                    continue

                if lane not in lane_q:
                    lane_q[lane] = 0
                    lane_w[lane] = 0.0

                if speed < 0.1:
                    lane_q[lane] += 1
                    lane_w[lane] += wt

            except Exception:
                pass

        self.prev_veh_ids = current

        # Initialize stable lane order on first call
        if not hasattr(self, "veh_lane_order"):
            raise RuntimeError("Vehicle lane order not initialized. Call initialize_from_sumo() first.")

        # Ensure every lane in _lane_order has an entry (pad zeros if missing)
        for l in self.veh_lane_order:
            if l not in lane_q:
                lane_q[l] = 0
                lane_w[l] = 0.0

        q_list = [lane_q[l] for l in self.veh_lane_order]
        w_list = [lane_w[l] for l in self.veh_lane_order]

        return q_list + w_list + [throughput]



    # ------------------------------------------------------------
    # Pedestrian state
    # ------------------------------------------------------------
    def get_ped_state(self):
        current = set(traci.person.getIDList())

        new = current - self.prev_ped_ids
        left = self.prev_ped_ids - current
        throughput = len(left)

        ped_q = 0
        ped_w = 0.0

        for pid in current:
            if pid in new:
                continue
            try:
                wt = traci.person.getWaitingTime(pid)
                speed = traci.person.getSpeed(pid)

                if wt > 1:
                    ped_w += wt
                    if speed < 0.1:
                        ped_q += 1
            except:
                pass

        self.prev_ped_ids = current

        return ped_q, ped_w, throughput
    
    
    def normalize_state_torch(self, states):
        # if input is numpy array, convert to tensor (assumes caller wants torch output and normalization on GPU)
        if not torch.is_tensor(states):
            states = torch.as_tensor(states, dtype=torch.float32)

        single_input = False
        if states.dim() == 1:
            states = states.unsqueeze(0)
            single_input = True

        veh_vals = states[:, :-2]
        ped_q = states[:, -2]
        phase = states[:, -1].long()

        veh_norm = veh_vals / self.MAX_VEH_QUEUE
        ped_norm = (ped_q / self.MAX_PED_QUEUE).unsqueeze(1)

        num_phases = self.get_num_phases()

        phase_onehot = torch.zeros(
            (states.size(0), num_phases),
            device=states.device,
            dtype=states.dtype
        )
        phase_onehot.scatter_(1, phase.unsqueeze(1), 1.0)

        out = torch.cat([veh_norm, ped_norm, phase_onehot], dim=1)

        if single_input:
            return out.squeeze(0)

        return out
    
    
    def get_reward(self, state, prev_state=None):
        # state = [veh_lane_vals..., ped_q, phase]
        *veh_vals, ped_q, _ = state

        # -----------------------------
        # 1. Normalize queues (PURE 0–1)
        # -----------------------------
        veh_total = sum(veh_vals)
        veh_norm = min(veh_total / self.MAX_VEH_QUEUE, 1.0)
        ped_norm = min(ped_q / self.MAX_PED_QUEUE, 1.0)

        # build a single congestion signal from vehicle + pedestrian queues
        # we reuse reward weights to reflect relative importance (vehicles vs pedestrians),
        # but normalize them so this stays a bounded [0,1] signal (not dependent on raw weight scale)
        w_v = self.VEH_Q_WEIGHT
        w_p = self.PED_Q_WEIGHT
        total = w_v + w_p
        # the total congestion signal is a weighted combination  (veh and ped reward weights) of vehicle and pedestrian congestion, normalized to [0,1]
        total_q_norm = (w_v / total) * veh_norm + (w_p / total) * ped_norm

        if prev_state is None:
            return 0.0

        *prev_veh_vals, prev_ped_q, _ = prev_state
        prev_veh_total = sum(prev_veh_vals)

        prev_veh_norm = min(prev_veh_total / self.MAX_VEH_QUEUE, 1.0)
        prev_ped_norm = min(prev_ped_q / self.MAX_PED_QUEUE, 1.0)

        # -----------------------------
        # 2. Queue improvement delta reward (WEIGHTED HERE)
        # -----------------------------
        queue_reward = (
            self.VEH_Q_WEIGHT * (prev_veh_norm - veh_norm) +
            self.PED_Q_WEIGHT * (prev_ped_norm - ped_norm)
        )

        # absolute queue penalty (use normalized signal only and penalty grows superlinearly to discourage large queues)
        queue_penalty = self.REWARD_ABS_QUEUE_PENALTY_WEIGHT * (total_q_norm ** 1.5)

        # -----------------------------
        # 3. Throughput reward
        # -----------------------------
        veh_thru_norm = min(self.veh_thru / self.MAX_VEH_THRU, 1.0)
        ped_thru_norm = min(self.ped_thru / self.MAX_PED_THRU, 1.0)

        # reward for a veh or ped completing their route, using normalized throughput signals and weighted
        throughput_reward = (
            self.VEH_THRU_WEIGHT * veh_thru_norm +
            self.PED_THRU_WEIGHT * ped_thru_norm
        )

        # scale down throughput reward under high congestion to encourage clearing queues before maximizing throughput
        throughput_scale = 1.0 / (1.0 + total_q_norm)
        throughput_reward *= max(0.0, throughput_scale)

        # -----------------------------
        # 4. Wait-time penalty
        # -----------------------------
        veh_wait_norm = min(self.veh_wait / self.MAX_VEH_WAIT, 1.0)
        ped_wait_norm = min(self.ped_wait / self.MAX_PED_WAIT, 1.0)

        # wait penalty grows with total wait time and is weighted. This encourages the agent to minimize wait times, especially under congestion.
        wait_penalty = (
            self.VEH_WAIT_WEIGHT * veh_wait_norm +
            self.PED_WAIT_WEIGHT * ped_wait_norm
        )

        # -----------------------------
        # 5. Final reward
        # -----------------------------
        # the reward is a combination of queue improvement, throughput, and wait time penalties, then is clipped
        reward = queue_reward + throughput_reward - (queue_penalty + wait_penalty)
        reward = float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))

        return reward


# ================================================================
# Helper Functions (NOT part of the environment)
# ================================================================

def make_sumo_config(
    cfg_path=None,
    step_length="0.50",
    lateral_res="0",
    gui=False,
):
    # Start from this file's directory
    cur = os.path.abspath(os.path.dirname(__file__))

    # Walk upward until we find the simulation/sumo folder
    while True:
        candidate = os.path.join(cur, "simulation", "sumo", "test.sumocfg")
        if os.path.exists(candidate):
            PROJECT_ROOT = cur
            break

        parent = os.path.abspath(os.path.join(cur, ".."))
        if parent == cur:
            raise RuntimeError("Could not locate simulation/sumo/test.sumocfg")
        cur = parent


    if cfg_path is None:
        cfg_path = os.path.join(
            PROJECT_ROOT,
            "simulation",
            "sumo",
            "test.sumocfg"
        )

    cfg_path = os.path.normpath(os.path.abspath(cfg_path))

    binary = "sumo-gui" if gui else "sumo"

    return [
        binary,
        "-c", cfg_path,
        "--step-length", str(step_length),
        "--lateral-resolution", str(lateral_res),
    ]


def one_hot_phase(phase_idx: int, num_phases: int) -> np.ndarray:
    vec = np.zeros(num_phases, dtype=np.float32)
    if 0 <= phase_idx < num_phases:
        vec[phase_idx] = 1.0
    return vec



def save_step_csv(step_rows, path):
    if not step_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_rows[0].keys())
        writer.writeheader()
        writer.writerows(step_rows)


def save_episode_csv(episode_rows, path):
    if not episode_rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)


def plot_metrics(episode_rows, path, MODE=None, RUN_ID=None):
    if not episode_rows:
        return

    eps = [r["episode"] for r in episode_rows]
    cum_rewards = [r["cumulative_reward"] for r in episode_rows]
    avg_veh_q = [r["avg_vehicle_queue"] for r in episode_rows]
    avg_ped_q = [r["avg_ped_queue"] for r in episode_rows]

    plt.figure(figsize=(10, 6))

    # Reward plot
    plt.subplot(2, 1, 1)
    plt.plot(eps, cum_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    mode_str = MODE.upper() if MODE else "Mode: N/A"
    run_str = RUN_ID if RUN_ID else "RUN_ID: N/A"
    plt.title(f"Episode Metrics ({mode_str} - {run_str})")
    plt.grid(True)

    # avg veh and ped Queue plot
    plt.subplot(2, 1, 2)
    plt.plot(eps, avg_veh_q, marker="o", label="Avg Vehicle Queue")
    plt.plot(eps, avg_ped_q, marker="o", label="Avg Pedestrian Queue")
    plt.xlabel("Episode")
    plt.ylabel("Avg Queues")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()





def detect_warmup_index(df, cols=None, window_size=30, activity_frac=0.2, eps=1e-6, min_rows_after=50):
    """
    Return the first index considered steady state.
    - cols: list of columns to monitor; if None, uses common columns.
    - window_size: number of rows in sliding window.
    - activity_frac: fraction of rows in window that must be 'active'.
    - eps: threshold for considering a numeric value active.
    - min_rows_after: require at least this many rows after the detected index.
    """
    import numpy as np

    if cols is None:
        cols = ["vehicle_queue", "ped_queue", "vehicle_wait", "ped_wait", "veh_thru_step", "ped_thru_step"]

    cols = [c for c in cols if c in df.columns]
    n = len(df)
    if n == 0 or len(cols) == 0:
        return 0

    # activity per row: True if any monitored column exceeds eps
    activity = np.zeros(n, dtype=bool)
    for c in cols:
        series = df[c].fillna(0).to_numpy()
        activity |= (series > eps)

    # If window_size is small or dataset short, fall back to first active row
    if window_size <= 1 or n <= window_size:
        idxs = np.where(activity)[0]
        start = int(idxs[0]) if len(idxs) > 0 else 0
    else:
        # moving-window fraction of active rows
        cumsum = np.concatenate([[0], activity.cumsum()])
        window_active = (cumsum[window_size:] - cumsum[:-window_size])
        frac_active = window_active / float(window_size)
        candidates = np.where(frac_active >= activity_frac)[0]
        start = int(candidates[0]) if len(candidates) > 0 else 0

    # ensure enough rows remain after start; if not, back off so min_rows_after remain
    if n - start < min_rows_after:
        start = max(0, n - min_rows_after)

    return start


def detect_warmup_by_consecutive(df, cols=None, consec=5, eps=1e-6):
    """
    Alternative warmup detector: first index with `consec` consecutive active rows.
    Use this for very sparse or bursty traffic.
    """
    import numpy as np

    if cols is None:
        cols = ["vehicle_queue", "ped_queue", "vehicle_wait", "ped_wait", "veh_thru_step", "ped_thru_step"]
    cols = [c for c in cols if c in df.columns]
    n = len(df)
    if n == 0 or len(cols) == 0:
        return 0

    activity = np.zeros(n, dtype=bool)
    for c in cols:
        activity |= (df[c].fillna(0).to_numpy() > eps)

    run = 0
    for i, a in enumerate(activity):
        run = run + 1 if a else 0
        if run >= consec:
            return i - consec + 1
    return 0


def auto_tune_max_vals(env, csv_path, percentile=90, margin=1.1,
                       warmup_detector_kwargs=None, fallback_keep_any_nonzero=True,
                       outlier_clip_percentile=99.0, verbose=True):
    """
    Compute robust normalization constants from csv_path and set them on env.

    - Uses detect_warmup_index to remove warmup rows by default.
    - Optionally caps extreme outliers in wait columns before computing percentiles.
    - Default strategy favors typical steady behavior: percentile=90, margin=1.1, outlier cap=99.0.
    - Ensures throughput maxima are at least 1.0 and falls back safely if filtering removes all rows.
    - Prints diagnostics when verbose=True.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    if warmup_detector_kwargs is None:
        warmup_detector_kwargs = {"window_size": 30, "activity_frac": 0.2, "eps": 1e-6, "min_rows_after": 50}

    # detect warmup and slice
    start_idx = detect_warmup_index(df, **warmup_detector_kwargs)
    df_filtered = df.iloc[start_idx:].copy()

    # fallback: keep any rows with nonzero signals if filtered set is empty
    if df_filtered.empty and fallback_keep_any_nonzero:
        df_filtered = df[
            (df.get("vehicle_queue", 0) > 0) |
            (df.get("ped_queue", 0) > 0) |
            (df.get("vehicle_wait", 0) > 0) |
            (df.get("ped_wait", 0) > 0)
        ].copy()

    # final fallback: use full dataframe
    if df_filtered.empty:
        df_filtered = df.copy()

    # optional: cap extreme outliers for wait columns to avoid huge MAX_* values
    if outlier_clip_percentile is not None and 0 < outlier_clip_percentile < 100:
        for wait_col in ("vehicle_wait", "ped_wait"):
            if wait_col in df_filtered.columns:
                vals = df_filtered[wait_col].dropna().values
                if len(vals) > 0:
                    cap = np.percentile(vals, outlier_clip_percentile)
                    df_filtered[wait_col] = np.where(df_filtered[wait_col] > cap, cap, df_filtered[wait_col])

    # helper to compute percentile safely
    def p(col):
        if col not in df_filtered.columns:
            return 0.0
        vals = df_filtered[col].dropna().values
        if len(vals) == 0:
            return 0.0
        return float(np.percentile(vals, percentile))

    env.MAX_VEH_QUEUE = float(margin * max(1.0, p("vehicle_queue")))
    env.MAX_PED_QUEUE = float(margin * max(1.0, p("ped_queue")))

    env.MAX_VEH_WAIT = float(margin * max(1.0, p("vehicle_wait")))
    env.MAX_PED_WAIT = float(margin * max(1.0, p("ped_wait")))

    env.MAX_VEH_THRU = float(max(1.0, margin * p("veh_thru_step")))
    env.MAX_PED_THRU = float(max(1.0, margin * p("ped_thru_step")))

    if verbose:
        # diagnostics: pre/post percentiles and distribution summaries
        try:
            full_vw = df["vehicle_wait"].dropna().values if "vehicle_wait" in df.columns else []
            filt_vw = df_filtered["vehicle_wait"].dropna().values if "vehicle_wait" in df_filtered.columns else []
            full_pw = df["ped_wait"].dropna().values if "ped_wait" in df.columns else []
            filt_pw = df_filtered["ped_wait"].dropna().values if "ped_wait" in df_filtered.columns else []

            print("\n[AUTO-TUNE DIAGNOSTICS] vehicle_wait (full)  count:", len(full_vw))
            if len(full_vw) > 0:
                print("  min:", np.min(full_vw), "median:", np.median(full_vw),
                      "90th:", np.percentile(full_vw, 90), "95th:", np.percentile(full_vw, 95), "max:", np.max(full_vw))
            print("vehicle_wait (filtered) count:", len(filt_vw))
            if len(filt_vw) > 0:
                print("  min:", np.min(filt_vw), "median:", np.median(filt_vw),
                      "90th:", np.percentile(filt_vw, 90), "95th:", np.percentile(filt_vw, 95), "max:", np.max(filt_vw))
            print("ped_wait (full) count:", len(full_pw))
            if len(full_pw) > 0:
                print("  min:", np.min(full_pw), "median:", np.median(full_pw),
                      "90th:", np.percentile(full_pw, 90), "95th:", np.percentile(full_pw, 95), "max:", np.max(full_pw))
            print("ped_wait (filtered) count:", len(filt_pw))
            if len(filt_pw) > 0:
                print("  min:", np.min(filt_pw), "median:", np.median(filt_pw),
                      "90th:", np.percentile(filt_pw, 90), "95th:", np.percentile(filt_pw, 95), "max:", np.max(filt_pw))
        except Exception as e:
            print("[AUTO-TUNE DIAGNOSTICS] failed to compute detailed diagnostics:", e)

        print("\n[AUTO-TUNE] Updated normalization constants:")
        print(f"warmup start index = {start_idx}")
        print(f"rows total = {len(df)}, rows used = {len(df_filtered)}")
        print(f"MAX_VEH_QUEUE = {env.MAX_VEH_QUEUE:.2f}")
        print(f"MAX_PED_QUEUE = {env.MAX_PED_QUEUE:.2f}")
        print(f"MAX_VEH_WAIT  = {env.MAX_VEH_WAIT:.2f}")
        print(f"MAX_PED_WAIT  = {env.MAX_PED_WAIT:.2f}")
        print(f"MAX_VEH_THRU  = {env.MAX_VEH_THRU:.2f}")
        print(f"MAX_PED_THRU  = {env.MAX_PED_THRU:.2f}")


def auto_tune_weights(env, csv_path, percentile=90, margin=1.1,
                      eps=1e-8, min_weight=1e-4, max_weight=1e3,
                      reward_clip_bounds=(1.0, 10.0),
                      warmup_detector_kwargs=None, verbose=True):
    """
    Auto-tune all reward weights using normalized magnitudes from CSV.

    - VEH_Q_WEIGHT now scales with average vehicle queue delta.
    - PED_Q_WEIGHT scales relative to VEH_Q_WEIGHT.
    - *_WAIT_WEIGHT and *_THRU_WEIGHT scale inversely to their normalized magnitudes.
    - Uses percentile to compute REWARD_CLIP and caps extremes.
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_path)

    if warmup_detector_kwargs is None:
        warmup_detector_kwargs = {
            "window_size": 30,
            "activity_frac": 0.2,
            "eps": 1e-6,
            "min_rows_after": 50,
        }

    # detect warmup rows
    start_idx = detect_warmup_index(df, **warmup_detector_kwargs)
    df_filtered = df.iloc[start_idx:].copy()
    if df_filtered.empty:
        df_filtered = df.copy()

    # compute absolute deltas
    df_filtered["veh_q_delta"] = df_filtered["vehicle_queue"].diff().abs()
    df_filtered["ped_q_delta"] = df_filtered["ped_queue"].diff().abs()
    df_filtered = df_filtered.dropna().reset_index(drop=True)

    # Ensure MAX_* constants exist
    required_max_attrs = [
        "MAX_VEH_QUEUE", "MAX_PED_QUEUE",
        "MAX_VEH_WAIT", "MAX_PED_WAIT",
        "MAX_VEH_THRU", "MAX_PED_THRU"
    ]
    missing = [a for a in required_max_attrs if not hasattr(env, a)]
    if missing:
        auto_tune_max_vals(env, csv_path, percentile=percentile, margin=margin,
                           warmup_detector_kwargs=warmup_detector_kwargs, verbose=verbose)

    # helper for safe division
    def safe_div(series, denom):
        return series / (float(denom) + eps)

    # normalized series
    veh_q_norm = safe_div(df_filtered["vehicle_queue"], env.MAX_VEH_QUEUE)
    ped_q_norm = safe_div(df_filtered["ped_queue"], env.MAX_PED_QUEUE)
    veh_wait_norm = safe_div(df_filtered["vehicle_wait"], env.MAX_VEH_WAIT)
    ped_wait_norm = safe_div(df_filtered["ped_wait"], env.MAX_PED_WAIT)
    veh_thru_norm = safe_div(df_filtered["veh_thru_step"], env.MAX_VEH_THRU)
    ped_thru_norm = safe_div(df_filtered["ped_thru_step"], env.MAX_PED_THRU)

    # average magnitudes
    veh_q_mag = float(df_filtered["veh_q_delta"].mean())
    ped_q_mag = float(df_filtered["ped_q_delta"].mean())
    veh_wait_mag = float(veh_wait_norm.mean())
    ped_wait_mag = float(ped_wait_norm.mean())
    veh_thru_mag = float(veh_thru_norm.mean())
    ped_thru_mag = float(ped_thru_norm.mean())

    # VEH_Q_WEIGHT scales with magnitude
    env.VEH_Q_WEIGHT = float(np.clip(veh_q_mag * margin, min_weight, max_weight))
    # PED_Q_WEIGHT relative to VEH_Q_WEIGHT
    env.PED_Q_WEIGHT = float(np.clip((ped_q_mag / (veh_q_mag + eps)) * env.VEH_Q_WEIGHT, min_weight, max_weight))
    # Wait and throughput weights inversely scale to normalized magnitude
    env.VEH_WAIT_WEIGHT = float(np.clip(env.VEH_Q_WEIGHT / (veh_wait_mag + eps), min_weight, max_weight))
    env.PED_WAIT_WEIGHT = float(np.clip(env.VEH_Q_WEIGHT / (ped_wait_mag + eps), min_weight, max_weight))
    env.VEH_THRU_WEIGHT = float(np.clip(env.VEH_Q_WEIGHT / (veh_thru_mag + eps), min_weight, max_weight))
    env.PED_THRU_WEIGHT = float(np.clip(env.VEH_Q_WEIGHT / (ped_thru_mag + eps), min_weight, max_weight))

    # queue penalty
    total_q = 0.5 * veh_q_norm + 0.5 * ped_q_norm
    avg_congestion = float(total_q.mean()) if len(total_q) > 0 else 0.0
    env.REWARD_ABS_QUEUE_PENALTY_WEIGHT = float(np.clip(0.4 / (avg_congestion**1.5 + eps), 0.01, 100.0))

    # approximate reward magnitude for clipping
    approx_rewards = []
    for i in range(len(df_filtered)):
        r_norm = (
            env.VEH_Q_WEIGHT * (df_filtered["veh_q_delta"].iloc[i] / (env.MAX_VEH_QUEUE + eps))
            + env.PED_Q_WEIGHT * (df_filtered["ped_q_delta"].iloc[i] / (env.MAX_PED_QUEUE + eps))
            + env.VEH_THRU_WEIGHT * (df_filtered["veh_thru_step"].iloc[i] / (env.MAX_VEH_THRU + eps))
            + env.PED_THRU_WEIGHT * (df_filtered["ped_thru_step"].iloc[i] / (env.MAX_PED_THRU + eps))
            - env.VEH_WAIT_WEIGHT * (df_filtered["vehicle_wait"].iloc[i] / (env.MAX_VEH_WAIT + eps))
            - env.PED_WAIT_WEIGHT * (df_filtered["ped_wait"].iloc[i] / (env.MAX_PED_WAIT + eps))
        )
        approx_rewards.append(abs(r_norm))
    approx_rewards = np.array(approx_rewards) if len(approx_rewards) > 0 else np.array([1.0])

    clip_norm = float(np.percentile(approx_rewards, percentile) + eps)
    median_norm = float(np.median(approx_rewards) + eps)
    scale_factor = float(np.clip(1.0 / (median_norm + eps), 0.1, 10.0))
    env.REWARD_CLIP = float(np.clip(clip_norm * scale_factor, reward_clip_bounds[0], reward_clip_bounds[1]))

    if verbose:
        print("\n[AUTO-TUNE] Updated reward weights:")
        print(f"warmup start index = {start_idx}")
        print(f"rows total = {len(df)}, rows used = {len(df_filtered)}")
        print(f"VEH_Q_WEIGHT = {env.VEH_Q_WEIGHT:.4f}")
        print(f"PED_Q_WEIGHT = {env.PED_Q_WEIGHT:.4f}")
        print(f"VEH_WAIT_WEIGHT = {env.VEH_WAIT_WEIGHT:.6f}")
        print(f"PED_WAIT_WEIGHT = {env.PED_WAIT_WEIGHT:.6f}")
        print(f"VEH_THRU_WEIGHT = {env.VEH_THRU_WEIGHT:.4f}")
        print(f"PED_THRU_WEIGHT = {env.PED_THRU_WEIGHT:.4f}")
        print("\n[AUTO-TUNE] Penalty + clipping:")
        print(f"REWARD_ABS_QUEUE_PENALTY_WEIGHT = {env.REWARD_ABS_QUEUE_PENALTY_WEIGHT:.4f}")
        print(f"REWARD_CLIP = {env.REWARD_CLIP:.4f}")