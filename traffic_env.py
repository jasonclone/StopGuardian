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
        
        # Track last switch step to enforce minimum phase time if needed in future
        # self.last_switch_step = 0
        
        # reward normalization constants (default until calibrated)
        self.MAX_VEH_QUEUE = 100
        self.MAX_PED_QUEUE = 20
        self.MAX_VEH_WAIT = 5500.0
        self.MAX_PED_WAIT = 1500.0
        self.MAX_VEH_THRU = 3
        self.MAX_PED_THRU = 2

        # initialized from get_state() but set here for clarity (tracking vars)
        self.veh_thru = 0
        self.ped_thru = 0
        self.veh_wait = 0
        self.ped_wait = 0
        
        # previous step's values, also initialized by get_state()
        self.prev_veh_wait = 0.0
        self.prev_ped_wait = 0.0
        self.prev_veh_thru = 0
        self.prev_ped_thru = 0

        
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


    # called every step in main training loop. also updates env variables
    def get_state(self):
        # --- raw data ---
        veh = self.get_vehicle_state()
        ped_q, ped_w, ped_thru = self.get_ped_state()

        veh_thru = veh[-1]

        # --- unpack vehicle ---
        if len(veh) >= 1:
            n = (len(veh) - 1) // 2
            q_list = veh[:n]
            w_list = veh[n:2*n]
        else:
            q_list, w_list = [], []

        # --- save previous values BEFORE updating ---
        self.prev_veh_wait = self.veh_wait
        self.prev_ped_wait = self.ped_wait
        self.prev_veh_thru = self.veh_thru
        self.prev_ped_thru = self.ped_thru

        # --- update current values ---
        self.veh_thru = veh_thru
        self.ped_thru = ped_thru
        self.veh_wait = sum(w_list)
        self.ped_wait = ped_w

        phase = self.get_phase()

        # --- RL STATE ---
        state = q_list + [ped_q, phase]

        # --- EXTRA INFO ---
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
    
    
     
    def get_reward(self, state, prev_state=None, action=None):
        *veh_vals, ped_q, phase = state

        # --- Current congestion ---
        veh_total = float(sum(veh_vals))
        veh_norm = veh_total / (self.MAX_VEH_QUEUE)
        ped_norm = float(ped_q) / (self.MAX_PED_QUEUE)
        current_congestion = veh_norm + ped_norm

        # --- Previous congestion ---
        if prev_state is not None:
            *prev_veh_vals, prev_ped_q, prev_phase = prev_state
            prev_veh_total = float(sum(prev_veh_vals))
            prev_veh_norm = prev_veh_total / (self.MAX_VEH_QUEUE)
            prev_ped_norm = float(prev_ped_q) / (self.MAX_PED_QUEUE)
            prev_congestion = prev_veh_norm + prev_ped_norm
        else:
            prev_congestion = current_congestion

        # --- Delta reward ---
        delta = prev_congestion - current_congestion

        # --- Base reward ---
        reward = -current_congestion + delta

        # --- Switch penalty  ---
        if action == 1:
            alpha = 65.0
            # penalty for switching depends on how bad current action was. 
            penalty = alpha * max(0, -delta)
            reward -= penalty

        return float(reward)


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

