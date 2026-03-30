# traffic_env.py

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import traci

from rl.config import (
    MAX_VEH_QUEUE,
    MAX_PED_QUEUE,
    REWARD_ABS_PENALTY_WEIGHT,
    PED_REWARD_WEIGHT,
    VEH_REWARD_WEIGHT,
    REWARD_SCALE
)

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
        
        # bind config constants to instance
        self.MAX_VEH_QUEUE = MAX_VEH_QUEUE
        self.MAX_PED_QUEUE = MAX_PED_QUEUE
        self.REWARD_ABS_PENALTY_WEIGHT = REWARD_ABS_PENALTY_WEIGHT
        self.PED_REWARD_WEIGHT = PED_REWARD_WEIGHT
        self.VEH_REWARD_WEIGHT = VEH_REWARD_WEIGHT
        self.REWARD_SCALE = REWARD_SCALE
        
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

    # ------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------
    def get_phase(self):
        return traci.trafficlight.getPhase(self.tls_id)

    def get_num_phases(self):
        program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        return len(program.phases)

    # ------------------------------------------------------------
    # Full state
    # ------------------------------------------------------------
    # def get_state(self, normalized=False):
    #     veh = self.get_vehicle_state()
    #     ped = self.get_ped_state()
    #     phase = self.get_phase()

    #     qN, qE, qS, qW, _, _, _, _, _ = veh
    #     ped_q, _, _ = ped

    #     state = (qN, qE, qS, qW, ped_q, phase)

    #     if normalized:
    #         return self.normalize_state(state)

    #     return state
    
    def get_state(self, normalized=False):
        # veh = [q_list..., w_list..., throughput]. we dont need wait or throughput for state, just the lane queues
        veh = self.get_vehicle_state()
        ped_q, _, _ = self.get_ped_state()
        phase = self.get_phase()

        # number of lanes
        if len(veh) >= 1:
            n = (len(veh) - 1) // 2
            q_list = veh[:n]
        else:
            q_list = []

        # state contains only lane queue counts, pedestrian queue, and phase
        state = q_list + [ped_q, phase]

        if normalized:
            return self.normalize_state(state)

        return state


    # ------------------------------------------------------------
    # Vehicle state
    # ------------------------------------------------------------

    # def get_vehicle_state(self):
    #     current = set(traci.vehicle.getIDList())

    #     new = current - self.prev_veh_ids
    #     left = self.prev_veh_ids - current
    #     throughput = len(left)

    #     qN = qE = qS = qW = 0
    #     wN = wE = wS = wW = 0.0

    #     for vid in current:
    #         if vid in new:
    #             continue

    #         try:
    #             lane = traci.vehicle.getLaneID(vid)
    #             wt = traci.vehicle.getWaitingTime(vid)
    #             speed = traci.vehicle.getSpeed(vid)

    #             if lane.endswith("_0"):
    #                 continue
    #             if wt <= 1:
    #                 continue

    #             if lane.startswith("NC"):
    #                 wN += wt
    #                 if speed < 0.1: qN += 1
    #             elif lane.startswith("EC"):
    #                 wE += wt
    #                 if speed < 0.1: qE += 1
    #             elif lane.startswith("SC"):
    #                 wS += wt
    #                 if speed < 0.1: qS += 1
    #             elif lane.startswith("WC"):
    #                 wW += wt
    #                 if speed < 0.1: qW += 1

    #         except:
    #             pass

    #     self.prev_veh_ids = current

    #     return (qN, qE, qS, qW, wN, wE, wS, wW, throughput)

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

    # def normalize_state(self, s):
    #     """
    #     Normalize state into a flat vector:
    #     [qN_norm, qE_norm, qS_norm, qW_norm, ped_norm, one_hot_phase...]
    #     """
    #     qN, qE, qS, qW, ped_q, phase = s

    #     qN_n = normalize_scalar(qN, self.MAX_VEH_QUEUE)
    #     qE_n = normalize_scalar(qE, self.MAX_VEH_QUEUE)
    #     qS_n = normalize_scalar(qS, self.MAX_VEH_QUEUE)
    #     qW_n = normalize_scalar(qW, self.MAX_VEH_QUEUE)
    #     ped_n = normalize_scalar(ped_q, self.MAX_PED_QUEUE)

    #     phase_vec = one_hot_phase(phase, self.get_num_phases())

    #     return np.concatenate([
    #         np.array([qN_n, qE_n, qS_n, qW_n, ped_n], dtype=np.float32),
    #         phase_vec
    #     ])
        
    def normalize_state(self, s):
        # unpack: everything except last two are vehicle values
        *veh_vals, ped_q, phase = s

        # normalize all vehicle lane values
        veh_norm = [normalize_scalar(v, self.MAX_VEH_QUEUE) for v in veh_vals]

        ped_n = normalize_scalar(ped_q, self.MAX_PED_QUEUE)
        phase_vec = one_hot_phase(phase, self.get_num_phases())

        return np.concatenate([
            np.array(veh_norm + [ped_n], dtype=np.float32),
            phase_vec
        ])



    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    # def get_reward(self, state, prev_state=None):
    #     max_possible_queue = (4 * self.MAX_VEH_QUEUE) + self.MAX_PED_QUEUE
    #     reward = 0.0

    #     qN, qE, qS, qW, ped_q, _ = state
    #     total_q = (qN + qE + qS + qW) + ped_q # total queue is sum of all vehicle queues and pedestrian queue from current state
    #     total_q_norm = total_q / max_possible_queue # normalize current state total queue to [0, 1]


    #     # DELTA reward based on change in total queue (main reward)
    #     if prev_state is None:
    #         # initial step: no delta available, return neutral reward
    #         return 0.0

    #     pqN, pqE, pqS, pqW, p_ped_q, _ = prev_state
    #     prev_total = (pqN + pqE + pqS + pqW) + p_ped_q
    #     prev_norm = prev_total / max_possible_queue # normalize previous state total queue to [0, 1]

    #     # reward is positive if total queue decreased, negative if increased, and subtract absolute penalty based on current total queue to encourage keeping queues low
    #     reward = (prev_norm - total_q_norm) - self.REWARD_ABS_PENALTY_WEIGHT * total_q_norm # reward domain is [-1 - REWARD_ABS_PENALTY_WEIGHT, 1]


    #     return float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))
    
    def get_reward(self, state, prev_state=None):
        # state = [all veh lane vals..., ped_q, phase]
        *veh_vals, ped_q, _ = state

        # FIX: use fixed number of lanes for consistent scaling
        num_lanes = len(self.veh_lane_order)

        veh_total = sum(veh_vals)

        veh_norm = (
            veh_total / (self.MAX_VEH_QUEUE * num_lanes)
            if num_lanes > 0 else 0.0
        )

        ped_norm = ped_q / self.MAX_PED_QUEUE

        # Weighted total 
        total_q_norm = (
            self.VEH_REWARD_WEIGHT * veh_norm +
            self.PED_REWARD_WEIGHT * ped_norm
        )

        if prev_state is None:
            return 0.0

        *prev_veh_vals, prev_ped_q, _ = prev_state

        prev_veh_total = sum(prev_veh_vals)

        prev_veh_norm = (
            prev_veh_total / (self.MAX_VEH_QUEUE * num_lanes)
            if num_lanes > 0 else 0.0
        )

        prev_ped_norm = prev_ped_q / self.MAX_PED_QUEUE

        prev_total_q_norm = (
            self.VEH_REWARD_WEIGHT * prev_veh_norm +
            self.PED_REWARD_WEIGHT * prev_ped_norm
        )

        # ------------------------------------------------------------
        # Delta reward + penalty
        # ------------------------------------------------------------
        reward = (
            prev_total_q_norm - total_q_norm
        ) - self.REWARD_ABS_PENALTY_WEIGHT * total_q_norm

        reward *= self.REWARD_SCALE

        return float(reward)



# ================================================================
# Helper Functions (NOT part of the environment)
# ================================================================

def make_sumo_config(
    cfg_path="simulation/sumo/test.sumocfg",
    step_length="0.50",
    lateral_res="0",
    gui=False,
):
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


def normalize_scalar(x: float, max_val: float) -> float:
    if max_val <= 0:
        return 0.0
    return float(np.clip(x / max_val, 0.0, 1.0))



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
    avg_total = [r["avg_total_queue"] for r in episode_rows]

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

    # Queue plot
    plt.subplot(2, 1, 2)
    plt.plot(eps, avg_total, marker="o", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Total Queue (veh + ped)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(path)
    plt.close()