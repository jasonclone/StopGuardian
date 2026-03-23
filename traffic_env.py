# traffic_env.py

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import traci

from rl.config import (
    REWARD_SCALE,
    REWARD_CLIP,
    MAX_VEH_QUEUE,
    MAX_PED_QUEUE,
)

# ================================================================
# Traffic Environment Class
# ================================================================

class TrafficEnv:


    def __init__(self, tls_id="C"):
        self.tls_id = tls_id
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()
        
        # bind config constants to instance
        self.REWARD_SCALE = REWARD_SCALE
        self.REWARD_CLIP = REWARD_CLIP
        self.MAX_VEH_QUEUE = MAX_VEH_QUEUE
        self.MAX_PED_QUEUE = MAX_PED_QUEUE

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
    def get_state(self, normalized=False):
        veh = self.get_vehicle_state()
        ped = self.get_ped_state()
        phase = self.get_phase()

        qN, qE, qS, qW, _, _, _, _, _ = veh
        ped_q, _, _ = ped

        state = (qN, qE, qS, qW, ped_q, phase)

        if normalized:
            return self.normalize_state(state)

        return state

    # ------------------------------------------------------------
    # Vehicle state
    # ------------------------------------------------------------
    def get_vehicle_state(self):
        current = set(traci.vehicle.getIDList())

        new = current - self.prev_veh_ids
        left = self.prev_veh_ids - current
        throughput = len(left)

        qN = qE = qS = qW = 0
        wN = wE = wS = wW = 0.0

        for vid in current:
            if vid in new:
                continue

            try:
                lane = traci.vehicle.getLaneID(vid)
                wt = traci.vehicle.getWaitingTime(vid)
                speed = traci.vehicle.getSpeed(vid)

                if lane.endswith("_0"):
                    continue
                if wt <= 1:
                    continue

                if lane.startswith("NC"):
                    wN += wt
                    if speed < 0.1: qN += 1
                elif lane.startswith("EC"):
                    wE += wt
                    if speed < 0.1: qE += 1
                elif lane.startswith("SC"):
                    wS += wt
                    if speed < 0.1: qS += 1
                elif lane.startswith("WC"):
                    wW += wt
                    if speed < 0.1: qW += 1

            except:
                pass

        self.prev_veh_ids = current

        return (qN, qE, qS, qW, wN, wE, wS, wW, throughput)

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

    def normalize_state(self, s):
        """
        Normalize state into a flat vector:
        [qN_norm, qE_norm, qS_norm, qW_norm, ped_norm, one_hot_phase...]
        """
        qN, qE, qS, qW, ped_q, phase = s

        qN_n = normalize_scalar(qN, self.MAX_VEH_QUEUE)
        qE_n = normalize_scalar(qE, self.MAX_VEH_QUEUE)
        qS_n = normalize_scalar(qS, self.MAX_VEH_QUEUE)
        qW_n = normalize_scalar(qW, self.MAX_VEH_QUEUE)
        ped_n = normalize_scalar(ped_q, self.MAX_PED_QUEUE)

        phase_vec = one_hot_phase(phase, self.get_num_phases())

        return np.concatenate([
            np.array([qN_n, qE_n, qS_n, qW_n, ped_n], dtype=np.float32),
            phase_vec
        ])


    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    def get_reward(self, state, prev_state=None):
        qN, qE, qS, qW, ped_q, _ = state
        vehicle_q = qN + qE + qS + qW
        total_q = vehicle_q + ped_q

        reward = -total_q / self.REWARD_SCALE

        if prev_state is not None:
            pqN, pqE, pqS, pqW, p_ped_q, _ = prev_state
            prev_total = (pqN + pqE + pqS + pqW) + p_ped_q
            reward += 0.25 * ((prev_total - total_q) / self.REWARD_SCALE)

        return float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))


# ================================================================
# Helper Functions (NOT part of the environment)
# ================================================================

def make_sumo_config(
    cfg_path="simulation/sumo/test.sumocfg",
    step_length="0.10",
    delay="1000",
    lateral_res="0",
    gui=False,
):
    binary = "sumo-gui" if gui else "sumo"
    return [
        binary,
        "-c", cfg_path,
        "--step-length", str(step_length),
        "--delay", str(delay),
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
