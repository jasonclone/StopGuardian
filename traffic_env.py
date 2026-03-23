# traffic_env.py

import traci
import numpy as np
import csv
import matplotlib.pyplot as plt


# ================================================================
# Traffic Environment Class
# ================================================================
# This class encapsulates ALL SUMO-dependent environment logic.
# Every method inside depends on instance state (tls_id, prev IDs),
# so NONE of these should be static.
# ================================================================

class TrafficEnv:
    REWARD_SCALE = 150.0
    REWARD_CLIP = 1.0

    def __init__(self, tls_id="C"):
        self.tls_id = tls_id
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()

    # ------------------------------------------------------------
    # Reset persistent tracking
    # ------------------------------------------------------------
    def reset_tracking(self):
        """Clear previous vehicle/pedestrian IDs at episode start."""
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()

    # ------------------------------------------------------------
    # Phase
    # ------------------------------------------------------------
    def get_phase(self):
        """Return current traffic light phase index."""
        return traci.trafficlight.getPhase(self.tls_id)

    # ------------------------------------------------------------
    # Full state
    # ------------------------------------------------------------
    def get_state(self):
        """Return full environment state tuple."""
        veh = self.get_vehicle_state()
        ped = self.get_ped_state()
        phase = self.get_phase()
        return (*veh[:8], ped[0], ped[1], phase)

    # ------------------------------------------------------------
    # Vehicle state
    # ------------------------------------------------------------
    def get_vehicle_state(self):
        """Return queue, wait, and throughput for vehicles."""
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
        """Return queue, wait, and throughput for pedestrians."""
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

    # ------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------
    def get_reward(self, state, prev_state=None):
        """Compute reward based on queue reduction and scaling."""
        qN, qE, qS, qW, wN, wE, wS, wW, ped_q, ped_w, phase = state
        vehicle_q = qN + qE + qS + qW
        total_q = vehicle_q + ped_q

        reward = -total_q / self.REWARD_SCALE

        if prev_state is not None:
            pqN, pqE, pqS, pqW, _, _, _, _, p_ped_q, _, _ = prev_state
            prev_total = (pqN + pqE + pqS + pqW) + p_ped_q
            reward += 0.25 * ((prev_total - total_q) / self.REWARD_SCALE)

        return float(np.clip(reward, -self.REWARD_CLIP, self.REWARD_CLIP))


# ================================================================
# Helper Functions (NOT part of the environment)
# These are intentionally OUTSIDE the class.
# They do not depend on environment state.
# ================================================================

def save_step_csv(step_rows, STEP_CSV):
    if not step_rows:
        return
    with open(STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_rows[0].keys())
        writer.writeheader()
        writer.writerows(step_rows)


def save_episode_csv(episode_rows, EPISODE_CSV):
    if not episode_rows:
        return
    with open(EPISODE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)


def plot_metrics(episode_rows, PLOT_PATH, MODE=None,RUN_ID=None):
    if not episode_rows:
        return

    eps = [r["episode"] for r in episode_rows]
    cum_rewards = [r["cumulative_reward"] for r in episode_rows]
    avg_total = [r["avg_total_queue"] for r in episode_rows]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(eps, cum_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    
    mode_str = MODE.upper() if MODE else "Mode: N/A"
    run_str = RUN_ID if RUN_ID else "RUN_ID: N/A"
    plt.title(f"Episode Metrics ({mode_str} - {run_str})")
    
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(eps, avg_total, marker="o", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Total Queue (veh + ped)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()
