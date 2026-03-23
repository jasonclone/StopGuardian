# baselines/b1.py
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import argparse
import time
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt
import traci

from traffic_env import TrafficEnv, save_step_csv, save_episode_csv, plot_metrics

env = TrafficEnv("C")

# ================================================================
# Argument parsing
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps_per_episode", type=int, default=10000)
parser.add_argument("--run_id", type=str, default="b1")
args = parser.parse_args()

RUN_ID = args.run_id
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps_per_episode

# ================================================================
# Paths
# ================================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_DIR = os.path.join(RESULTS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

STEP_CSV = os.path.join(RUN_DIR, "b1_step_metrics.csv")
EPISODE_CSV = os.path.join(RUN_DIR, "b1_episode_metrics.csv")
PLOT_PATH = os.path.join(RUN_DIR, "b1_plot_reward.png")

SUMO_CFG = "simulation/sumo/test.sumocfg"


# Baseline main

def run():
    start_time = time.time()

    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    total_steps_global = 0

    totals = {
        "sum_vehicle_queue": 0.0,
        "max_vehicle_queue": 0.0,
        "sum_ped_queue": 0.0,
        "max_ped_queue": 0.0,
        "sum_vehicle_wait": 0.0,
        "max_vehicle_wait": 0.0,
        "sum_ped_wait": 0.0,
        "max_ped_wait": 0.0,
        "vehicle_throughput": 0,
        "ped_throughput": 0,
        "switch_count": 0,
        "cumulative_reward": 0.0,
    }
    try:
        for ep in range(NUM_EPISODES):
            traci.start(["sumo", "-c", SUMO_CFG, "--start", "--no-step-log"])

            # Reset trackers
            env.reset_tracking()
            
            cumulative_reward = 0.0

            vehicle_queue_hist = []
            ped_queue_hist = []
            total_queue_hist = []

            vehicle_wait_hist = []
            ped_wait_hist = []
            total_wait_hist = []

            vehicle_throughput = 0
            ped_throughput = 0
            switch_count = 0
            prev_phase = env.get_phase()

            episode_ok = False

            episode = {
                "sum_vehicle_queue": 0.0,
                "max_vehicle_queue": 0.0,
                "sum_ped_queue": 0.0,
                "max_ped_queue": 0.0,
                "sum_vehicle_wait": 0.0,
                "max_vehicle_wait": 0.0,
                "sum_ped_wait": 0.0,
                "max_ped_wait": 0.0,
                "vehicle_throughput": 0,
                "ped_throughput": 0,
                "switch_count": 0,
            }
            
            try:
                for step in range(STEPS_PER_EPISODE):

                    # =========================
                    # 1. READ CURRENT STATE
                    # =========================
                    veh_data = env.get_vehicle_state()
                    ped_data = env.get_ped_state()

                    (
                        qN, qE, qS, qW,
                        wN, wE, wS, wW,
                        veh_thru_step,
                    ) = veh_data

                    ped_queue, ped_wait, ped_thru_step = ped_data
                    phase = env.get_phase()

                    state = (qN, qE, qS, qW, ped_queue, phase)


                    # =========================
                    # PHASE TRACKING
                    # =========================
                    cur_phase = env.get_phase()
                    if cur_phase != prev_phase:
                        switch_count += 1
                        prev_phase = cur_phase

                    # =========================
                    # STEP SIMULATION
                    # =========================
                    traci.simulationStep()

                    # =========================
                    # NEXT STATE (AFTER STEP)
                    # =========================
                    veh_data = env.get_vehicle_state()
                    ped_data = env.get_ped_state()

                    (
                        qN, qE, qS, qW,
                        wN, wE, wS, wW,
                        veh_thru_step,
                    ) = veh_data

                    ped_queue, ped_wait, ped_thru_step = ped_data
                    phase = env.get_phase()

                    next_state = (qN, qE, qS, qW, ped_queue, phase)

                    # =========================
                    # 6. METRICS
                    # =========================
                    vehicle_throughput += veh_thru_step
                    ped_throughput += ped_thru_step

                    vehicle_queue = qN + qE + qS + qW
                    total_queue = vehicle_queue + ped_queue

                    vehicle_wait = wN + wE + wS + wW
                    total_wait = vehicle_wait + ped_wait

                    # =========================
                    # 7. REWARD
                    # =========================
                    reward = env.get_reward(next_state, state)
                    cumulative_reward += reward

                    # =========================
                    # 8. LOGGING
                    # =========================
                    vehicle_queue_hist.append(vehicle_queue)
                    ped_queue_hist.append(ped_queue)
                    total_queue_hist.append(total_queue)

                    vehicle_wait_hist.append(vehicle_wait)
                    ped_wait_hist.append(ped_wait)
                    total_wait_hist.append(total_wait)

                    step_rows.append({
                        "total_steps_global": total_steps_global,
                        "episode": ep,
                        "Step": step,
                        "queue_N": qN,
                        "queue_E": qE,
                        "queue_S": qS,
                        "queue_W": qW,
                        "wait_N": wN,
                        "wait_E": wE,
                        "wait_S": wS,
                        "wait_W": wW,
                        "ped_queue": ped_queue,
                        "ped_wait": ped_wait,
                        "phase": phase,
                        "reward": reward,
                    })


                    if step % 100 == 0:
                        print(
                            f"[Episode {ep+1}/{NUM_EPISODES}] "
                            f"Step {step}/{STEPS_PER_EPISODE} | "
                            f"Global Step {total_steps_global} | "
                            f"Total Queue: {total_queue:.2f} | "
                            f"Reward: {reward:.4f} | "
                            f"CumReward: {cumulative_reward:.2f}"
                        )

                    total_steps_global += 1
                    
                episode_ok = True
                
            except Exception as e:
                print(f"[Episode {ep}] Exception: {e}")
            finally:
                try:
                    traci.close()
                except:
                    pass
                    

            # ===== EPISODE METRICS =====
            avg_vehicle_queue = float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else -1.0
            avg_ped_queue = float(np.mean(ped_queue_hist)) if ped_queue_hist else -1.0
            avg_total_queue = float(np.mean(total_queue_hist)) if total_queue_hist else -1.0

            avg_vehicle_wait = float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else -1.0
            avg_ped_wait = float(np.mean(ped_wait_hist)) if ped_wait_hist else -1.0
            avg_total_wait = float(np.mean(total_wait_hist)) if total_wait_hist else -1.0

            episode_rows.append({
                "episode": ep,
                "cumulative_reward": cumulative_reward,
                "avg_vehicle_queue": avg_vehicle_queue,
                "avg_ped_queue": avg_ped_queue,
                "avg_total_queue": avg_total_queue,
                "avg_vehicle_wait": avg_vehicle_wait,
                "avg_ped_wait": avg_ped_wait,
                "avg_total_wait": avg_total_wait,
                "vehicle_throughput": vehicle_throughput,
                "ped_throughput": ped_throughput,
                "switch_count": switch_count,
                "episode_ok": int(episode_ok),
            })

            print(
                f"R={cumulative_reward:.2f} | "
                f"Q_tot={avg_total_queue:.2f} | "
                f"W_tot={avg_total_wait:.2f} | "
                f"veh_thru={vehicle_throughput} | ped_thru={ped_throughput} | "
                f"switches={switch_count}"
            )
            
            # Save outputs\
            save_step_csv(step_rows, STEP_CSV)
            save_episode_csv(episode_rows, EPISODE_CSV)
            plot_metrics(episode_rows, PLOT_PATH, RUN_ID=RUN_ID)
    finally:
        try:
            traci.close()
        except:
            pass

if __name__ == "__main__":
    run()

