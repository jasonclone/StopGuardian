# baselines/b1.py
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import csv
import argparse
import time
from typing import List, Dict, Any
from torch.utils.tensorboard import SummaryWriter  # TensorBoard

import numpy as np
import matplotlib.pyplot as plt
import traci

from traffic_env import TrafficEnv, make_sumo_config, save_step_csv, save_episode_csv, plot_metrics



# ================================================================
# Argument parsing
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps_per_episode", type=int, default=1000)
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

#* initialize traffic env
env = TrafficEnv("C")

CALIB_DIR = os.path.join(RESULTS_DIR, "calib")
os.makedirs(CALIB_DIR, exist_ok=True)

CALIB_CSV = os.path.join(CALIB_DIR, "b1_calibration.csv")

# verify calibration source csv exists (run b1_cal.py), and if so apply auto-tuning to env params based on observed data
if os.path.exists(CALIB_CSV):
    print("[INFO] Applying tuned normalization and reward params from calibration run...")
    env.calibrate_env_params(CALIB_CSV)
else:
    print("[WARNING] Calibration CSV not found. Using default env normalization and reward params.")

# Baseline main

def run():
    start_time = time.time()

    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    total_steps_global = 0

    # TensorBoard writer
    tb_dir = os.path.join(RUN_DIR, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

    try:
        for ep in range(NUM_EPISODES):
            
            traci.start(make_sumo_config())
            # ensure SUMO populates lane/TLS metadata
            traci.simulationStep() 
            # re-initialize env metadata for this new TraCI session
            env.initialize_from_sumo()
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

            
            try:
                for t in range(STEPS_PER_EPISODE):

                    # # =========================
                    # # 1. READ CURRENT STATE
                    # # =========================
                    
                    state_raw, _ = env.get_state()  # for reward calculation


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
                    

                    phase = env.get_phase()


                    #* for reward calculation
                    next_state_raw, next_info = env.get_state()
                    
                    q_list = next_info["veh_q_list"]
                    w_list = next_info["veh_w_list"]
                    veh_thru = next_info["veh_thru"]

                    ped_queue = next_info["ped_q"]
                    ped_wait = next_info["ped_w"]
                    ped_thru = next_info["ped_thru"]

                    # =========================
                    # 6. METRICS
                    # =========================
                    
                    vehicle_throughput += veh_thru
                    ped_throughput += ped_thru

                    vehicle_queue = int(sum(q_list)) if q_list else 0 # sum of all vehicle lane queues
                    total_queue = vehicle_queue + ped_queue

                    vehicle_wait = float(sum(w_list)) if w_list else 0.0 # sum of all vehicle lane waits
                    total_wait = vehicle_wait + ped_wait


                    # =========================
                    # 7. REWARD
                    # =========================
                    
                    reward = env.get_reward(next_state_raw, state_raw)
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
                        "global_step": total_steps_global,
                        "episode": ep,
                        "step_in_episode": t,
                        "vehicle_queue": vehicle_queue,
                        "vehicle_wait": vehicle_wait,
                        "ped_queue": ped_queue,
                        "ped_wait": ped_wait,
                        "ped_thru_step": ped_thru,
                        "veh_thru_step": veh_thru,
                        "phase": phase,
                        "reward": reward,
                    })
                    
                    
                    # TensorBoard: step-level environment metrics
                    writer.add_scalar("env/queue_vehicle", vehicle_queue, total_steps_global)
                    writer.add_scalar("env/queue_ped", ped_queue, total_steps_global)
                    writer.add_scalar("env/wait_vehicle", vehicle_wait, total_steps_global)
                    writer.add_scalar("env/wait_ped", ped_wait, total_steps_global)
                    writer.add_scalar("env/reward_step", reward, total_steps_global)
                    writer.add_scalar("env/veh_thru_step", veh_thru, total_steps_global)
                    writer.add_scalar("env/ped_thru_step", ped_thru, total_steps_global)



                    if t % 100 == 0:
                        print(
                            f"[Episode {ep+1}/{NUM_EPISODES}] "
                            f"Step {t}/{STEPS_PER_EPISODE} | "
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

            avg_vehicle_wait = float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else -1.0
            avg_ped_wait = float(np.mean(ped_wait_hist)) if ped_wait_hist else -1.0
            
            max_vehicle_queue = max(vehicle_queue_hist) if vehicle_queue_hist else -1
            max_ped_queue = max(ped_queue_hist) if ped_queue_hist else -1
            
            max_veh_thru_step = max([row["veh_thru_step"] for row in step_rows if row["episode"] == ep]) if step_rows else -1
            max_ped_thru_step = max([row["ped_thru_step"] for row in step_rows if row["episode"] == ep]) if step_rows else -1
           
            max_veh_wait = max(vehicle_wait_hist) if vehicle_wait_hist else -1.0
            max_ped_wait = max(ped_wait_hist) if ped_wait_hist else -1.0

            episode_rows.append({
                "episode": ep,
                "cumulative_reward": cumulative_reward,
                "avg_vehicle_queue": avg_vehicle_queue,
                "avg_ped_queue": avg_ped_queue,
                "avg_vehicle_wait": avg_vehicle_wait,
                "avg_ped_wait": avg_ped_wait,
                "max_vehicle_queue": max_vehicle_queue,
                "max_ped_queue": max_ped_queue,
                "max_veh_wait": max_veh_wait,
                "max_ped_wait": max_ped_wait,
                "max_veh_thru_step": max_veh_thru_step,
                "max_ped_thru_step": max_ped_thru_step,
                "vehicle_total_throughput": vehicle_throughput,
                "ped_total_throughput": ped_throughput,
                "switch_count": switch_count,
                "episode_ok": int(episode_ok),
            })
            
            # TensorBoard: episode-level metrics
            writer.add_scalar("episode/cumulative_reward", cumulative_reward, ep)
            writer.add_scalar("episode/avg_vehicle_queue", avg_vehicle_queue, ep)
            writer.add_scalar("episode/avg_ped_queue", avg_ped_queue, ep)
            writer.add_scalar("episode/avg_vehicle_wait", avg_vehicle_wait, ep)
            writer.add_scalar("episode/avg_ped_wait", avg_ped_wait, ep)
            writer.add_scalar("episode/max_vehicle_queue", max_vehicle_queue, ep)
            writer.add_scalar("episode/max_ped_queue", max_ped_queue, ep)
            writer.add_scalar("episode/max_veh_wait", max_veh_wait, ep)
            writer.add_scalar("episode/max_ped_wait", max_ped_wait, ep)
            writer.add_scalar("episode/max_veh_thru_step", max_veh_thru_step, ep)
            writer.add_scalar("episode/max_ped_thru_step", max_ped_thru_step, ep)
            writer.add_scalar("episode/vehicle_total_throughput", vehicle_throughput, ep)
            writer.add_scalar("episode/ped_total_throughput", ped_throughput, ep)
            writer.add_scalar("episode/switch_count", switch_count, ep)
            writer.add_scalar("episode/ok_flag", int(episode_ok), ep)

            print(
                f"R={cumulative_reward:.2f} | "
                f"veh_total_thru={vehicle_throughput} | ped_total_thru={ped_throughput} | "
                f"switches={switch_count}"
            )
            
            # Save outputs
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

