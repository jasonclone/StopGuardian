# baselines/b1_cal.py
# generates calibration file from baseline data to auto-tune reward normalization and weights
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import traci
import numpy as np
from traffic_env import TrafficEnv, make_sumo_config, save_step_csv

RESULTS_DIR = "results"
CALIB_DIR = os.path.join(RESULTS_DIR, "calib")
os.makedirs(CALIB_DIR, exist_ok=True)

CALIB_CSV = os.path.join(CALIB_DIR, "b1_calibration.csv")

env = TrafficEnv("C")

def run(steps=1000):
    calib_rows = []

    traci.start(make_sumo_config())
    traci.simulationStep()

    env.initialize_from_sumo()
    env.reset_tracking()

    for t in range(steps):
        state, _ = env.get_state()
        traci.simulationStep()
        next_state, info = env.get_state()

        calib_rows.append({
            "vehicle_queue": sum(info["veh_q_list"]),
            "ped_queue": info["ped_q"],
            "vehicle_wait": sum(info["veh_w_list"]),
            "ped_wait": info["ped_w"],
            "veh_thru_step": info["veh_thru"],
            "ped_thru_step": info["ped_thru"],
        })

    traci.close()

    save_step_csv(calib_rows, CALIB_CSV)
    print(f"[DONE] Calibration data saved → {CALIB_CSV}")

if __name__ == "__main__":
    run()