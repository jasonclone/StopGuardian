import sys, os, time, csv
import traci
import numpy as np
import matplotlib.pyplot as plt

HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

SUMO_CFG = "simulation/sumo/test.sumocfg"
SIM_STEPS = 10000

STEP_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_metrics.csv")
SUMMARY_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_summary.csv")

# EXACT SAME DETECTORS AS RL
DETECTORS = [
    "C_NC_1","C_NC_2","C_NC_3","C_NC_4",
    "C_EC_1","C_EC_2","C_EC_3","C_EC_4",
    "C_SC_1","C_SC_2","C_SC_3","C_SC_4",
    "C_WC_1","C_WC_2","C_WC_3","C_WC_4"
]

def get_ped_queue():
    return sum(
        1 for pid in traci.person.getIDList()
        if traci.person.getWaitingTime(pid) > 1
    )

def run_baseline():
    start_time = time.time()

    traci.start(["sumo", "-c", SUMO_CFG, "--start", "--no-step-log"])

    step_metrics = []

    cumulative_reward = 0.0
    sum_vehicle_queue = 0
    max_vehicle_queue = 0

    switch_count = 0
    prev_phase = traci.trafficlight.getPhase("C")

    vehicle_throughput = 0
    ped_throughput = 0

    for step in range(SIM_STEPS):
        # Capture IDs BEFORE stepping
        prev_v = set(traci.vehicle.getIDList())
        prev_p = set(traci.person.getIDList())

        traci.simulationStep()

        # Capture IDs AFTER stepping
        cur_v = set(traci.vehicle.getIDList())
        cur_p = set(traci.person.getIDList())

        # Spawn logging
        spawned_vehicles = list(cur_v - prev_v)
        spawned_peds = list(cur_p - prev_p)

        # Throughput
        vehicle_throughput += len(prev_v - cur_v)
        ped_throughput += len(prev_p - cur_p)

        # Vehicle queue
        q_vals = [traci.lanearea.getLastStepVehicleNumber(d) for d in DETECTORS]
        vehicle_queue = sum(q_vals)

        # Ped queue
        ped_queue = get_ped_queue()

        # Reward
        reward = -float(vehicle_queue)
        cumulative_reward += reward

        sum_vehicle_queue += vehicle_queue
        max_vehicle_queue = max(max_vehicle_queue, vehicle_queue)

        # Phase switching
        phase = traci.trafficlight.getPhase("C")
        if phase != prev_phase:
            switch_count += 1
        prev_phase = phase

        step_metrics.append({
            "step": step,
            "vehicle_queue": vehicle_queue,
            "ped_queue": ped_queue,
            "tls_phase": phase,
            "switch_count": switch_count,
            "vehicle_throughput": vehicle_throughput,
            "ped_throughput": ped_throughput,
            "reward": reward,
            "cumulative_reward": cumulative_reward,
            "spawned_vehicles": ";".join(spawned_vehicles),
            "spawned_peds": ";".join(spawned_peds),
        })


    traci.close()

    # Save step-level CSV
    with open(STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
        writer.writeheader()
        writer.writerows(step_metrics)

    # Save summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "avg_vehicle_queue","max_vehicle_queue",
            "vehicle_throughput","ped_throughput",
            "switch_count","cumulative_reward","runtime_seconds"
        ])
        writer.writeheader()
        writer.writerow({
            "avg_vehicle_queue": sum_vehicle_queue / SIM_STEPS,
            "max_vehicle_queue": max_vehicle_queue,
            "vehicle_throughput": vehicle_throughput,
            "ped_throughput": ped_throughput,
            "switch_count": switch_count,
            "cumulative_reward": cumulative_reward,
            "runtime_seconds": round(time.time() - start_time, 2)
        })

    print(f"Baseline complete in {round(time.time() - start_time, 2)} seconds")
    
    # === Visualization of Results (same as RL) ===

    # Extract arrays for plotting
    steps = [row["step"] for row in step_metrics]
    queue_lengths = [row["vehicle_queue"] for row in step_metrics]
    cumulative_rewards = [row["cumulative_reward"] for row in step_metrics]

    # Plot Cumulative Reward over Simulation Steps
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_rewards, marker='o', linestyle='-', label="Cumulative Reward")
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Fixed-Time Baseline: Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HOST_RESULTS_DIR, "b1_cumulative_reward.png"))

    # Plot Total Queue Length over Simulation Steps
    plt.figure(figsize=(10, 6))
    plt.plot(steps, queue_lengths, marker='o', linestyle='-', label="Total Queue Length")
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title("Fixed-Time Baseline: Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HOST_RESULTS_DIR, "b1_queue_length.png"))


if __name__ == "__main__":
    run_baseline()
