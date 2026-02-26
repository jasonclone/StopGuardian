# baselines/fixed_time_metrics.py
import os
import time
import traci
import csv
import traceback

# -----------------------------
# Paths & Simulation Parameters
# -----------------------------

# * change to docker results directory if running in docker
HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

# * change to docker config path if running in docker
SUMO_CFG = "simulation/sumo/simulation.sumocfg"

SIM_STEPS = 1000

STEP_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_metrics.csv")
SUMMARY_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_summary.csv")


# -----------------------------
# Utility Functions
# -----------------------------
def wait_for_tl(tl_id, timeout=15.0):
    start = time.time()
    while time.time() - start < timeout:
        try:
            if tl_id in traci.trafficlight.getIDList():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


# TRUE QUEUE (excludes newly spawned vehicles)
def get_true_queue(lane, new_vehicles):
    veh_ids = traci.lane.getLastStepVehicleIDs(lane)
    queue = 0
    for vid in veh_ids:
        if vid in new_vehicles:
            continue
        if traci.vehicle.getSpeed(vid) < 0.1:
            queue += 1
    return queue


# -----------------------------
# Fixed-Time Baseline
# -----------------------------
def run_fixed_time():
    print("Starting SUMO with config:", SUMO_CFG)

    try:
        traci.start(["sumo", "-c", SUMO_CFG, "--start", "--no-step-log"])
    except Exception:
        print("ERROR: Failed to start SUMO")
        traceback.print_exc()
        return

    tl_id = "C"

    if not wait_for_tl(tl_id):
        print("ERROR: Traffic light 'C' not found.")
        traci.close()
        return

    incoming_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))
    if not incoming_lanes:
        print("ERROR: No incoming lanes detected.")
        traci.close()
        return

    try:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        num_phases = len(logic.phases)
    except Exception:
        print("ERROR: Could not read traffic light logic.")
        traci.close()
        return

    print(f"Traffic light {tl_id} | Phases: {num_phases}")
    print("Running fixed-time baseline using SUMO's default timing...")

    # -----------------------------
    # Episode-level trackers
    # -----------------------------
    sum_wait = 0.0
    sum_queue = 0.0
    max_queue = 0

    sum_ped_wait = 0.0
    sum_ped_queue = 0.0
    max_ped_queue = 0

    cumulative_reward = 0.0
    vehicles_completed = set()

    step_metrics = []

    step = 0
    try:
        while step < SIM_STEPS:
            prev_vehicles = set(traci.vehicle.getIDList())

            # DO NOT override SUMO's timing
            traci.simulationStep()

            current_vehicles = set(traci.vehicle.getIDList())
            new_vehicles = current_vehicles - prev_vehicles
            exited_vehicles = prev_vehicles - current_vehicles
            vehicles_completed.update(exited_vehicles)

            # -----------------------------
            # Vehicle metrics
            # -----------------------------
            total_wait = sum(traci.lane.getWaitingTime(l) for l in incoming_lanes)
            total_queue = sum(get_true_queue(l, new_vehicles) for l in incoming_lanes)
            total_vehicles = sum(
                traci.lane.getLastStepVehicleNumber(l) for l in incoming_lanes
            )

            # -----------------------------
            # Pedestrian metrics
            # -----------------------------
            ped_ids = traci.person.getIDList()
            ped_wait = 0.0
            ped_queue = 0

            for pid in ped_ids:
                try:
                    speed = traci.person.getSpeed(pid)
                    if speed < 0.1:
                        ped_queue += 1
                    # waiting time for persons is supported in recent SUMO;
                    # if not, this will just stay 0 and you can replace with your own logic
                    ped_wait += traci.person.getWaitingTime(pid)
                except Exception:
                    # if any person API is missing, skip gracefully
                    continue

            # -----------------------------
            # Reward (single scalar, multi-component)
            # -----------------------------
            # You can tune these weights later if you want to emphasize pedestrians more/less
            vehicle_term = total_wait + total_queue
            pedestrian_term = ped_wait + ped_queue
            reward = -(vehicle_term + pedestrian_term)

            # -----------------------------
            # Accumulate episode-level stats
            # -----------------------------
            sum_wait += total_wait
            sum_queue += total_queue
            max_queue = max(max_queue, total_queue)

            sum_ped_wait += ped_wait
            sum_ped_queue += ped_queue
            max_ped_queue = max(max_ped_queue, ped_queue)

            cumulative_reward += reward

            step_metrics.append({
                "step": step,
                "total_waiting_time": total_wait,
                "total_queue": total_queue,
                "total_vehicles": total_vehicles,
                "ped_waiting_time": ped_wait,
                "ped_queue": ped_queue,
                "ped_count": len(ped_ids),
                "reward": reward
            })

            step += 1

    except Exception:
        print("ERROR during simulation loop")
        traceback.print_exc()
        traci.close()
        return

    traci.close()

    # -----------------------------
    # Episode-level results
    # -----------------------------
    avg_wait = sum_wait / SIM_STEPS
    avg_queue = sum_queue / SIM_STEPS
    avg_ped_wait = sum_ped_wait / SIM_STEPS
    avg_ped_queue = sum_ped_queue / SIM_STEPS
    throughput = len(vehicles_completed)

    # -----------------------------
    # Write Step CSV
    # -----------------------------
    with open(STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=step_metrics[0].keys()
        )
        writer.writeheader()
        writer.writerows(step_metrics)

    # -----------------------------
    # Write Summary CSV
    # -----------------------------
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "avg_waiting_time",
                "avg_queue_length",
                "max_queue_length",
                "avg_ped_waiting_time",
                "avg_ped_queue_length",
                "max_ped_queue_length",
                "throughput",
                "cumulative_reward",
                "simulation_steps"
            ]
        )
        writer.writeheader()
        writer.writerow({
            "avg_waiting_time": avg_wait,
            "avg_queue_length": avg_queue,
            "max_queue_length": max_queue,
            "avg_ped_waiting_time": avg_ped_wait,
            "avg_ped_queue_length": avg_ped_queue,
            "max_ped_queue_length": max_ped_queue,
            "throughput": throughput,
            "cumulative_reward": cumulative_reward,
            "simulation_steps": SIM_STEPS
        })

    print("Fixed-time baseline complete")
    print("Step metrics:", STEP_CSV)
    print("Summary metrics:", SUMMARY_CSV)


if __name__ == "__main__":
    run_fixed_time()
