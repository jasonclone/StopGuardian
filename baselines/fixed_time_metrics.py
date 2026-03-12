# baselines/fixed_time_metrics.py
import os
import time
import traci
import csv
import traceback


# Paths & Simulation Parameters
HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

# Your actual SUMO config file
SUMO_CFG = "simulation/sumo/test.sumocfg"

# One episode = 10,000 steps (1000 sec @ 0.1 step-length)
SIM_STEPS = 10000

STEP_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_metrics.csv")
SUMMARY_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_summary.csv")


# Utility Functions
def wait_for_tl(tl_id, timeout=15.0):
    """Wait until the traffic light appears in TraCI."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            if tl_id in traci.trafficlight.getIDList():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


# Generic Queue Function
def get_true_queue_generic(ids, new_ids, get_speed_fn):
    """
    Generic queue counter for vehicles or pedestrians.
    Counts entities with speed < 0.1, excluding newly spawned ones.
    """
    queue = 0
    for eid in ids:
        if eid in new_ids:
            continue
        if get_speed_fn(eid) < 0.1:
            queue += 1
    return queue


def get_vehicle_queue(lane, new_vehicles):
    veh_ids = traci.lane.getLastStepVehicleIDs(lane)
    return get_true_queue_generic(
        veh_ids,
        new_vehicles,
        lambda vid: traci.vehicle.getSpeed(vid)
    )


def get_ped_queue(ped_ids, new_peds):
    return get_true_queue_generic(
        ped_ids,
        new_peds,
        lambda pid: traci.person.getSpeed(pid)
    )


# Fixed-Time Baseline
def run_fixed_time():
    print("Starting SUMO with config:", SUMO_CFG)

    try:
        # use sumo-gui instead of sumo
        traci.start(["sumo-gui", "-c", SUMO_CFG, "--start", "--no-step-log"])
    except Exception:
        print("ERROR: Failed to start SUMO")
        traceback.print_exc()
        return

    tl_id = "C"

    if not wait_for_tl(tl_id):
        print("ERROR: Traffic light 'C' not found.")
        traci.close()
        return

    # All lanes controlled by TL C
    incoming_lanes = list(set(traci.trafficlight.getControlledLanes(tl_id)))

    if not incoming_lanes:
        print("ERROR: No incoming lanes detected.")
        traci.close()
        return

    # Filter out pedestrian lanes for vehicle metrics
    vehicle_lanes = []
    for lane in incoming_lanes:
        allowed = traci.lane.getAllowed(lane)
        if "pedestrian" not in allowed:
            vehicle_lanes.append(lane)

    try:
        logic = traci.trafficlight.getCompleteRedYellowGreenDefinition(tl_id)[0]
        num_phases = len(logic.phases)
    except Exception:
        print("ERROR: Could not read traffic light logic.")
        traci.close()
        return

    print(f"Traffic light {tl_id} | Phases: {num_phases}")
    print("Running fixed-time baseline using SUMO's default timing...")

    # Episode-level trackers
    sum_vehicle_wait = 0.0
    sum_vehicle_queue = 0.0
    max_vehicle_queue = 0

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
            prev_peds = set(traci.person.getIDList())

            # DO NOT override SUMO's timing
            traci.simulationStep()

            current_vehicles = set(traci.vehicle.getIDList())
            current_peds = set(traci.person.getIDList())

            new_vehicles = current_vehicles - prev_vehicles
            new_peds = current_peds - prev_peds

            exited_vehicles = prev_vehicles - current_vehicles
            vehicles_completed.update(exited_vehicles)

            # Vehicle Metrics
            vehicle_wait = sum(traci.lane.getWaitingTime(l) for l in vehicle_lanes)
            vehicle_queue = sum(get_vehicle_queue(l, new_vehicles) for l in vehicle_lanes)
            vehicle_count = sum(traci.lane.getLastStepVehicleNumber(l) for l in vehicle_lanes)

            # Pedestrian Metrics
            ped_ids = current_peds
            ped_wait = 0.0

            ped_queue = get_ped_queue(ped_ids, new_peds)

            for pid in ped_ids:
                try:
                    ped_wait += traci.person.getWaitingTime(pid)
                except Exception:
                    continue

            # Reward Function
            vehicle_term = vehicle_wait + vehicle_queue
            pedestrian_term = ped_wait + ped_queue
            reward = -(vehicle_term + pedestrian_term)

            # Accumulate episode stats
            sum_vehicle_wait += vehicle_wait
            sum_vehicle_queue += vehicle_queue
            max_vehicle_queue = max(max_vehicle_queue, vehicle_queue)

            sum_ped_wait += ped_wait
            sum_ped_queue += ped_queue
            max_ped_queue = max(max_ped_queue, ped_queue)

            cumulative_reward += reward

            # Save step metrics
            step_metrics.append({
                "step": step,
                "vehicle_waiting_time": vehicle_wait,
                "vehicle_queue": vehicle_queue,
                "vehicle_count": vehicle_count,
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

    # Episode Summary
    avg_vehicle_wait = sum_vehicle_wait / SIM_STEPS
    avg_vehicle_queue = sum_vehicle_queue / SIM_STEPS
    avg_ped_wait = sum_ped_wait / SIM_STEPS
    avg_ped_queue = sum_ped_queue / SIM_STEPS
    throughput = len(vehicles_completed)

    # Write Step CSV
    with open(STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
        writer.writeheader()
        writer.writerows(step_metrics)

    # Write Summary CSV
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "avg_vehicle_waiting_time",
                "avg_vehicle_queue_length",
                "max_vehicle_queue_length",
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
            "avg_vehicle_waiting_time": avg_vehicle_wait,
            "avg_vehicle_queue_length": avg_vehicle_queue,
            "max_vehicle_queue_length": max_vehicle_queue,
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
