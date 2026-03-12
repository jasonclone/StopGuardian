# baselines/fixed_time_metrics.py
import os
import time
import traci
import csv
import traceback


# Paths & Simulation Parameters
HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

# SUMO CONFIG
SUMO_CFG = "simulation/sumo/test.sumocfg"

# One episode = 10,000 steps (1000 sec @ 0.1 step-length)
SIM_STEPS = 10000
STEP_LENGTH = 0.1  # seconds per step

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


def get_vehicle_queue(lane, new_vehicles):
    """stopped vehicles with waitingTime > 0."""
    veh_ids = traci.lane.getLastStepVehicleIDs(lane)
    queue = 0
    for vid in veh_ids:
        if vid in new_vehicles:
            continue
        try:
            if traci.vehicle.getSpeed(vid) < 0.1 and traci.vehicle.getWaitingTime(vid) > 0:
                queue += 1
        except Exception:
            continue
    return queue


def get_ped_queue(ped_ids, new_peds):
    """stopped pedestrians with waitingTime > 0."""
    queue = 0
    for pid in ped_ids:
        if pid in new_peds:
            continue
        try:
            if traci.person.getSpeed(pid) < 0.1 and traci.person.getWaitingTime(pid) > 0:
                queue += 1
        except Exception:
            continue
    return queue


# Fixed-Time Baseline
def run_fixed_time():
    print("Starting SUMO with config:", SUMO_CFG)

    try:
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

    # Filter out pedestrian lanes
    vehicle_lanes = []
    for lane in incoming_lanes:
        allowed = traci.lane.getAllowed(lane)
        if "pedestrian" not in allowed:
            vehicle_lanes.append(lane)

    # Episode-level trackers
    sum_vehicle_queue = 0.0
    max_vehicle_queue = 0

    sum_ped_queue = 0.0
    max_ped_queue = 0

    cumulative_reward = 0.0

    # Accumulated waiting time tracking
    veh_accum_wait = {}
    ped_accum_wait = {}

    finished_vehicle_waits = []
    finished_ped_waits = []

    step_metrics = []

    step = 0
    try:
        while step < SIM_STEPS:

            prev_vehicles = set(traci.vehicle.getIDList())
            prev_peds = set(traci.person.getIDList())

            traci.simulationStep()

            current_vehicles = set(traci.vehicle.getIDList())
            current_peds = set(traci.person.getIDList())

            new_vehicles = current_vehicles - prev_vehicles
            new_peds = current_peds - prev_peds

            exited_vehicles = prev_vehicles - current_vehicles
            exited_peds = prev_peds - current_peds

            # Initialize new agents
            for vid in new_vehicles:
                veh_accum_wait[vid] = 0.0
            for pid in new_peds:
                ped_accum_wait[pid] = 0.0

            # Update accumulated waiting time for vehicles
            for vid in current_vehicles:
                try:
                    if traci.vehicle.getSpeed(vid) < 0.1 and traci.vehicle.getWaitingTime(vid) > 0:
                        veh_accum_wait[vid] += STEP_LENGTH
                except Exception:
                    continue

            # Update accumulated waiting time for pedestrians
            # Use SUMO's waitingTime to avoid counting spawn/arrival as waiting
            for pid in current_peds:
                try:
                    if traci.person.getWaitingTime(pid) > 0:
                        ped_accum_wait[pid] += STEP_LENGTH
                except Exception:
                    continue

            # Store final waiting times for exited vehicles
            for vid in exited_vehicles:
                finished_vehicle_waits.append(veh_accum_wait.get(vid, 0))

            # Store final waiting times for exited pedestrians
            for pid in exited_peds:
                finished_ped_waits.append(ped_accum_wait.get(pid, 0))

            # Vehicle queue metrics
            vehicle_queue = sum(get_vehicle_queue(l, new_vehicles) for l in vehicle_lanes)
            sum_vehicle_queue += vehicle_queue
            max_vehicle_queue = max(max_vehicle_queue, vehicle_queue)

            # Pedestrian queue metrics
            ped_queue = get_ped_queue(current_peds, new_peds)
            sum_ped_queue += ped_queue
            max_ped_queue = max(max_ped_queue, ped_queue)

            # Step-level totals (current accumulated waits of active agents)
            total_vehicle_wait = sum(veh_accum_wait.get(vid, 0) for vid in current_vehicles)
            total_ped_wait = sum(ped_accum_wait.get(pid, 0) for pid in current_peds)

            # Reward (matches RL env)
            reward = -(vehicle_queue + ped_queue)
            cumulative_reward += reward

            # Store full step metrics
            step_metrics.append({
                "step": step,
                "vehicle_waiting_time": total_vehicle_wait,
                "vehicle_queue": vehicle_queue,
                "vehicle_count": len(current_vehicles),
                "ped_waiting_time": total_ped_wait,
                "ped_queue": ped_queue,
                "ped_count": len(current_peds),
                "reward": reward
            })

            step += 1

    except Exception:
        print("ERROR during simulation loop")
        traceback.print_exc()
        traci.close()
        return

    traci.close()

    # Add waiting times for agents still alive at the end
    finished_vehicle_waits.extend(veh_accum_wait.values())
    finished_ped_waits.extend(ped_accum_wait.values())

    # NORMALIZED METRICS
    avg_vehicle_wait = (
        sum(finished_vehicle_waits) / len(finished_vehicle_waits)
        if finished_vehicle_waits else 0
    )
    avg_ped_wait = (
        sum(finished_ped_waits) / len(finished_ped_waits)
        if finished_ped_waits else 0
    )

    avg_vehicle_queue = sum_vehicle_queue / SIM_STEPS
    avg_ped_queue = sum_ped_queue / SIM_STEPS

    throughput = len(finished_vehicle_waits)

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
