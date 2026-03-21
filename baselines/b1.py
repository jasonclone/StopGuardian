# baselines/b1.py
import os
import time
import csv
import argparse

import traci
import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# Fixed-time baseline for SUMO Traffic Signal Control
# - Uses SAME detector set and naming conventions as rl.py
# - Logs step-level metrics with RL-aligned identifiers:
#   global_step, episode, mode, step_in_episode,
#   vehicle_queue, ped_queue,
#   wait_N, wait_E, wait_S, wait_W,
#   total_vehicle_wait, ped_wait_time,
#   vehicle_wait, ped_wait, total_wait, total_queue,
#   tls_phase, switch_count,
#   vehicle_throughput, ped_throughput, total_throughput,
#   reward, cumulative_reward,
#   spawned_vehicles, spawned_peds
# ================================================================

HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

SUMO_CFG = "simulation/sumo/test.sumocfg"

STEP_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_metrics.csv")
SUMMARY_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_summary.csv")

EPISODE_CSV = os.path.join(HOST_RESULTS_DIR, "fixed_time_episode_metrics.csv")


# EXACT SAME DETECTORS AS RL
VEHICLE_DETECTORS = [
    "C_NC_1", "C_NC_2", "C_NC_3", "C_NC_4",
    "C_EC_1", "C_EC_2", "C_EC_3", "C_EC_4",
    "C_SC_1", "C_SC_2", "C_SC_3", "C_SC_4",
    "C_WC_1", "C_WC_2", "C_WC_3", "C_WC_4",
]

# ================================================================
# Argument parsing (episodes + steps per episode)
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--episodes",
    type=int,
    default=10,
    help="Number of baseline episodes to run",
)
parser.add_argument(
    "--steps_per_episode",
    type=int,
    default=10000,
    help="Number of simulation steps per episode",
)
args = parser.parse_args()

BASELINE_EPISODES = args.episodes
BASELINE_STEPS_PER_EPISODE = args.steps_per_episode


# ================================================================
# Helper functions (aligned with rl.py naming)
# ================================================================

def get_vehicle_queue_length(detector_id: str) -> int:
    """Vehicle queue length on a lanearea detector (vehicles only)."""
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_vehicle_wait_time(detector_id: str) -> float:
    """
    Fallback waiting-time approximation for SUMO versions
    that do NOT support lanearea.getWaitingTime().
    """
    try:
        return traci.lanearea.getWaitingTime(detector_id)
    except Exception:
        # Fallback: count halting vehicles (speed < 0.1)
        return float(traci.lanearea.getLastStepHaltingNumber(detector_id))


def get_vehicle_wait_time_for_prefix(det_prefix: str) -> float:
    return sum(
        get_vehicle_wait_time(f"{det_prefix}_{i}")
        for i in range(1, 5)
    )



def get_pedestrian_queue_count() -> int:
    """Number of pedestrians currently waiting (waiting time > 1s)."""
    return sum(
        1 for pid in traci.person.getIDList()
        if traci.person.getWaitingTime(pid) > 1.0
    )


def get_pedestrian_total_wait_time() -> float:
    """Total pedestrian waiting time over all pedestrians (seconds)."""
    return sum(traci.person.getWaitingTime(pid) for pid in traci.person.getIDList())


def get_current_phase(tls_id: str = "C") -> int:
    """Current traffic light phase index."""
    return traci.trafficlight.getPhase(tls_id)


# ================================================================
# Baseline main
# ================================================================

def run_baseline():
    start_time = time.time()

    # Global accumulators across all episodes
    global_step = 0
    step_metrics = []
    
    episode_rewards = []
    
    episode_rows = []



    total_sum_vehicle_queue = 0.0
    total_max_vehicle_queue = 0.0

    total_sum_pedestrian_queue = 0.0
    total_max_pedestrian_queue = 0.0

    total_sum_vehicle_wait = 0.0
    total_max_vehicle_wait = 0.0

    total_sum_pedestrian_wait = 0.0
    total_max_pedestrian_wait = 0.0

    total_vehicle_throughput = 0
    total_pedestrian_throughput = 0

    total_switch_count = 0
    total_cumulative_reward = 0.0

    for ep in range(BASELINE_EPISODES):
        traci.start(["sumo", "-c", SUMO_CFG, "--start", "--no-step-log"])

        cumulative_reward_episode = 0.0

        # Per-episode accumulators
        episode_sum_vehicle_queue = 0.0
        episode_max_vehicle_queue = 0.0

        episode_sum_pedestrian_queue = 0.0
        episode_max_pedestrian_queue = 0.0

        episode_sum_vehicle_wait = 0.0
        episode_max_vehicle_wait = 0.0

        episode_sum_pedestrian_wait = 0.0
        episode_max_pedestrian_wait = 0.0

        episode_switch_count = 0
        tls_phase = get_current_phase("C")

        episode_vehicle_throughput = 0
        episode_pedestrian_throughput = 0

        prev_vehicle_queue_total = None
        prev_pedestrian_queue = None

        for step_in_episode in range(BASELINE_STEPS_PER_EPISODE):
            # IDs before step for throughput and spawn logging
            prev_vehicle_ids = set(traci.vehicle.getIDList())
            prev_pedestrian_ids = set(traci.person.getIDList())

            traci.simulationStep()

            # IDs after step
            cur_vehicle_ids = set(traci.vehicle.getIDList())
            cur_pedestrian_ids = set(traci.person.getIDList())

            spawned_vehicles = list(cur_vehicle_ids - prev_vehicle_ids)
            spawned_pedestrians = list(cur_pedestrian_ids - prev_pedestrian_ids)

            # Throughput (vehicles/peds that left the network)
            episode_vehicle_throughput += len(prev_vehicle_ids - cur_vehicle_ids)
            episode_pedestrian_throughput += len(prev_pedestrian_ids - cur_pedestrian_ids)

            # Vehicle queue (sum over all detectors)
            vehicle_queue_total = sum(
                get_vehicle_queue_length(det_id) for det_id in VEHICLE_DETECTORS
            )

            # Pedestrian queue
            pedestrian_queue = get_pedestrian_queue_count()

            # Vehicle waiting times per approach
            vehicle_wait_N = get_vehicle_wait_time_for_prefix("C_NC")
            vehicle_wait_E = get_vehicle_wait_time_for_prefix("C_EC")
            vehicle_wait_S = get_vehicle_wait_time_for_prefix("C_SC")
            vehicle_wait_W = get_vehicle_wait_time_for_prefix("C_WC")

            total_vehicle_wait = (
                vehicle_wait_N + vehicle_wait_E + vehicle_wait_S + vehicle_wait_W
            )

            # Pedestrian waiting time (aggregate)
            pedestrian_wait_time = get_pedestrian_total_wait_time()

            # For RL-aligned naming
            vehicle_wait = total_vehicle_wait
            ped_wait = pedestrian_wait_time
            total_wait = vehicle_wait + ped_wait

            total_queue = vehicle_queue_total + pedestrian_queue

            # ============================================================
            # Reward (vehicles + pedestrians), aligned EXACTLY with RL:
            #   - base = -total_queue
            #   - shaping = 0.25 * (prev_total_queue - total_queue)
            #   - scale by 1/REWARD_SCALE
            #   - clip to [-1, 1]
            # ============================================================

            REWARD_SCALE = 50.0
            REWARD_CLIP = 1.0

            base_reward = -float(total_queue)

            shaping = 0.0
            if prev_vehicle_queue_total is not None and prev_pedestrian_queue is not None:
                prev_total_queue = prev_vehicle_queue_total + prev_pedestrian_queue
                shaping = 0.25 * (prev_total_queue - total_queue)

            reward = (base_reward + shaping) / REWARD_SCALE
            reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))

            cumulative_reward_episode += reward

            prev_vehicle_queue_total = vehicle_queue_total
            prev_pedestrian_queue = pedestrian_queue

            # Update per-episode stats
            episode_sum_vehicle_queue += vehicle_queue_total
            episode_max_vehicle_queue = max(episode_max_vehicle_queue, vehicle_queue_total)

            episode_sum_pedestrian_queue += pedestrian_queue
            episode_max_pedestrian_queue = max(episode_max_pedestrian_queue, pedestrian_queue)

            episode_sum_vehicle_wait += total_vehicle_wait
            episode_max_vehicle_wait = max(episode_max_vehicle_wait, total_vehicle_wait)

            episode_sum_pedestrian_wait += pedestrian_wait_time
            episode_max_pedestrian_wait = max(episode_max_pedestrian_wait, pedestrian_wait_time)

            # Phase and switching
            current_phase = get_current_phase("C")
            if current_phase != tls_phase:
                episode_switch_count += 1
            tls_phase = current_phase

            # Step-level logging (RL-aligned fields)
            step_metrics.append(
                {
                    "global_step": global_step,
                    "episode": ep,
                    "mode": "baseline",
                    "step_in_episode": step_in_episode,
                    "vehicle_queue": vehicle_queue_total,
                    "ped_queue": pedestrian_queue,
                    "Vehicle_wait_N": vehicle_wait_N,
                    "Vehicle_wait_E": vehicle_wait_E,
                    "Vehicle_wait_S": vehicle_wait_S,
                    "Vehicle_wait_W": vehicle_wait_W,
                    "total_vehicle_wait": total_vehicle_wait,
                    "ped_wait_time": pedestrian_wait_time,
                    "vehicle_wait": vehicle_wait,
                    "ped_wait": ped_wait,
                    "total_wait": total_wait,
                    "total_queue": total_queue,
                    "tls_phase": tls_phase,
                    "switch_count": episode_switch_count,
                    "vehicle_throughput": episode_vehicle_throughput,
                    "ped_throughput": episode_pedestrian_throughput,
                    "total_throughput": (
                        episode_vehicle_throughput + episode_pedestrian_throughput
                    ),
                    "reward": reward,
                    "cumulative_reward": cumulative_reward_episode,
                    "spawned_vehicles": ";".join(spawned_vehicles),
                    "spawned_peds": ";".join(spawned_pedestrians),
                }
            )

            global_step += 1

        traci.close()
        
        # === Per-episode cumulative reward tracking ===
        episode_rewards.append(cumulative_reward_episode)
        
        episode_rows.append({
            "episode": ep,
            "cumulative_reward": cumulative_reward_episode,
            "avg_vehicle_queue": episode_sum_vehicle_queue / BASELINE_STEPS_PER_EPISODE,
            "avg_ped_queue": episode_sum_pedestrian_queue / BASELINE_STEPS_PER_EPISODE,
            "avg_total_queue": (episode_sum_vehicle_queue + episode_sum_pedestrian_queue) / BASELINE_STEPS_PER_EPISODE,
            "avg_vehicle_wait": episode_sum_vehicle_wait / BASELINE_STEPS_PER_EPISODE,
            "avg_ped_wait": episode_sum_pedestrian_wait / BASELINE_STEPS_PER_EPISODE,
            "avg_total_wait": (episode_sum_vehicle_wait + episode_sum_pedestrian_wait) / BASELINE_STEPS_PER_EPISODE,
            "vehicle_throughput": episode_vehicle_throughput,
            "ped_throughput": episode_pedestrian_throughput,
            "switch_count": episode_switch_count,
        })
        
        # Write per-episode CSV immediately for each episode (overwrites each time, but ensures we have intermediate results even if something crashes later)
        with open(EPISODE_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)



        # === Update per-episode cumulative reward plot ===
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(episode_rewards)), episode_rewards, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        plt.title("Fixed-Time Baseline: Cumulative Reward per Episode")
        plt.grid(True)
        plt.savefig(os.path.join(HOST_RESULTS_DIR, "b1_episode_cumulative_reward.png"))
        plt.close()

        # Fold episode stats into global totals
        total_sum_vehicle_queue += episode_sum_vehicle_queue
        total_max_vehicle_queue = max(total_max_vehicle_queue, episode_max_vehicle_queue)

        total_sum_pedestrian_queue += episode_sum_pedestrian_queue
        total_max_pedestrian_queue = max(
            total_max_pedestrian_queue, episode_max_pedestrian_queue
        )

        total_sum_vehicle_wait += episode_sum_vehicle_wait
        total_max_vehicle_wait = max(total_max_vehicle_wait, episode_max_vehicle_wait)

        total_sum_pedestrian_wait += episode_sum_pedestrian_wait
        total_max_pedestrian_wait = max(
            total_max_pedestrian_wait, episode_max_pedestrian_wait
        )

        total_vehicle_throughput += episode_vehicle_throughput
        total_pedestrian_throughput += episode_pedestrian_throughput
        total_switch_count += episode_switch_count
        total_cumulative_reward += cumulative_reward_episode

    total_steps = BASELINE_EPISODES * BASELINE_STEPS_PER_EPISODE

    # Save step-level CSV
    if step_metrics:
        with open(STEP_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
            writer.writeheader()
            writer.writerows(step_metrics)

    # Save summary CSV (aggregated over all episodes)
    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "avg_vehicle_queue", "max_vehicle_queue",
                "avg_ped_queue", "max_ped_queue",
                "avg_total_queue", "max_total_queue",
                "avg_vehicle_wait", "max_vehicle_wait",
                "avg_ped_wait", "max_ped_wait",
                "vehicle_throughput", "ped_throughput", "total_throughput",
                "switch_count", "cumulative_reward", "runtime_seconds",
                "episodes", "steps_per_episode",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "avg_vehicle_queue": total_sum_vehicle_queue / total_steps,
                "max_vehicle_queue": total_max_vehicle_queue,
                "avg_ped_queue": total_sum_pedestrian_queue / total_steps,
                "max_ped_queue": total_max_pedestrian_queue,
                "avg_total_queue": (
                    (total_sum_vehicle_queue + total_sum_pedestrian_queue) / total_steps
                ),
                "max_total_queue": total_max_vehicle_queue + total_max_pedestrian_queue,
                "avg_vehicle_wait": total_sum_vehicle_wait / total_steps,
                "max_vehicle_wait": total_max_vehicle_wait,
                "avg_ped_wait": total_sum_pedestrian_wait / total_steps,
                "max_ped_wait": total_max_pedestrian_wait,
                "vehicle_throughput": total_vehicle_throughput,
                "ped_throughput": total_pedestrian_throughput,
                "total_throughput": (
                    total_vehicle_throughput + total_pedestrian_throughput
                ),
                "switch_count": total_switch_count,
                "cumulative_reward": total_cumulative_reward,
                "runtime_seconds": round(time.time() - start_time, 2),
                "episodes": BASELINE_EPISODES,
                "steps_per_episode": BASELINE_STEPS_PER_EPISODE,
            }
        )

    print(
        f"Baseline complete in {round(time.time() - start_time, 2)} seconds "
        f"over {BASELINE_EPISODES} episode(s)"
    )

    # === Visualization (simple, over all steps) ===
    steps = [row["global_step"] for row in step_metrics]
    queue_lengths = [row["total_queue"] for row in step_metrics]
    cumulative_rewards = [row["cumulative_reward"] for row in step_metrics]

    # Plot Cumulative Reward
    plt.figure(figsize=(10, 6))
    plt.plot(steps, cumulative_rewards, marker="o", linestyle="-", label="Cumulative Reward")
    plt.xlabel("Global Step")
    plt.ylabel("Cumulative Reward")
    plt.title("Fixed-Time Baseline: Cumulative Reward over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HOST_RESULTS_DIR, "b1_cumulative_reward.png"))

    # Plot Total Queue Length
    plt.figure(figsize=(10, 6))
    plt.plot(steps, queue_lengths, marker="o", linestyle="-", label="Total Queue Length")
    plt.xlabel("Global Step")
    plt.ylabel("Total Queue Length")
    plt.title("Fixed-Time Baseline: Queue Length over Steps")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(HOST_RESULTS_DIR, "b1_queue_length.png"))



if __name__ == "__main__":
    run_baseline()
