# evaluate_rl.py
import os
import csv
import traceback
import traci

import numpy as np
import torch

from traffic_env import TrafficEnv
from dqn_agent import DQNAgent


HOST_RESULTS_DIR = "results"
os.makedirs(HOST_RESULTS_DIR, exist_ok=True)

RL_STEP_CSV = os.path.join(HOST_RESULTS_DIR, "rl_metrics.csv")
RL_SUMMARY_CSV = os.path.join(HOST_RESULTS_DIR, "rl_summary.csv")

MODEL_PATH = "models/dqn_ep200.pt"  # adjust if needed


def load_agent(env: TrafficEnv, model_path: str) -> DQNAgent:
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
    if os.path.isfile(model_path):
        state = torch.load(model_path, map_location=agent.device)
        agent.q_net.load_state_dict(state["q_net"])
        agent.target_net.load_state_dict(state["target_net"])
        agent.total_steps = state.get("total_steps", 0)
        print(f"Loaded model from {model_path}")
    else:
        print(f"WARNING: model file not found at {model_path}, using untrained agent.")
    return agent


def evaluate_rl():
    env = TrafficEnv(render_mode=None)
    agent = load_agent(env, MODEL_PATH)

    step_metrics = []

    # Episode-level trackers (mirror baseline)
    veh_accum_wait_finished = []
    ped_accum_wait_finished = []

    sum_vehicle_queue = 0.0
    max_vehicle_queue = 0

    sum_ped_queue = 0.0
    max_ped_queue = 0

    cumulative_reward = 0.0
    throughput = 0  # FIXED: now tracks exited vehicles

    try:
        state, _ = env.reset()
        done = False
        truncated = False
        step = 0

        prev_vehicles = set()
        prev_peds = set()

        while not (done or truncated) and step < env.max_steps:

            # Track before step
            prev_vehicles = set(traci.vehicle.getIDList())
            prev_peds = set(traci.person.getIDList())

            action = agent.act(state, eval_mode=True)
            next_state, reward, done, truncated, info = env.step(action)

            # Track after step
            current_vehicles = set(traci.vehicle.getIDList())
            current_peds = set(traci.person.getIDList())

            exited_vehicles = prev_vehicles - current_vehicles
            exited_peds = prev_peds - current_peds

            # FIXED: throughput now matches baseline
            throughput += len(exited_vehicles)

            # Extract metrics from observation
            vehicle_queue = float(next_state[0])
            ped_queue = float(next_state[1])
            total_vehicle_wait = float(next_state[2])
            total_ped_wait = float(next_state[3])
            vehicle_count = int(next_state[4])
            ped_count = int(next_state[5])

            sum_vehicle_queue += vehicle_queue
            max_vehicle_queue = max(max_vehicle_queue, int(vehicle_queue))

            sum_ped_queue += ped_queue
            max_ped_queue = max(max_ped_queue, int(ped_queue))

            cumulative_reward += reward

            step_metrics.append(
                {
                    "step": step,
                    "vehicle_waiting_time": total_vehicle_wait,
                    "vehicle_queue": vehicle_queue,
                    "vehicle_count": vehicle_count,
                    "ped_waiting_time": total_ped_wait,
                    "ped_queue": ped_queue,
                    "ped_count": ped_count,
                    "reward": reward,
                }
            )

            state = next_state
            step += 1

        # Averages (approximate, consistent with baseline style)
        avg_vehicle_wait = float(np.mean([m["vehicle_waiting_time"] for m in step_metrics])) if step_metrics else 0.0
        avg_ped_wait = float(np.mean([m["ped_waiting_time"] for m in step_metrics])) if step_metrics else 0.0

        avg_vehicle_queue = sum_vehicle_queue / max(1, step)
        avg_ped_queue = sum_ped_queue / max(1, step)

        # Write step CSV
        if step_metrics:
            with open(RL_STEP_CSV, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=step_metrics[0].keys())
                writer.writeheader()
                writer.writerows(step_metrics)

        # Write summary CSV
        with open(RL_SUMMARY_CSV, "w", newline="") as f:
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
                    "simulation_steps",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "avg_vehicle_waiting_time": avg_vehicle_wait,
                    "avg_vehicle_queue_length": avg_vehicle_queue,
                    "max_vehicle_queue_length": max_vehicle_queue,
                    "avg_ped_waiting_time": avg_ped_wait,
                    "avg_ped_queue_length": avg_ped_queue,
                    "max_ped_queue_length": max_ped_queue,
                    "throughput": throughput,  # FIXED
                    "cumulative_reward": cumulative_reward,
                    "simulation_steps": step,
                }
            )

        print("RL evaluation complete")
        print("Step metrics:", RL_STEP_CSV)
        print("Summary metrics:", RL_SUMMARY_CSV)

    except Exception:
        print("ERROR during RL evaluation")
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    evaluate_rl()
