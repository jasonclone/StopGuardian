# rl/train.py

import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from typing import List, Dict, Any

import numpy as np
import torch
import traci

from traffic_env import (
    TrafficEnv,
    save_step_csv,
    save_episode_csv,
    plot_metrics,
    make_sumo_config,
)

from model import build_rainbow_model
from replay import PrioritizedReplayBuffer
from learner import train_step
from action import apply_action_safe, select_action

from config import (
    DEVICE,
    GAMMA,
    N_STEPS,
    BUFFER_SIZE,
    BATCH_SIZE,
    MIN_REPLAY_SIZE,
    TARGET_UPDATE_FREQ,
    WARMUP_STEPS,
    USE_EPSILON,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    ACTIONS,
    NUM_ACTIONS,
    PRIORITY_ALPHA,
    PRIORITY_BETA_START,
    PRIORITY_BETA_END,
    NOISY_SIGMA, # not used here, but in model.py
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)

env = TrafficEnv("C")

# ================================================================
# Args & Modes
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "eval", "infer"],
)
parser.add_argument("--run_id", type=str, default="rl")
parser.add_argument("--episodes", type=int, default=1)
parser.add_argument("--steps_per_episode", type=int, default=5000)
args = parser.parse_args()

MODE = args.mode
RUN_ID = args.run_id
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps_per_episode

TRAIN_EPISODES = max(1, int(NUM_EPISODES * 0.8))
EVAL_EPISODES = NUM_EPISODES - TRAIN_EPISODES

# ================================================================
# Paths
# ================================================================

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RUN_DIR = os.path.join(RESULTS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

REPLAY_DIR = os.path.join(RUN_DIR, "replay")
os.makedirs(REPLAY_DIR, exist_ok=True)

RL_STEP_CSV = os.path.join(RUN_DIR, "rl_step_metrics.csv")
RL_EPISODE_CSV = os.path.join(RUN_DIR, "rl_episode_metrics.csv")
PLOT_PATH = os.path.join(RUN_DIR, "rl_plot_reward.png")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
REPLAY_PATH = os.path.join(REPLAY_DIR, "replay.pkl")

# ================================================================
# SUMO setup
# ================================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

# ================================================================
# Init models & replay
# ================================================================

def init_models_and_replay():
    traci.start(make_sumo_config())
    traci.simulationStep()
    
    dummy_state = env.get_state()
    dummy_norm = env.normalize_state(dummy_state) # must happen before close
    traci.close()

    state_size = len(dummy_norm)

    online_model, online_optimizer = build_rainbow_model(
        state_size=state_size,
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        lr=LEARNING_RATE,
        device=DEVICE,
    )
    target_model, _ = build_rainbow_model(
        state_size=state_size,
        num_actions=NUM_ACTIONS,
        num_atoms=NUM_ATOMS,
        v_min=V_MIN,
        v_max=V_MAX,
        lr=LEARNING_RATE,
        device=DEVICE,
    )
    target_model.load_state_dict(online_model.state_dict())

    replay_buffer = PrioritizedReplayBuffer(
        capacity=BUFFER_SIZE,
        alpha=PRIORITY_ALPHA,
        n_steps=N_STEPS,
        gamma=GAMMA,
    )

    if os.path.exists(LAST_MODEL_PATH):
        state_dict = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
        online_model.load_state_dict(state_dict)
        target_model.load_state_dict(state_dict)
        print(f"Loaded last model from {LAST_MODEL_PATH}")

    if os.path.exists(REPLAY_PATH):
        replay_buffer.load(REPLAY_PATH)
        print(f"Loaded replay buffer from {REPLAY_PATH}")

    return online_model, target_model, replay_buffer, online_optimizer

# ================================================================
# Main loop
# ================================================================

best_metric = None
if os.path.exists(BEST_MODEL_PATH):
    print(f"Best model already exists at {BEST_MODEL_PATH}")


def run():
    global best_metric

    online_model, target_model, replay_buffer, optimizer = init_models_and_replay()

    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    total_steps_global = 0
    total_grad_steps_global = 0

    total_planned_updates = max(NUM_EPISODES * STEPS_PER_EPISODE, 1)
    beta = PRIORITY_BETA_START
    beta_increment = (PRIORITY_BETA_END - PRIORITY_BETA_START) / total_planned_updates

    print(f"\n=== Starting Rainbow DQN ({MODE.upper()} | run_id={RUN_ID}) ===")
    print(
        f"Episodes: {NUM_EPISODES} (Train: {TRAIN_EPISODES}, Eval: {EVAL_EPISODES}), "
        f"Steps per episode: {STEPS_PER_EPISODE}"
    )
    print("Training mode: ONLINE UPDATES (train every step)\n")

    try:
        for ep in range(NUM_EPISODES):
            if MODE == "train":
                ep_mode = "train" if ep < TRAIN_EPISODES else "eval"
            else:
                ep_mode = MODE

            traci.start(make_sumo_config())
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
                    # 2. ACTION
                    # =========================
                    action = select_action(
                        state=state,
                        step=t,
                        mode=ep_mode,
                        online_model=online_model,
                        device=DEVICE,
                        actions=ACTIONS,
                        warmup_steps=WARMUP_STEPS,
                        use_epsilon=USE_EPSILON,
                        epsilon_start=EPSILON_START,
                        epsilon_end=EPSILON_END,
                        epsilon_decay_steps=EPSILON_DECAY_STEPS,
                        normalize_state=env.normalize_state,
                    )

                    apply_action_safe(action, "C")

                    # =========================
                    # 3. PHASE TRACKING
                    # =========================
                    cur_phase = env.get_phase()
                    if cur_phase != prev_phase:
                        switch_count += 1
                        prev_phase = cur_phase

                    # =========================
                    # 4. STEP SIMULATION
                    # =========================
                    traci.simulationStep()

                    # =========================
                    # 5. NEXT STATE (AFTER STEP)
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

                    done = (t == STEPS_PER_EPISODE - 1)

                    step_rows.append({
                        "global_step": total_steps_global,
                        "episode": ep,
                        "mode": ep_mode,
                        "step_in_episode": t,
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
                        "loss": None,
                        "action": None,
                    })

                    # =========================
                    # 9. TRAINING
                    # =========================
                    if ep_mode == "train":
                        replay_buffer.add(state, action, reward, next_state, done)

                        loss = train_step(
                            beta=beta,
                            online_model=online_model,
                            target_model=target_model,
                            replay_buffer=replay_buffer,
                            optimizer=optimizer,
                            device=DEVICE,
                            batch_size=BATCH_SIZE,
                            min_replay_size=MIN_REPLAY_SIZE,
                            gamma=GAMMA,
                            n_steps=N_STEPS,
                            num_atoms=NUM_ATOMS,
                            v_min=V_MIN,
                            v_max=V_MAX,
                            delta_z=DELTA_Z,
                            normalize_state=env.normalize_state,
                        )

                        if loss is not None:
                            total_grad_steps_global += 1
                            beta = min(1.0, beta + beta_increment)

                            step_rows[-1]["loss"] = loss
                            step_rows[-1]["action"] = action

                            if total_grad_steps_global % TARGET_UPDATE_FREQ == 0:
                                target_model.load_state_dict(online_model.state_dict())

                    total_steps_global += 1

                episode_ok = True

            except Exception as e:
                print(f"[Episode {ep}] Exception: {e}")
            finally:
                try:
                    traci.close()
                except:
                    pass

            if ep_mode == "train":
                replay_buffer.flush()

            avg_vehicle_queue = float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else -1.0
            avg_ped_queue = float(np.mean(ped_queue_hist)) if ped_queue_hist else -1.0
            avg_total_queue = float(np.mean(total_queue_hist)) if total_queue_hist else -1.0

            avg_vehicle_wait = float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else -1.0
            avg_ped_wait = float(np.mean(ped_wait_hist)) if ped_wait_hist else -1.0
            avg_total_wait = float(np.mean(total_wait_hist)) if total_wait_hist else -1.0

            episode_rows.append({
                "episode": ep,
                "mode": ep_mode,
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
                f"[Ep {ep:04d} | {ep_mode}] "
                f"R={cumulative_reward:.2f} | "
                f"Q_tot={avg_total_queue:.2f} | "
                f"W_tot={avg_total_wait:.2f} | "
                f"veh_thru={vehicle_throughput} | ped_thru={ped_throughput} | "
                f"switches={switch_count}"
            )

            save_step_csv(step_rows, RL_STEP_CSV)
            save_episode_csv(episode_rows, RL_EPISODE_CSV)
            plot_metrics(episode_rows, PLOT_PATH, RUN_ID=RUN_ID)

            if ep_mode == "train" and episode_ok:
                metric = avg_total_queue
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    torch.save(online_model.state_dict(), BEST_MODEL_PATH)
                    print(f"  -> New best model saved (avg_total_queue={metric:.3f})")

                if ep % CHECKPOINT_EVERY_EPISODES == 0:
                    torch.save(online_model.state_dict(), LAST_MODEL_PATH)
                    replay_buffer.save(REPLAY_PATH)

        torch.save(online_model.state_dict(), LAST_MODEL_PATH)
        replay_buffer.save(REPLAY_PATH)

    finally:
        try:
            traci.close()
        except:
            pass


if __name__ == "__main__":
    run()
