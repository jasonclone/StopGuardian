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


# set tune mode early to avoid importing TensorBoard in ray tune workers
if os.environ.get("TUNE_MODE") != "1":
    from torch.utils.tensorboard import SummaryWriter
else:
    SummaryWriter = None

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
    NOISY_SIGMA,  # not used here, but in model.py
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)






MODE = "train" # default
RUN_ID = "rl" # default
NUM_EPISODES = 5 # default
STEPS_PER_EPISODE = 1000 # default

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

# create environment
env = TrafficEnv("C")

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

def init_models_and_replay(state_size):
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

    return online_model, target_model, replay_buffer, online_optimizer


# ================================================================
# Main loop
# ================================================================

best_metric = None
if os.path.exists(BEST_MODEL_PATH):
    print(f"Best model already exists at {BEST_MODEL_PATH}")


def run():
    global best_metric
    
    # Start SUMO once to compute state size
    traci.start(make_sumo_config())
    traci.simulationStep()
    env.initialize_from_sumo()
    state_size = env.compute_state_size()
    traci.close()

    online_model, target_model, replay_buffer, optimizer = init_models_and_replay(state_size=state_size)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',       # maximize cumulative reward
    factor=0.5,       # cut LR in half
    patience=5,       # wait 5 episodes with no improvement
    min_lr=1e-6
)
    
    print("Using device:", DEVICE)
    print("Model device:", next(online_model.parameters()).device)
    print("Tensor device test:", torch.tensor([1.0]).to(DEVICE).device) # sanity check for device compatibility

    # TensorBoard writer
    tb_dir = os.path.join(RUN_DIR, "tb")
    writer = SummaryWriter(log_dir=tb_dir)

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

                    # =========================
                    # 1. READ CURRENT STATE
                    # =========================
                    
                    state_raw, _ = env.get_state()  # for reward calculation
                    
                    # =========================
                    # 2. ACTION
                    # =========================
                    action = select_action(
                        env=env,
                        state_raw=state_raw,
                        global_step=total_steps_global,
                        mode=ep_mode,
                        online_model=online_model,
                        device=DEVICE,
                        actions=ACTIONS,
                        warmup_steps=WARMUP_STEPS,
                        use_epsilon=USE_EPSILON,
                        epsilon_start=EPSILON_START,
                        epsilon_end=EPSILON_END,
                        epsilon_decay_steps=EPSILON_DECAY_STEPS,
                        normalize_state_torch=env.normalize_state_torch, # pass in the env's normalization gpu function for states to be used inside action selection
                    )

                    apply_action_safe(action, env, current_step_global=total_steps_global)

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
                    reward = env.get_reward(next_state_raw, state_raw, action) # use raw (unnormalized) states for reward calculation
                    cumulative_reward += reward

                    # =========================
                    # 8. LOGGING (in-memory + TensorBoard step-level)
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
                        "vehicle_queue": vehicle_queue,
                        "vehicle_wait": vehicle_wait,
                        "ped_queue": ped_queue,
                        "ped_wait": ped_wait,
                        "ped_thru_step": ped_thru,
                        "veh_thru_step": veh_thru,
                        "phase": phase,
                        "reward": reward,
                        "loss": None,
                        "action": None,
                    })


                    # TensorBoard: step-level environment metrics
                    writer.add_scalar("env/queue_vehicle", vehicle_queue, total_steps_global)
                    writer.add_scalar("env/queue_ped", ped_queue, total_steps_global)
                    writer.add_scalar("env/wait_vehicle", vehicle_wait, total_steps_global)
                    writer.add_scalar("env/wait_ped", ped_wait, total_steps_global)
                    writer.add_scalar("env/reward_step", reward, total_steps_global)
                    writer.add_scalar("env/veh_thru_step", veh_thru, total_steps_global)
                    writer.add_scalar("env/ped_thru_step", ped_thru, total_steps_global)
                    

                    # =========================
                    # 9. TRAINING
                    # =========================
                    if ep_mode == "train":
                        replay_buffer.add(state_raw, action, reward, next_state_raw, done)

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
                            normalize_state_torch=env.normalize_state_torch, # pass in the env's normalization gpu function for states to be used inside train_step
                        )

                        if loss is not None:
                            total_grad_steps_global += 1
                            beta = min(1.0, beta + beta_increment)

                            step_rows[-1]["loss"] = loss
                            step_rows[-1]["action"] = action

                            # TensorBoard: training metrics
                            writer.add_scalar("train/loss", loss, total_grad_steps_global)
                            writer.add_scalar("train/beta", beta, total_grad_steps_global)
                            writer.add_scalar("train/learning_rate", optimizer.param_groups[0]["lr"], total_grad_steps_global)
                            writer.add_scalar("train/replay_size", len(replay_buffer), total_grad_steps_global)

                            # Optional: log weights/gradients every 1000 grad steps
                            if total_grad_steps_global % 1000 == 0:
                                for name, param in online_model.named_parameters():
                                    writer.add_histogram(f"weights/{name}", param, total_grad_steps_global)
                                    if param.grad is not None:
                                        writer.add_histogram(f"grads/{name}", param.grad, total_grad_steps_global)

                            if total_grad_steps_global % TARGET_UPDATE_FREQ == 0:
                                target_model.load_state_dict(online_model.state_dict())
                                writer.add_scalar("train/target_update", 1, total_grad_steps_global)

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
                "mode": ep_mode,
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
                f"[Ep {ep:04d} | {ep_mode}] "
                f"R={cumulative_reward:.2f} | "
                f"Total_veh_thru={vehicle_throughput} | Total_ped_thru={ped_throughput} | "
                f"switches={switch_count}"
            )

            save_step_csv(step_rows, RL_STEP_CSV)
            save_episode_csv(episode_rows, RL_EPISODE_CSV)
            plot_metrics(episode_rows, PLOT_PATH, RUN_ID=RUN_ID)

            if ep_mode == "train" and episode_ok:
                metric = cumulative_reward
                scheduler.step(metric)

                current_lr = optimizer.param_groups[0]["lr"]
                print(f"  -> Current learning rate: {current_lr:.8f}") 


                if best_metric is None or metric > best_metric:
                    best_metric = metric
                    torch.save(online_model.state_dict(), BEST_MODEL_PATH)
                    print(f"  -> New best model saved (new metric record achieved={metric:.3f})")

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
        writer.close()  # close TensorBoard writer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval", "infer"],
    )
    parser.add_argument("--run_id", type=str, default="rl")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--steps_per_episode", type=int, default=1000)
    args = parser.parse_args()

    MODE = args.mode
    RUN_ID = args.run_id
    NUM_EPISODES = args.episodes
    STEPS_PER_EPISODE = args.steps_per_episode

    TRAIN_EPISODES = max(1, int(NUM_EPISODES * 0.8))
    EVAL_EPISODES = NUM_EPISODES - TRAIN_EPISODES

    run()