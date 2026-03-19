"""
Rainbow DQN for SUMO Traffic Signal Control Episode-Based Pipeline (PyTorch)
-----------------------------------------------------------------------------

Features:
- Noisy Networks, Dueling, Double DQN
- Prioritized Replay with N-step returns
- Warm-up exploration (TRAIN)
- Clean state normalization
- TRAIN / EVAL / INFER modes (via --mode)
- Episode-based training (outer loop over episodes)
- New SUMO seed each episode (to avoid overfitting one scenario)
- Persistent training, crash recovery
- Checkpointing + best-model tracking (by episode metric)
- Replay buffer persistence
- CSV (per-step + per-episode) + PNG plot per episode
- Deterministic base seed + controlled episode variation
- Multi-run scalability via --run_id

"""

import os
import sys
import csv
import random
import argparse
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ================================================================
#  Args & Modes
# ================================================================

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    default="train",
    choices=["train", "eval", "infer"],
    help="train: learn; eval: no exploration, log; infer: run greedy, no logging",
)
parser.add_argument(
    "--run_id",
    type=str,
    default="default",
    help="Run identifier for checkpoints/logs",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=30,
    help="Number of episodes to run",
)
parser.add_argument(
    "--steps_per_episode",
    type=int,
    default=1000,
    help="Number of environment steps per episode",
)
args = parser.parse_args()

MODE = args.mode
RUN_ID = args.run_id
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps_per_episode

# ================================================================
#  Deterministic Base Seed
# ================================================================

GLOBAL_SEED = 12345
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
#  Directories & Output
# ================================================================

BASE_RESULTS_DIR = "results"
RUN_DIR = os.path.join(BASE_RESULTS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

REPLAY_DIR = os.path.join(RUN_DIR, "replay")
os.makedirs(REPLAY_DIR, exist_ok=True)

TB_LOG_DIR = os.path.join(RUN_DIR, "tensorboard")
os.makedirs(TB_LOG_DIR, exist_ok=True)

RL_STEP_CSV = os.path.join(RUN_DIR, "rl_step_metrics.csv")
RL_EPISODE_CSV = os.path.join(RUN_DIR, "rl_episode_metrics.csv")
PLOT_PATH = os.path.join(RUN_DIR, "rl_combined.png")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.keras")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.keras")
REPLAY_PATH = os.path.join(REPLAY_DIR, "replay.pkl")

# No TensorBoard in PyTorch version; logic otherwise identical
tb_writer = None

# ================================================================
#  SUMO Setup (function so we can restart per episode)
# ================================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci  # noqa: E402


def make_sumo_config(seed: int) -> List[str]:
    # If you want GUI for eval/infer, you can switch to "sumo-gui" here.
    binary = "sumo"
    return [
        binary,
        "-c",
        "simulation/sumo/test.sumocfg",
        "--step-length",
        "0.10",
        "--delay",
        "1000",
        "--lateral-resolution",
        "0",
        "--seed",
        str(seed),
    ]


# ================================================================
#  Hyperparameters
# ================================================================

GAMMA = 0.99
N_STEPS = 5

BUFFER_SIZE = 120000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 2000  # minimum transitions before training

TARGET_UPDATE_FREQ = 2500
WARMUP_STEPS = 2000

USE_EPSILON = False
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_STEPS = 10000

ACTIONS = [0, 1]
NUM_ACTIONS = len(ACTIONS)

PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_END = 1.0

NOISY_SIGMA = 0.5
LEARNING_RATE = 0.00025

QUEUE_NORM = 20.0
PHASE_NORM = 10.0

CHECKPOINT_EVERY_EPISODES = 5  # checkpoint every N episodes

# ================================================================
#  Environment Functions
# ================================================================


def get_queue_length(detector_id: str) -> int:
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tls_id="C") -> int:
    return traci.trafficlight.getPhase(tls_id)


def get_state():
    """Return (q_N, q_E, q_S, q_W, phase)."""
    q_N = sum(get_queue_length(f"C_NC_{i}") for i in range(1, 5))
    q_E = sum(get_queue_length(f"C_EC_{i}") for i in range(1, 5))
    q_S = sum(get_queue_length(f"C_SC_{i}") for i in range(1, 5))
    q_W = sum(get_queue_length(f"C_WC_{i}") for i in range(1, 5))
    phase = get_current_phase("C")
    return (q_N, q_E, q_S, q_W, phase)


def normalize_state(s):
    q_N, q_E, q_S, q_W, phase = s
    return np.array(
        [q_N / QUEUE_NORM, q_E / QUEUE_NORM, q_S / QUEUE_NORM, q_W / QUEUE_NORM, phase / PHASE_NORM],
        dtype=np.float32,
    )


def get_reward(state, prev_state=None):
    total_queue = sum(state[:-1])
    reward = -float(total_queue)
    if prev_state is not None:
        prev_q = sum(prev_state[:-1])
        reward += 0.25 * (prev_q - total_queue)  # queue reduction shaping
    return reward


def apply_action_safe(action, tls_id="C"):
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phase = get_current_phase(tls_id)
    state = program.phases[phase].state

    # Only switch when green
    if "y" in state or ("G" not in state and "g" not in state):
        return

    if action == 1:
        traci.trafficlight.setPhase(tls_id, (phase + 1) % len(program.phases))


# ================================================================
#  Noisy Linear Layer (PyTorch)
# ================================================================


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu_range = 1 / np.sqrt(in_features)
        self.weight_mu = nn.Parameter(torch.empty(in_features, out_features).uniform_(-mu_range, mu_range))
        self.weight_sigma = nn.Parameter(
            torch.full((in_features, out_features), sigma_init / np.sqrt(in_features), dtype=torch.float32)
        )

        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-mu_range, mu_range))
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init / np.sqrt(in_features), dtype=torch.float32)
        )

    def forward(self, x):
        if self.training:
            eps_in = torch.randn(self.in_features, device=x.device)
            eps_out = torch.randn(self.out_features, device=x.device)

            f_in = torch.sign(eps_in) * torch.sqrt(torch.abs(eps_in))
            f_out = torch.sign(eps_out) * torch.sqrt(torch.abs(eps_out))

            w_noise = torch.ger(f_in, f_out)
            b_noise = f_out

            weight = self.weight_mu + self.weight_sigma * w_noise
            bias = self.bias_mu + self.bias_sigma * b_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return x @ weight + bias


# ================================================================
#  Rainbow DQN Model (Dueling + Noisy) - PyTorch
# ================================================================


class RainbowDQN(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream
        self.v_noisy1 = NoisyLinear(128, 128)
        self.v_relu = nn.ReLU()
        self.v_noisy2 = NoisyLinear(128, 1)

        # Advantage stream
        self.a_noisy1 = NoisyLinear(128, 128)
        self.a_relu = nn.ReLU()
        self.a_noisy2 = NoisyLinear(128, num_actions)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        v = self.v_noisy1(x)
        v = self.v_relu(v)
        v = self.v_noisy2(v)  # (batch, 1)

        a = self.a_noisy1(x)
        a = self.a_relu(a)
        a = self.a_noisy2(a)  # (batch, num_actions)

        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q


def build_rainbow_model(state_size, num_actions):
    model = RainbowDQN(state_size, num_actions).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


# ================================================================
#  Prioritized Replay Buffer (with persistence)
# ================================================================


@dataclass
class ReplayState:
    buffer: list
    pos: int
    priorities: np.ndarray
    n_step_buffer: list


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha, n_steps, gamma):
        self.capacity = capacity
        self.alpha = alpha
        self.n_steps = n_steps
        self.gamma = gamma
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.n_step_buffer = []

    def __len__(self):
        return len(self.buffer)

    def _priority(self, td_error):
        return (abs(td_error) + 1e-6) ** self.alpha

    def add(self, s, a, r, s_next, done):
        self.n_step_buffer.append((s, a, r, s_next, done))
        if len(self.n_step_buffer) < self.n_steps:
            return
        R = sum((self.gamma ** i) * r_i for i, (_, _, r_i, _, _) in enumerate(self.n_step_buffer))
        s0, a0, _, _, _ = self.n_step_buffer[0]
        _, _, _, sN, dN = self.n_step_buffer[-1]
        if len(self.buffer) < self.capacity:
            self.buffer.append((s0, a0, R, sN, dN))
        else:
            self.buffer[self.pos] = (s0, a0, R, sN, dN)
        max_prio = self.priorities.max() if self.buffer else 1.0
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.n_step_buffer.pop(0)

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: self.pos]
        prio_sum = prios.sum()
        if prio_sum <= 0 or np.isnan(prio_sum):
            prios = np.ones_like(prios)
            prio_sum = prios.sum()
        probs = prios / prio_sum
        idxs = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in idxs]
        weights = (len(self.buffer) * probs[idxs]) ** (-beta)
        weights /= weights.max()
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
            idxs,
            weights.astype(np.float32),
        )

    def update_priorities(self, idxs, td_errors):
        for i, td in zip(idxs, td_errors):
            self.priorities[i] = self._priority(td)

    def save(self, path: str):
        state = ReplayState(
            buffer=self.buffer,
            pos=self.pos,
            priorities=self.priorities,
            n_step_buffer=self.n_step_buffer,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state: ReplayState = pickle.load(f)
        self.buffer = state.buffer
        self.pos = state.pos
        self.priorities = state.priorities
        self.n_step_buffer = state.n_step_buffer


# ================================================================
#  Build Models & Replay
# ================================================================


def init_models_and_replay():
    # Start a short SUMO run just to get state size
    traci.start(make_sumo_config(GLOBAL_SEED))
    dummy_state = get_state()
    traci.close()

    state_size = len(dummy_state)

    online_model, online_optimizer = build_rainbow_model(state_size, NUM_ACTIONS)
    target_model, _ = build_rainbow_model(state_size, NUM_ACTIONS)
    target_model.load_state_dict(online_model.state_dict())

    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, PRIORITY_ALPHA, N_STEPS, GAMMA)
    replay_buffer.load(REPLAY_PATH)

    # Load last model if exists (persistent training / infer)
    if os.path.exists(LAST_MODEL_PATH):
        state_dict = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
        online_model.load_state_dict(state_dict)
        target_model.load_state_dict(state_dict)
        print(f"Loaded last model from {LAST_MODEL_PATH}")

    return online_model, target_model, replay_buffer, online_optimizer


best_metric = None
if os.path.exists(BEST_MODEL_PATH):
    print(f"Best model already exists at {BEST_MODEL_PATH}")

# ================================================================
#  Action Selection
# ================================================================


def select_action(state, step, mode: str, online_model):
    s_norm = normalize_state(state).reshape(1, -1)
    s_tensor = torch.from_numpy(s_norm).float().to(DEVICE)

    # TRAIN: warmup + (optional) epsilon; NoisyNet already gives exploration
    if mode == "train":
        if step < WARMUP_STEPS:
            return random.choice(ACTIONS)
        if USE_EPSILON:
            eps = max(
                EPSILON_END,
                EPSILON_START - (EPSILON_START - EPSILON_END) * step / EPSILON_DECAY_STEPS,
            )
            if random.random() < eps:
                return random.choice(ACTIONS)

    # EVAL / INFER: greedy, no exploration
    online_model.eval()
    with torch.no_grad():
        q_vals = online_model(s_tensor)[0].cpu().numpy()
    return int(np.argmax(q_vals))


# ================================================================
#  Training Step
# ================================================================


def train_step(beta, online_model, target_model, replay_buffer, optimizer) -> Optional[float]:
    if MODE != "train":
        return None
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(BATCH_SIZE, beta)
    states_norm = np.array([normalize_state(s) for s in states], dtype=np.float32)
    next_states_norm = np.array([normalize_state(s) for s in next_states], dtype=np.float32)

    states_tensor = torch.from_numpy(states_norm).float().to(DEVICE)
    next_states_tensor = torch.from_numpy(next_states_norm).float().to(DEVICE)
    actions_tensor = torch.from_numpy(actions).long().to(DEVICE)
    rewards_tensor = torch.from_numpy(rewards).float().to(DEVICE)
    dones_tensor = torch.from_numpy(dones.astype(np.float32)).float().to(DEVICE)
    weights_tensor = torch.from_numpy(weights).float().to(DEVICE)

    # Double DQN
    online_model.eval()
    target_model.eval()
    with torch.no_grad():
        online_next_q = online_model(next_states_tensor)  # (B, A)
        next_actions = online_next_q.argmax(dim=1)  # (B,)
        target_next_q = target_model(next_states_tensor)  # (B, A)
        target_q_vals = target_next_q.gather(1, next_actions.unsqueeze(1)).squeeze(1)  # (B,)

    online_model.train()
    q_pred_all = online_model(states_tensor)  # (B, A)
    q_taken = q_pred_all.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)  # (B,)

    # Targets (same logic as original)
    gamma_n = (GAMMA ** N_STEPS)
    q_target = rewards_tensor + (1.0 - dones_tensor) * gamma_n * target_q_vals

    td_errors = (q_target - q_taken).detach().cpu().numpy()
    loss = (weights_tensor * (q_taken - q_target).pow(2)).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    replay_buffer.update_priorities(idxs, td_errors)
    return float(loss.item())


# ================================================================
#  Helpers
# ================================================================


def moving_avg(data, window=50):
    if len(data) < window:
        return float(np.mean(data)) if data else 0.0
    return float(np.mean(data[-window:]))


def save_step_csv(step_rows: List[Dict[str, Any]]):
    if not step_rows:
        return
    fieldnames = list(step_rows[0].keys())
    with open(RL_STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(step_rows)


def save_episode_csv(episode_rows: List[Dict[str, Any]]):
    if not episode_rows:
        return
    fieldnames = list(episode_rows[0].keys())
    with open(RL_EPISODE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(episode_rows)


def plot_metrics(episode_rows: List[Dict[str, Any]]):
    if not episode_rows:
        return

    eps = [r["episode"] for r in episode_rows]
    cum_rewards = [r["cumulative_reward"] for r in episode_rows]
    avg_queues = [r["avg_queue"] for r in episode_rows]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(eps, cum_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Episode Metrics ({MODE.upper()} - {RUN_ID})")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(eps, avg_queues, marker="o", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Queue Length")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


# ================================================================
#  Main Episode-Based Loop (with Option A2 behavior)
# ================================================================


def run():
    global best_metric

    online_model, target_model, replay_buffer, optimizer = init_models_and_replay()

    # For CSV logging
    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    total_steps_global = 0
    beta = PRIORITY_BETA_START
    beta_increment = (PRIORITY_BETA_END - PRIORITY_BETA_START) / max(NUM_EPISODES * STEPS_PER_EPISODE, 1)

    print(f"\n=== Starting Rainbow DQN ({MODE.upper()} | run_id={RUN_ID}) ===")
    print(f"Episodes: {NUM_EPISODES}, Steps per episode: {STEPS_PER_EPISODE}")

    # Global safety: always save CSVs/plot/replay/last model even on crash
    try:
        for ep in range(NUM_EPISODES):
            # New seed per episode to avoid overfitting one scenario
            episode_seed = GLOBAL_SEED + ep * 1000
            traci.start(make_sumo_config(episode_seed))

            cumulative_reward = 0.0
            queue_history_ep: List[float] = []
            prev_state = None

            print(f"\n--- Episode {ep+1}/{NUM_EPISODES} (seed={episode_seed}) ---")

            episode_ok = False
            try:
                for t in range(STEPS_PER_EPISODE):
                    step_idx = total_steps_global

                    state = get_state()
                    action = select_action(state, step_idx, MODE, online_model)
                    apply_action_safe(action)

                    traci.simulationStep()
                    new_state = get_state()
                    reward = get_reward(new_state, prev_state)
                    cumulative_reward += reward

                    if MODE == "train":
                        replay_buffer.add(state, action, reward, new_state, False)
                        loss = train_step(beta, online_model, target_model, replay_buffer, optimizer)
                        beta = min(PRIORITY_BETA_END, beta + beta_increment)
                    else:
                        loss = None

                    if step_idx % TARGET_UPDATE_FREQ == 0 and step_idx > 0 and MODE == "train":
                        target_model.load_state_dict(online_model.state_dict())

                    queue_len = sum(new_state[:-1])
                    queue_history_ep.append(queue_len)
                    prev_state = new_state

                    queue_ma = moving_avg(queue_history_ep, window=50)
                    loss_str = f"{loss:.2f}" if loss is not None else "--"

                    if step_idx % 100 == 0:
                        print(
                            f"[Ep {ep+1:3d} | Step {t:4d} | Global {step_idx:6d}] "
                            f"Mode: {MODE.upper()} | R: {reward:6.2f} | CumEp: {cumulative_reward:8.2f} | "
                            f"Queue MA50: {queue_ma:6.2f} | Loss: {loss_str}"
                        )

                    # (TensorBoard logging removed in PyTorch version)

                    # Save per-step row
                    step_rows.append(
                        {
                            "global_step": step_idx,
                            "episode": ep,
                            "step_in_episode": t,
                            "reward": reward,
                            "cumulative_reward_episode": cumulative_reward,
                            "queue_length": queue_len,
                        }
                    )

                    total_steps_global += 1

                episode_ok = True
            finally:
                # If episode crashes mid-loop, ensure SUMO is closed
                if not episode_ok:
                    traci.close()

            # Episode-level metrics (computed after loop)
            avg_queue_ep = float(np.mean(queue_history_ep)) if queue_history_ep else float("inf")
            min_queue_ep = float(np.min(queue_history_ep)) if queue_history_ep else float("inf")
            max_queue_ep = float(np.max(queue_history_ep)) if queue_history_ep else float("inf")

            # (Per-episode TensorBoard logging removed)

            # Print episode summary BEFORE closing SUMO (Option A2)
            print(
                f"Episode {ep+1} summary: "
                f"Cumulative Reward = {cumulative_reward:.2f}, "
                f"Avg Queue = {avg_queue_ep:.2f}, "
                f"Min Queue = {min_queue_ep:.2f}, "
                f"Max Queue = {max_queue_ep:.2f}"
            )

            # Save per-episode row
            episode_rows.append(
                {
                    "episode": ep,
                    "seed": episode_seed,
                    "cumulative_reward": cumulative_reward,
                    "avg_queue": avg_queue_ep,
                    "min_queue": min_queue_ep,
                    "max_queue": max_queue_ep,
                }
            )

            # Now close SUMO AFTER printing (Option A2 normal path)
            if episode_ok:
                traci.close()

            # Checkpointing by episode
            if MODE == "train" and (ep + 1) % CHECKPOINT_EVERY_EPISODES == 0:
                torch.save(online_model.state_dict(), LAST_MODEL_PATH)
                replay_buffer.save(REPLAY_PATH)
                print(f"[Checkpoint] Saved last model + replay at episode {ep+1}")

            # Best-model tracking by episode metric (negative avg queue)
            if MODE == "train":
                metric = -avg_queue_ep
                if best_metric is None or metric > best_metric or not os.path.exists(BEST_MODEL_PATH):
                    best_metric = metric
                    torch.save(online_model.state_dict(), BEST_MODEL_PATH)
                    print(f"[Best] Updated best model at episode {ep+1} (metric={metric:.4f})")

            # Save CSV + plot after EVERY episode (Option A)
            save_step_csv(step_rows)
            save_episode_csv(episode_rows)
            plot_metrics(episode_rows)

    finally:
        # Global safety: always save last model, replay, CSVs, and plot
        if MODE == "train":
            torch.save(online_model.state_dict(), LAST_MODEL_PATH)
            replay_buffer.save(REPLAY_PATH)
            print("[Final] Saved last model and replay buffer (global finally).")

        save_step_csv(step_rows)
        save_episode_csv(episode_rows)
        plot_metrics(episode_rows)

    print("\nRun complete.")
    print(online_model)


if __name__ == "__main__":
    run()
