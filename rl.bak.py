# ================================================================
# Rainbow DQN for SUMO Traffic Signal Control (PyTorch)
# Final version matching project writeup:
# - State:
#     Vehicle waiting time (per approach)
#     Vehicle queue length (per approach)
#     Pedestrians (aggregate count)
#     Pedestrian waiting time (aggregate)
#     Signal phase (one-hot)
# - Min-Max style normalization to [0, 1] for all numeric features
# - One-hot encoding for categorical phase
# - Automatic 80/20 train/eval split by episodes
# - Observation normalization, reward scaling + clipping
#     * Dueling architecture
#     * Noisy linear layers
#     * Double DQN
#     * Prioritized replay
#     * N-step returns
#     * Distributional C51
# - No frame stacking (not needed for this project)
# - Intersection logic unchanged
# - Logging aligned with baseline: queues + waits (+ throughput, switches)
# ================================================================

import os
import sys
import csv
import random
import argparse
import pickle
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# ================================================================
# Args & Modes
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
    default=50,
    help="Number of episodes to run",
)
parser.add_argument(
    "--steps_per_episode",
    type=int,
    default=10000,
    help="Number of environment steps per episode",
)
args = parser.parse_args()

MODE = args.mode
RUN_ID = args.run_id
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps_per_episode

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Automatic 80/20 split by episodes (RL-style train/eval)
TRAIN_EPISODES = max(1, int(NUM_EPISODES * 0.8))
EVAL_EPISODES = NUM_EPISODES - TRAIN_EPISODES

# ================================================================
# Directories
# ================================================================

BASE_RESULTS_DIR = "results"
RUN_DIR = os.path.join(BASE_RESULTS_DIR, RUN_ID)
os.makedirs(RUN_DIR, exist_ok=True)

CHECKPOINT_DIR = os.path.join(RUN_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

REPLAY_DIR = os.path.join(RUN_DIR, "replay")
os.makedirs(REPLAY_DIR, exist_ok=True)

RL_STEP_CSV = os.path.join(RUN_DIR, "rl_step_metrics.csv")
RL_EPISODE_CSV = os.path.join(RUN_DIR, "rl_episode_metrics.csv")
PLOT_PATH = os.path.join(RUN_DIR, "rl_combined.png")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pth")
REPLAY_PATH = os.path.join(REPLAY_DIR, "replay.pkl")

# ================================================================
# SUMO Setup
# ================================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci  # noqa: E402


def make_sumo_config():
    return [
        "sumo",
        "-c",
        "simulation/sumo/test.sumocfg",
        "--step-length",
        "0.10",
        "--delay",
        "1000",
        "--lateral-resolution",
        "0",
    ]


# ================================================================
# Hyperparameters
# ================================================================

GAMMA = 0.99
N_STEPS = 5
BUFFER_SIZE = 100000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_FREQ = 500
WARMUP_STEPS = 5000

# NoisyNet handles exploration; epsilon off by default
USE_EPSILON = False
EPSILON_START = 0.3
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 200000

ACTIONS = [0, 1]
NUM_ACTIONS = len(ACTIONS)

PRIORITY_ALPHA = 0.6
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_END = 1.0

NOISY_SIGMA = 0.2
LEARNING_RATE = 1e-4

# Min-Max style normalization ranges (approximate, project-based)
MAX_VEH_QUEUE = 200.0        # max vehicles per approach (tunable)
MAX_VEH_WAIT = 10000.0        # vehicle waiting time in seconds
MAX_PED = 100.0               # pedestrians
MAX_PED_WAIT = 10000.0        # pedestrian waiting time in seconds

# Reward scaling & clipping
REWARD_SCALE = 50.0          # scale down queue-based reward
REWARD_CLIP = 1.0            # clip reward to [-1, 1]

CHECKPOINT_EVERY_EPISODES = 1

# Distributional C51
NUM_ATOMS = 51
V_MIN = -100.0
V_MAX = 100.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

# Will be set after inspecting SUMO logic
NUM_PHASES = None

# ================================================================
# Environment Functions
# ================================================================


def get_vehicle_queue_length(detector_id: str) -> int:
    """Vehicle queue length on a lanearea detector (vehicles only)."""
    return traci.lanearea.getLastStepVehicleNumber(detector_id)


def get_vehicle_wait_time(detector_id: str) -> float:
    """Vehicle waiting time aggregated on a lanearea detector (seconds)."""
    try:
        return traci.lanearea.getWaitingTime(detector_id)
    except Exception:
        return 0.0


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
    return traci.trafficlight.getPhase(tls_id)


def get_num_phases(tls_id: str = "C") -> int:
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    return len(program.phases)


def one_hot_phase(phase_idx: int, num_phases: int) -> np.ndarray:
    vec = np.zeros(num_phases, dtype=np.float32)
    if 0 <= phase_idx < num_phases:
        vec[phase_idx] = 1.0
    return vec


def get_state():
    """
    State:
      - qN, qE, qS, qW: vehicle queue lengths (sum over detectors)
      - wN, wE, wS, wW: vehicle waiting times (sum over detectors)
      - ped_queue: total waiting pedestrians (count)
      - ped_wait: total pedestrian waiting time (seconds)
      - phase: current phase index (categorical, later one-hot)
    """
    # Vehicle queues (per approach)
    q_N = sum(get_vehicle_queue_length(f"C_NC_{i}") for i in range(1, 5))
    q_E = sum(get_vehicle_queue_length(f"C_EC_{i}") for i in range(1, 5))
    q_S = sum(get_vehicle_queue_length(f"C_SC_{i}") for i in range(1, 5))
    q_W = sum(get_vehicle_queue_length(f"C_WC_{i}") for i in range(1, 5))

    # Vehicle waiting times per approach
    w_N = sum(get_vehicle_wait_time(f"C_NC_{i}") for i in range(1, 5))
    w_E = sum(get_vehicle_wait_time(f"C_EC_{i}") for i in range(1, 5))
    w_S = sum(get_vehicle_wait_time(f"C_SC_{i}") for i in range(1, 5))
    w_W = sum(get_vehicle_wait_time(f"C_WC_{i}") for i in range(1, 5))

    ped_queue = get_pedestrian_queue_count()
    ped_wait = get_pedestrian_total_wait_time()
    phase = get_current_phase("C")

    return (
        q_N, q_E, q_S, q_W,
        w_N, w_E, w_S, w_W,
        ped_queue, ped_wait, phase,
    )


def normalize_scalar(x: float, max_val: float) -> float:
    if max_val <= 0:
        return 0.0
    return float(np.clip(x / max_val, 0.0, 1.0))


def normalize_state(s):
    """
    Min-Max style normalization to [0, 1] for all numeric features:
      - vehicle queue lengths: 0–MAX_VEH_QUEUE
      - vehicle waiting times: 0–MAX_VEH_WAIT
      - ped_queue: 0–MAX_PED
      - ped_wait: 0–MAX_PED_WAIT
      - phase: one-hot encoded (no fake ordering)
    """
    global NUM_PHASES
    (
        q_N, q_E, q_S, q_W,
        w_N, w_E, w_S, w_W,
        ped_queue, ped_wait, phase,
    ) = s

    qN_n = normalize_scalar(q_N, MAX_VEH_QUEUE)
    qE_n = normalize_scalar(q_E, MAX_VEH_QUEUE)
    qS_n = normalize_scalar(q_S, MAX_VEH_QUEUE)
    qW_n = normalize_scalar(q_W, MAX_VEH_QUEUE)

    wN_n = normalize_scalar(w_N, MAX_VEH_WAIT)
    wE_n = normalize_scalar(w_E, MAX_VEH_WAIT)
    wS_n = normalize_scalar(w_S, MAX_VEH_WAIT)
    wW_n = normalize_scalar(w_W, MAX_VEH_WAIT)

    ped_queue_n = normalize_scalar(ped_queue, MAX_PED)
    ped_wait_n = normalize_scalar(ped_wait, MAX_PED_WAIT)

    if NUM_PHASES is None:
        NUM_PHASES = 1
    phase_oh = one_hot_phase(int(phase), NUM_PHASES)

    return np.concatenate(
        [
            np.array(
                [
                    qN_n, qE_n, qS_n, qW_n,
                    wN_n, wE_n, wS_n, wW_n,
                    ped_queue_n, ped_wait_n,
                ],
                dtype=np.float32,
            ),
            phase_oh,
        ],
        axis=0,
    )


PED_WEIGHT = 3.0   # or 2.0–4.0 as a reasonable band


def get_reward(state, prev_state=None):
    """
    Reward:
      - Based on total queue (vehicles + pedestrians)
      - Scaled and clipped for stability
      - Includes shaping term for reduction in total queue
      - NOTE: reward uses ped_queue, not ped_wait, to stay aligned with baseline
    """
    qN, qE, qS, qW, wN, wE, wS, wW, ped_queue, ped_wait, phase = state
    vehicle_queue = qN + qE + qS + qW
    total_queue = vehicle_queue + ped_queue

    reward = -float(total_queue) / REWARD_SCALE

    if prev_state is not None:
        (
            pqN, pqE, pqS, pqW,
            p_wN, p_wE, p_wS, p_wW,
            p_ped_queue, p_ped_wait, p_phase,
        ) = prev_state
        prev_vehicle_queue = pqN + pqE + pqS + pqW
        prev_total_queue = prev_vehicle_queue + p_ped_queue
        reward += 0.25 * ((prev_total_queue - total_queue) / REWARD_SCALE)

    reward = float(np.clip(reward, -REWARD_CLIP, REWARD_CLIP))
    return reward


def apply_action_safe(action, tls_id: str = "C"):
    """
    Intersection logic unchanged:
      - Only switch when current phase is green (no yellow)
      - Action 0 = keep, 1 = switch
    """
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phase = get_current_phase(tls_id)
    state = program.phases[phase].state

    # can only switch if in phase with one or more green lights (no yellows), otherwise keep
    if "y" in state or ("G" not in state and "g" not in state):
        return

    if action == 1:
        traci.trafficlight.setPhase(tls_id, (phase + 1) % len(program.phases))


# ================================================================
# Noisy Linear Layer
# ================================================================


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        mu_range = 1 / np.sqrt(in_features)
        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).uniform_(-mu_range, mu_range)
        )
        self.weight_sigma = nn.Parameter(
            torch.full((out_features, in_features), sigma_init / np.sqrt(in_features))
        )

        self.bias_mu = nn.Parameter(
            torch.empty(out_features).uniform_(-mu_range, mu_range)
        )
        self.bias_sigma = nn.Parameter(
            torch.full((out_features,), sigma_init / np.sqrt(in_features))
        )

    def forward(self, x):
        if self.training:
            eps_in = torch.randn(self.in_features, device=x.device)
            eps_out = torch.randn(self.out_features, device=x.device)

            f_in = torch.sign(eps_in) * torch.sqrt(torch.abs(eps_in))
            f_out = torch.sign(eps_out) * torch.sqrt(torch.abs(eps_out))

            w_noise = torch.ger(f_out, f_in)
            b_noise = f_out

            weight = self.weight_mu + self.weight_sigma * w_noise
            bias = self.bias_mu + self.bias_sigma * b_noise
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return x @ weight.t() + bias


# ================================================================
# Rainbow DQN Model (Dueling + Noisy + Distributional C51)
# ================================================================


class RainbowDQN(nn.Module):
    def __init__(self, state_size, num_actions, num_atoms, v_min, v_max):
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream
        self.v_noisy1 = NoisyLinear(128, 128)
        self.v_relu = nn.ReLU()
        self.v_noisy2 = NoisyLinear(128, num_atoms)

        # Advantage stream
        self.a_noisy1 = NoisyLinear(128, 128)
        self.a_relu = nn.ReLU()
        self.a_noisy2 = NoisyLinear(128, num_actions * num_atoms)

        self.relu = nn.ReLU()

        # Support
        self.register_buffer(
            "support",
            torch.linspace(self.v_min, self.v_max, self.num_atoms)
        )

    def forward(self, x):
        """
        Returns logits over atoms for each action:
        shape: (batch, num_actions, num_atoms)
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        v = self.v_noisy1(x)
        v = self.v_relu(v)
        v = self.v_noisy2(v)  # (batch, num_atoms)

        a = self.a_noisy1(x)
        a = self.a_relu(a)
        a = self.a_noisy2(a)  # (batch, num_actions * num_atoms)
        a = a.view(-1, self.num_actions, self.num_atoms)

        v = v.view(-1, 1, self.num_atoms)
        a_mean = a.mean(dim=1, keepdim=True)

        q_logits = v + (a - a_mean)  # (batch, num_actions, num_atoms)
        return q_logits

    def q_values(self, x):
        """
        Returns expected Q-values by integrating over the support.
        shape: (batch, num_actions)
        """
        logits = self.forward(x)  # (batch, num_actions, num_atoms)
        probs = torch.softmax(logits, dim=-1)
        q = torch.sum(probs * self.support, dim=-1)
        return q


def build_rainbow_model(state_size, num_actions):
    model = RainbowDQN(
        state_size,
        num_actions,
        NUM_ATOMS,
        V_MIN,
        V_MAX,
    ).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    return model, optimizer


# ================================================================
# Prioritized Replay Buffer
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

        R = sum(
            (self.gamma ** i) * r_i
            for i, (_, _, r_i, _, _) in enumerate(self.n_step_buffer)
        )
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

    def flush(self):
        while len(self.n_step_buffer) > 0:
            R = sum(
                (self.gamma ** i) * r_i
                for i, (_, _, r_i, _, _) in enumerate(self.n_step_buffer)
            )
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

    def save(self, path):
        state = ReplayState(
            buffer=self.buffer,
            pos=self.pos,
            priorities=self.priorities,
            n_step_buffer=self.n_step_buffer,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state: ReplayState = pickle.load(f)
        self.buffer = state.buffer
        self.pos = state.pos
        self.priorities = state.priorities
        self.n_step_buffer = state.n_step_buffer


# ================================================================
# Init Models & Replay
# ================================================================


def init_models_and_replay():
    global NUM_PHASES

    traci.start(make_sumo_config())
    dummy_state = get_state()
    NUM_PHASES = get_num_phases("C")
    traci.close()

    dummy_norm = normalize_state(dummy_state)
    state_size = len(dummy_norm)

    online_model, online_optimizer = build_rainbow_model(state_size, NUM_ACTIONS)
    target_model, _ = build_rainbow_model(state_size, NUM_ACTIONS)
    target_model.load_state_dict(online_model.state_dict())

    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, PRIORITY_ALPHA, N_STEPS, GAMMA)

    if os.path.exists(LAST_MODEL_PATH):
        state_dict = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
        online_model.load_state_dict(state_dict)
        target_model.load_state_dict(state_dict)
        print(f"Loaded last model from {LAST_MODEL_PATH}")

    if os.path.exists(REPLAY_PATH):
        replay_buffer.load(REPLAY_PATH)
        print(f"Loaded replay buffer from {REPLAY_PATH}")

    return online_model, target_model, replay_buffer, online_optimizer


best_metric = None
if os.path.exists(BEST_MODEL_PATH):
    print(f"Best model already exists at {BEST_MODEL_PATH}")


# ================================================================
# Action Selection
# ================================================================


def select_action(state, step, mode, online_model):
    s_norm = normalize_state(state).reshape(1, -1)
    s_tensor = torch.from_numpy(s_norm).float().to(DEVICE)

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

    online_model.eval()
    with torch.no_grad():
        q_vals = online_model.q_values(s_tensor)[0].cpu().numpy()
    return int(np.argmax(q_vals))


# ================================================================
# Distributional Projection (C51)
# ================================================================


def projection_distribution(next_dist, rewards, dones, gamma_n, support):
    """
    next_dist: (batch, num_atoms) target probs for chosen next action
    rewards: (batch,)
    dones: (batch,)
    support: (num_atoms,)
    """
    batch_size = rewards.size(0)
    num_atoms = support.size(0)

    rewards = rewards.unsqueeze(1)
    dones = dones.unsqueeze(1)

    tz = rewards + (1.0 - dones) * gamma_n * support.unsqueeze(0)
    tz = tz.clamp(V_MIN, V_MAX)

    b = (tz - V_MIN) / DELTA_Z
    l = b.floor().long()
    u = b.ceil().long()

    l = l.clamp(0, num_atoms - 1)
    u = u.clamp(0, num_atoms - 1)

    proj_dist = torch.zeros(batch_size, num_atoms, device=next_dist.device)

    offset = torch.linspace(
        0,
        (batch_size - 1) * num_atoms,
        batch_size,
        device=next_dist.device,
    ).long().unsqueeze(1)

    proj_dist.view(-1).index_add_(
        0,
        (l + offset).view(-1),
        (next_dist * (u.float() - b)).view(-1),
    )
    proj_dist.view(-1).index_add_(
        0,
        (u + offset).view(-1),
        (next_dist * (b - l.float())).view(-1),
    )

    return proj_dist


# ================================================================
# Training Step
# ================================================================


def train_step(beta, online_model, target_model, replay_buffer, optimizer):
    if MODE != "train":
        return None
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(
        BATCH_SIZE, beta
    )

    states_norm = np.array([normalize_state(s) for s in states], dtype=np.float32)
    next_states_norm = np.array([normalize_state(s) for s in next_states], dtype=np.float32)

    states_tensor = torch.from_numpy(states_norm).float().to(DEVICE)
    next_states_tensor = torch.from_numpy(next_states_norm).float().to(DEVICE)
    actions_tensor = torch.from_numpy(actions).long().to(DEVICE)
    rewards_tensor = torch.from_numpy(rewards).float().to(DEVICE)
    dones_tensor = torch.from_numpy(dones.astype(np.float32)).float().to(DEVICE)
    weights_tensor = torch.from_numpy(weights).float().to(DEVICE)

    gamma_n = GAMMA ** N_STEPS

    # Target distribution
    online_model.eval()
    target_model.eval()
    with torch.no_grad():
        # Next-state Q-values (for Double DQN)
        next_q_values = online_model.q_values(next_states_tensor)  # (batch, num_actions)
        next_actions = next_q_values.argmax(dim=1)  # (batch,)

        # Target logits and probs
        target_logits = target_model(next_states_tensor)  # (batch, num_actions, num_atoms)
        target_logits = target_logits.gather(
            1, next_actions.view(-1, 1, 1).expand(-1, 1, NUM_ATOMS)
        ).squeeze(1)  # (batch, num_atoms)
        target_probs = torch.softmax(target_logits, dim=-1)  # (batch, num_atoms)

        support = target_model.support  # (num_atoms,)
        proj_dist = projection_distribution(
            target_probs,
            rewards_tensor,
            dones_tensor,
            gamma_n,
            support,
        )  # (batch, num_atoms)

    # Predicted distribution for taken actions
    online_model.train()
    logits = online_model(states_tensor)  # (batch, num_actions, num_atoms)
    logits = logits.gather(
        1, actions_tensor.view(-1, 1, 1).expand(-1, 1, NUM_ATOMS)
    ).squeeze(1)  # (batch, num_atoms)

    log_probs = torch.log_softmax(logits, dim=-1)  # (batch, num_atoms)
    probs = torch.softmax(logits, dim=-1)

    # Cross-entropy loss
    loss_per_sample = -(proj_dist * log_probs).sum(dim=1)  # (batch,)
    loss = (weights_tensor * loss_per_sample).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # TD-errors for PER (based on expected values)
    with torch.no_grad():
        q_pred = torch.sum(probs * support, dim=1)  # (batch,)
        q_target = torch.sum(proj_dist * support, dim=1)  # (batch,)
        td_errors = (q_target - q_pred).detach().cpu().numpy()

    replay_buffer.update_priorities(idxs, td_errors)
    return float(loss.item())


# ================================================================
# Helpers
# ================================================================


def moving_avg(data, window=50):
    if len(data) < window:
        return float(np.mean(data)) if data else 0.0
    return float(np.mean(data[-window:]))


def save_step_csv(step_rows):
    if not step_rows:
        return
    with open(RL_STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=step_rows[0].keys())
        writer.writeheader()
        writer.writerows(step_rows)


def save_episode_csv(episode_rows):
    if not episode_rows:
        return
    with open(RL_EPISODE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(episode_rows)


def plot_metrics(episode_rows):
    if not episode_rows:
        return

    eps = [r["episode"] for r in episode_rows]
    cum_rewards = [r["cumulative_reward"] for r in episode_rows]
    avg_total = [r["avg_total_queue"] for r in episode_rows]

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(eps, cum_rewards, marker="o")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative Reward")
    plt.title(f"Episode Metrics ({MODE.upper()} - {RUN_ID})")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(eps, avg_total, marker="o", color="orange")
    plt.xlabel("Episode")
    plt.ylabel("Avg Total Queue (veh + ped)")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()


# ================================================================
# Main Loop (online training every step, 80/20 split)
# ================================================================


def run():
    global best_metric

    online_model, target_model, replay_buffer, optimizer = init_models_and_replay()

    # For CSV logging
    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    total_steps_global = 0
    total_grad_steps_global = 0

    # Schedule beta over approximate total gradient steps
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
            # Determine per-episode mode based on 80/20 split
            if MODE == "train":
                if ep < TRAIN_EPISODES:
                    ep_mode = "train"
                else:
                    ep_mode = "eval"
            else:
                ep_mode = MODE  # "eval" or "infer" from CLI

            traci.start(make_sumo_config())

            cumulative_reward = 0.0

            vehicle_queue_hist: List[float] = []
            ped_queue_hist: List[float] = []
            total_queue_hist: List[float] = []

            vehicle_wait_hist: List[float] = []
            ped_wait_hist: List[float] = []
            total_wait_hist: List[float] = []

            # Throughput + switching
            vehicle_throughput = 0
            ped_throughput = 0
            switch_count = 0
            prev_phase = get_current_phase("C")

            prev_state = None
            episode_ok = False

            try:
                for t in range(STEPS_PER_EPISODE):
                    step_idx = total_steps_global

                    # IDs before step for throughput
                    prev_vehicle_ids = set(traci.vehicle.getIDList())
                    prev_ped_ids = set(traci.person.getIDList())

                    state = get_state()
                    action = select_action(state, step_idx, ep_mode, online_model)
                    apply_action_safe(action, "C")

                    # Count phase switches
                    cur_phase = get_current_phase("C")
                    if cur_phase != prev_phase:
                        switch_count += 1
                        prev_phase = cur_phase

                    # Advance simulation
                    traci.simulationStep()

                    next_state = get_state()
                    reward = get_reward(next_state, state)
                    cumulative_reward += reward

                    # Queues and waits from next_state
                    (
                        qN, qE, qS, qW,
                        wN, wE, wS, wW,
                        ped_queue, ped_wait, phase,
                    ) = next_state

                    vehicle_queue = qN + qE + qS + qW
                    total_queue = vehicle_queue + ped_queue
                    vehicle_wait = wN + wE + wS + wW
                    total_wait = vehicle_wait + ped_wait

                    vehicle_queue_hist.append(vehicle_queue)
                    ped_queue_hist.append(ped_queue)
                    total_queue_hist.append(total_queue)

                    vehicle_wait_hist.append(vehicle_wait)
                    ped_wait_hist.append(ped_wait)
                    total_wait_hist.append(total_wait)

                    # Throughput: who left between steps
                    cur_vehicle_ids = set(traci.vehicle.getIDList())
                    cur_ped_ids = set(traci.person.getIDList())
                    vehicle_throughput += len(prev_vehicle_ids - cur_vehicle_ids)
                    ped_throughput += len(prev_ped_ids - cur_ped_ids)

                    # Store transition and train online (only in train episodes)
                    if ep_mode == "train":
                        replay_buffer.add(state, action, reward, next_state, False)

                        loss = train_step(beta, online_model, target_model, replay_buffer, optimizer)
                        if loss is not None:
                            total_grad_steps_global += 1
                            beta = min(1.0, beta + beta_increment)

                            step_rows.append(
                                {
                                    "global_step": total_steps_global,
                                    "episode": ep,
                                    "t": t,
                                    "mode": ep_mode,
                                    "loss": loss,
                                    "vehicle_queue": vehicle_queue,
                                    "ped_queue": ped_queue,
                                    "total_queue": total_queue,
                                    "vehicle_wait": vehicle_wait,
                                    "ped_wait": ped_wait,
                                    "total_wait": total_wait,
                                    "reward": reward,
                                    "action": action,
                                }
                            )

                            # Target network update
                            if total_grad_steps_global % TARGET_UPDATE_FREQ == 0:
                                target_model.load_state_dict(online_model.state_dict())

                    total_steps_global += 1

                episode_ok = True

            except Exception as e:
                print(f"[Episode {ep}] Exception during simulation: {e}")
            finally:
                try:
                    traci.close()
                except Exception:
                    pass

            # Flush remaining n-step buffer at episode end
            replay_buffer.flush()

            # Episode-level metrics
            avg_vehicle_queue = float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else 0.0
            avg_ped_queue = float(np.mean(ped_queue_hist)) if ped_queue_hist else 0.0
            avg_total_queue = float(np.mean(total_queue_hist)) if total_queue_hist else 0.0

            avg_vehicle_wait = float(np.mean(vehicle_wait_hist)) if vehicle_wait_hist else 0.0
            avg_ped_wait = float(np.mean(ped_wait_hist)) if ped_wait_hist else 0.0
            avg_total_wait = float(np.mean(total_wait_hist)) if total_wait_hist else 0.0

            # Episode-level metrics
            avg_vehicle_queue = float(np.mean(vehicle_queue_hist)) if vehicle_queue_hist else 0.0
            avg_ped_queue = float(np.mean(ped_queue_hist)) if ped_queue_hist else 0.0
            avg_total_queue = float(np.mean(total_queue_hist)) if total_queue_hist else 0.0

            episode_rows.append(
                {
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
                }
            )

            print(
                f"[Ep {ep:04d} | {ep_mode}] "
                f"R={cumulative_reward:.2f} | "
                f"Q_tot={avg_total_queue:.2f} | "
                f"W_tot={avg_total_wait:.2f} | "
                f"veh_thru={vehicle_throughput} | ped_thru={ped_throughput} | "
                f"switches={switch_count}"
            )
            
            # ============================================================
            # SAVE RESULTS AFTER EACH EPISODE (CSV + PLOT)
            # ============================================================
            save_step_csv(step_rows)
            save_episode_csv(episode_rows)
            plot_metrics(episode_rows)

            # Checkpointing (based on avg_total_queue: lower is better)
            if ep_mode == "train" and episode_ok:
                metric = avg_total_queue
                if best_metric is None or metric < best_metric:
                    best_metric = metric
                    torch.save(online_model.state_dict(), BEST_MODEL_PATH)
                    print(f"  -> New best model saved (avg_total_queue={metric:.3f})")

                if ep % CHECKPOINT_EVERY_EPISODES == 0:
                    torch.save(online_model.state_dict(), LAST_MODEL_PATH)
                    replay_buffer.save(REPLAY_PATH)

        # Final save
        torch.save(online_model.state_dict(), LAST_MODEL_PATH)
        replay_buffer.save(REPLAY_PATH)

    finally:
        try:
            traci.close()
        except Exception:
            pass


if __name__ == "__main__":
    run()
