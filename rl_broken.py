"""
BTR-style Vectorized IQN + Munchausen DQN for SUMO Traffic Signal Control (PyTorch + DirectML)
---------------------------------------------------------------------------------------------

Core features (Beyond The Rainbow style, adapted to low-dim SUMO state):

- Vectorized environments (multiple parallel SUMO instances per episode batch)
- Implicit Quantile Networks (IQN) instead of C51
- Munchausen RL (Soft-DQN target, no Double)
- Spectral Normalization on dense layers
- Noisy Networks + (optional) epsilon-greedy (epsilon disabled later)
- N-step TD learning
- Prioritized Experience Replay (PER) with |TD| priority
- BTR-style replay ratio (tunable, default 1 update per NUM_ENVS env transitions)
- Target network hard update every 500 gradient steps
- Clean state normalization
- TRAIN / EVAL / INFER modes (via --mode)
- Episode-based training, batched in parallel across NUM_ENVS envs
- New SUMO seed each episode (to avoid overfitting one scenario)
- Persistent training, crash recovery
- Checkpointing + best-model tracking (by episode metric)
- Replay buffer persistence
- CSV logging + simple plotting
- Deterministic base seed + controlled episode variation
- Multi-run scalability via --run_id

Note: Impala CNN + adaptive max pooling is not used here because SUMO state is low-dimensional (queues + phase).
"""

import os
import sys
import csv
import random
import argparse
import pickle
from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

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
    default="btr_iqn_munchausen_vec_pt",
    help="Run identifier for checkpoints/logs",
)
parser.add_argument(
    "--episodes",
    type=int,
    default=64,
    help="Total number of episodes to run (across all envs)",
)
parser.add_argument(
    "--steps_per_episode",
    type=int,
    default=1000,
    help="Number of environment steps per episode",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=8,
    help="Number of parallel SUMO environments (vectorized)",
)
parser.add_argument(
    "--replay_ratio",
    type=float,
    default=8.0,
    help="Replay ratio: updates per NUM_ENVS environment transitions (BTR-style). "
         "1.0 = 1 update per NUM_ENVS steps; >1 = more updates.",
)
args = parser.parse_args()

MODE = args.mode
RUN_ID = args.run_id
NUM_EPISODES = args.episodes
STEPS_PER_EPISODE = args.steps_per_episode
NUM_ENVS = args.num_envs
REPLAY_RATIO = max(args.replay_ratio, 0.0)

# ================================================================
#  Device selection (CUDA + DirectML + CPU)
# ================================================================

def get_device():
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        print(f"[Startup] Using CUDA GPU: {torch.cuda.get_device_name(0)}")
        return dev
    try:
        import torch_directml
        dev = torch_directml.device()
        print("[Startup] Using DirectML device (torch-directml).")
        return dev
    except Exception:
        pass
    print("[Startup] No GPU found, using CPU.")
    return torch.device("cpu")


DEVICE = get_device()

print("=== BTR-style Vectorized IQN + Munchausen DQN for SUMO (PyTorch) ===")
print(f"[Startup] DEVICE={DEVICE}")

# ================================================================
#  Deterministic Base Seed
# ================================================================

GLOBAL_SEED = 12345
os.environ["PYTHONHASHSEED"] = str(GLOBAL_SEED)
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

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

RL_STEP_CSV = os.path.join(RUN_DIR, "rl_step_metrics.csv")
RL_EPISODE_CSV = os.path.join(RUN_DIR, "rl_episode_metrics.csv")
PLOT_PATH = os.path.join(RUN_DIR, "rl_combined.png")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model.pt")
REPLAY_PATH = os.path.join(REPLAY_DIR, "replay.pkl")

# ================================================================
#  SUMO Setup
# ================================================================

if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)

import traci  # noqa: E402


def make_sumo_config(seed: int) -> List[str]:
    binary = "sumo"  # headless for speed
    return [
        binary,
        "-c",
        "simulation/sumo/test.sumocfg",
        "--step-length",
        "0.10",
        "--delay",
        "0",
        "--lateral-resolution",
        "0",
        "--seed",
        str(seed),
    ]


# ================================================================
#  Hyperparameters (BTR-style)
# ================================================================

GAMMA = 0.997
N_STEPS = 5

BUFFER_SIZE = 200_000
BATCH_SIZE = 512
MIN_REPLAY_SIZE = 10_000

REPLAY_BASE_ENVS = NUM_ENVS
REPLAY_RATIO_ACCUM_INIT = 0.0

USE_EPSILON = False
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 5_000_000
EPSILON_DISABLE_STEP = 1_000_000

ACTIONS = [0, 1]
NUM_ACTIONS = len(ACTIONS)

PRIORITY_ALPHA = 0.2
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_END = 1.0

NOISY_SIGMA = 0.5

QUEUE_NORM = 20.0
PHASE_NORM = 10.0

CHECKPOINT_EVERY_EPISODES = 5
TARGET_UPDATE_EVERY = 500

# ================================================================
#  IQN Hyperparameters
# ================================================================

NUM_QUANTILES = 128
NUM_QUANTILES_TARGET = 128
EMBEDDING_DIM = 64

# ================================================================
#  Munchausen Hyperparameters
# ================================================================

MUNCHAUSEN_ALPHA = 0.9
MUNCHAUSEN_TAU = 0.03
MUNCHAUSEN_CLIP = -1.0

# ================================================================
#  Learning Rate (BTR-style)
# ================================================================

LR = 1e-4

# ================================================================
#  Environment Functions (per-connection)
# ================================================================


def get_queue_length(detector_id: str, tc) -> int:
    return tc.lanearea.getLastStepVehicleNumber(detector_id)


def get_current_phase(tc, tls_id="C") -> int:
    return tc.trafficlight.getPhase(tls_id)


def get_state(tc) -> Tuple[int, int, int, int, int]:
    q_N = sum(get_queue_length(f"C_NC_{i}", tc) for i in range(1, 5))
    q_E = sum(get_queue_length(f"C_EC_{i}", tc) for i in range(1, 5))
    q_S = sum(get_queue_length(f"C_SC_{i}", tc) for i in range(1, 5))
    q_W = sum(get_queue_length(f"C_WC_{i}", tc) for i in range(1, 5))
    phase = get_current_phase(tc, "C")
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
        reward += 0.25 * (prev_q - total_queue)
    return reward


def apply_action_safe(action, tc, tls_id="C"):
    program = tc.trafficlight.getAllProgramLogics(tls_id)[0]
    phase = get_current_phase(tc, tls_id)
    state = program.phases[phase].state

    if "y" in state or ("G" not in state and "g" not in state):
        return

    if action == 1:
        tc.trafficlight.setPhase(tls_id, (phase + 1) % len(program.phases))


# ================================================================
#  Spectral Normalization Dense (PyTorch)
# ================================================================

class SpectralNormDense(nn.Module):
    def __init__(self, in_dim, out_dim, activation="relu", use_bias=True):
        super().__init__()
        linear = nn.Linear(in_dim, out_dim, bias=use_bias)
        self.linear = nn.utils.spectral_norm(linear)
        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        out = self.linear(x)
        if self.activation is not None:
            out = self.activation(out)
        return out


# ================================================================
#  Noisy Dense (for IQN head, PyTorch)
# ================================================================

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=NOISY_SIGMA, activation="relu"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        mu_range = 1 / np.sqrt(in_features)
        self.weight_mu = nn.Parameter(torch.empty(in_features, out_features).uniform_(-mu_range, mu_range))
        self.weight_sigma = nn.Parameter(torch.full((in_features, out_features), sigma_init / np.sqrt(in_features)))
        self.bias_mu = nn.Parameter(torch.empty(out_features).uniform_(-mu_range, mu_range))
        self.bias_sigma = nn.Parameter(torch.full((out_features,), sigma_init / np.sqrt(in_features)))

        self.activation = None
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

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

        out = x @ weight + bias
        if self.activation is not None:
            out = self.activation(out)
        return out


# ================================================================
#  IQN Network (state + taus -> quantiles) in PyTorch
# ================================================================

class IQN_Munchausen(nn.Module):
    def __init__(self, state_size, num_actions):
        super().__init__()
        self.state_size = state_size
        self.num_actions = num_actions

        self.fc1 = SpectralNormDense(state_size, 256, activation="relu")
        self.fc2 = SpectralNormDense(256, 256, activation="relu")

        self.embedding_dim = EMBEDDING_DIM
        self.num_quantiles = NUM_QUANTILES

        self.phi = SpectralNormDense(EMBEDDING_DIM, 256, activation="relu")

        self.noisy1 = NoisyLinear(256, 256, activation="relu")
        self.noisy2 = NoisyLinear(256, num_actions, activation=None)

        k = torch.arange(1, self.embedding_dim + 1, dtype=torch.float32).view(1, 1, -1)
        self.register_buffer("k", k)

    def forward(self, states, taus):
        B = states.size(0)
        N = taus.size(1)

        x = self.fc1(states)
        x = self.fc2(x)  # (B, 256)

        tau_expanded = taus.unsqueeze(-1)  # (B, N, 1)
        cos_emb = torch.cos(np.pi * self.k * tau_expanded)  # (B, N, E)

        cos_emb_flat = cos_emb.view(B * N, -1)  # (B*N, E)
        phi_out = self.phi(cos_emb_flat)       # (B*N, 256)
        phi_out = phi_out.view(B, N, 256)      # (B, N, 256)

        x_tiled = x.unsqueeze(1).expand(-1, N, -1)  # (B, N, 256)
        h = x_tiled * phi_out                       # (B, N, 256)

        h_flat = h.view(B * N, 256)                 # (B*N, 256)
        h_flat = self.noisy1(h_flat)
        q = self.noisy2(h_flat)                     # (B*N, A)
        q = q.view(B, N, self.num_actions)          # (B, N, A)
        return q


# ================================================================
#  Prioritized Replay Buffer (with N-step)
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

    def _append_transition(self, s0, a0, R, sN, dN):
        if len(self.buffer) < self.capacity:
            self.buffer.append((s0, a0, R, sN, dN))
        else:
            self.buffer[self.pos] = (s0, a0, R, sN, dN)

        if len(self.buffer) > 0:
            max_prio = self.priorities[: len(self.buffer)].max()
        else:
            max_prio = 1.0
        max_prio = max(max_prio, 1.0)

        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def add(self, s, a, r, s_next, done):
        self.n_step_buffer.append((s, a, r, s_next, done))
        if len(self.n_step_buffer) < self.n_steps:
            return

        R = 0.0
        for i, (_, _, r_i, _, _) in enumerate(self.n_step_buffer):
            R += (self.gamma ** i) * r_i

        s0, a0, _, _, _ = self.n_step_buffer[0]
        _, _, _, sN, dN = self.n_step_buffer[-1]

        self._append_transition(s0, a0, R, sN, dN)
        self.n_step_buffer.pop(0)

    def flush(self):
        while len(self.n_step_buffer) > 0:
            R = 0.0
            for i, (_, _, r_i, _, _) in enumerate(self.n_step_buffer):
                R += (self.gamma ** i) * r_i

            s0, a0, _, _, _ = self.n_step_buffer[0]
            _, _, _, sN, dN = self.n_step_buffer[-1]

            self._append_transition(s0, a0, R, sN, dN)
            self.n_step_buffer.pop(0)

    def sample(self, batch_size, beta):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[: len(self.buffer)]
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
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32),
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
    print("[Init] Spinning up a single SUMO instance on CPU to infer state size...")
    traci.start(make_sumo_config(GLOBAL_SEED))
    dummy_state = get_state(traci)
    traci.close()

    state_size = len(dummy_state)
    print(f"[Init] Inferred state_size={state_size} from SUMO (CPU).")

    online_model = IQN_Munchausen(state_size, NUM_ACTIONS).to(DEVICE)
    target_model = IQN_Munchausen(state_size, NUM_ACTIONS).to(DEVICE)
    target_model.load_state_dict(online_model.state_dict())

    optimizer = Adam(online_model.parameters(), lr=LR)

    print("[Init] Online model:")
    print(online_model)
    print("[Init] Target model:")
    print(target_model)

    replay_buffer = PrioritizedReplayBuffer(BUFFER_SIZE, PRIORITY_ALPHA, N_STEPS, GAMMA)
    replay_buffer.load(REPLAY_PATH)

    if os.path.exists(LAST_MODEL_PATH):
        print(f"[Init] Loading last model from {LAST_MODEL_PATH} onto {DEVICE}...")
        ckpt = torch.load(LAST_MODEL_PATH, map_location=DEVICE)
        online_model.load_state_dict(ckpt["online"])
        target_model.load_state_dict(ckpt["target"])
        optimizer.load_state_dict(ckpt["optimizer"])
        print("[Init] Loaded model checkpoint.")

    return online_model, target_model, optimizer, replay_buffer


best_metric = None
if os.path.exists(BEST_MODEL_PATH):
    print(f"[Init] Best model already exists at {BEST_MODEL_PATH}")

# ================================================================
#  Action Selection (single-state helper; kept for debugging)
# ================================================================

def select_action(state, step, mode: str, online_model):
    s_norm = normalize_state(state).reshape(1, -1)
    s_tensor = torch.from_numpy(s_norm).float().to(DEVICE)

    if mode == "train":
        if USE_EPSILON and step < EPSILON_DISABLE_STEP:
            eps_frac = min(step / EPSILON_DECAY_STEPS, 1.0)
            eps = EPSILON_START + eps_frac * (EPSILON_END - EPSILON_START)
            if random.random() < eps:
                return random.choice(ACTIONS)

    taus = np.random.rand(1, NUM_QUANTILES).astype(np.float32)
    taus_tensor = torch.from_numpy(taus).float().to(DEVICE)

    online_model.eval()
    with torch.no_grad():
        q_quantiles = online_model(s_tensor, taus_tensor)  # (1, N, A)
        q_mean = q_quantiles.mean(dim=1).cpu().numpy()[0]
    online_model.train()
    return int(np.argmax(q_mean))


# ================================================================
#  Training Step (IQN + Munchausen + PER, PyTorch)
# ================================================================

grad_step_counter = 0


def train_step(beta, online_model, target_model, optimizer, replay_buffer, global_step) -> Optional[float]:
    global grad_step_counter

    if MODE != "train":
        return None
    if len(replay_buffer) < MIN_REPLAY_SIZE:
        return None

    states, actions, rewards, next_states, dones, idxs, weights = replay_buffer.sample(BATCH_SIZE, beta)

    states_norm = np.array([normalize_state(s) for s in states], dtype=np.float32)
    next_states_norm = np.array([normalize_state(s) for s in next_states], dtype=np.float32)

    states_tf = torch.from_numpy(states_norm).float().to(DEVICE)
    next_states_tf = torch.from_numpy(next_states_norm).float().to(DEVICE)
    actions_tf = torch.from_numpy(actions).long().to(DEVICE)
    rewards_tf = torch.from_numpy(rewards).float().to(DEVICE)
    dones_tf = torch.from_numpy(dones).float().to(DEVICE)
    weights_tf = torch.from_numpy(weights).float().to(DEVICE)

    gamma_n = GAMMA ** N_STEPS

    online_model.train()
    target_model.eval()

    optimizer.zero_grad()

    taus = torch.rand(BATCH_SIZE, NUM_QUANTILES, device=DEVICE)
    taus_target = torch.rand(BATCH_SIZE, NUM_QUANTILES_TARGET, device=DEVICE)

    q_quantiles = online_model(states_tf, taus)  # (B, N, A)
    q_a_quantiles = q_quantiles.gather(2, actions_tf.view(-1, 1, 1).expand(-1, NUM_QUANTILES, 1)).squeeze(-1)

    with torch.no_grad():
        q_next_quantiles = target_model(next_states_tf, taus_target)  # (B, N_t, A)
        q_next_mean = q_next_quantiles.mean(dim=1)                    # (B, A)

        logits_next = q_next_mean / MUNCHAUSEN_TAU
        log_pi_next = F.log_softmax(logits_next, dim=1)
        pi_next = log_pi_next.exp()
        v_next = (pi_next * q_next_mean).sum(dim=1)                   # (B,)

        q_curr_mean = q_quantiles.mean(dim=1)                         # (B, A)
        logits_curr = q_curr_mean / MUNCHAUSEN_TAU
        log_pi_curr = F.log_softmax(logits_curr, dim=1)

        log_pi_a = log_pi_curr.gather(1, actions_tf.view(-1, 1)).squeeze(1)
        log_pi_a_clipped = torch.clamp(log_pi_a, MUNCHAUSEN_CLIP, 0.0)

        r_tilde = rewards_tf + MUNCHAUSEN_ALPHA * log_pi_a_clipped

        targets = r_tilde + (1.0 - dones_tf) * gamma_n * v_next
        targets = targets.detach()

    targets_expanded = targets.view(-1, 1).expand(-1, NUM_QUANTILES)
    td_errors = targets_expanded - q_a_quantiles  # (B, N)

    huber_loss = torch.where(
        td_errors.abs() <= 1.0,
        0.5 * td_errors.pow(2),
        1.0 * (td_errors.abs() - 0.5),
    )

    taus_expanded = taus.unsqueeze(2)  # (B, N, 1)
    td_sign = (td_errors < 0.0).float().unsqueeze(2)  # (B, N, 1)
    quantile_weight = (taus_expanded - td_sign).abs()  # (B, N, 1)

    huber_loss_expanded = huber_loss.unsqueeze(2)      # (B, N, 1)
    quantile_loss = (quantile_weight * huber_loss_expanded).sum(dim=1).squeeze(1)  # (B,)

    loss = (weights_tf * quantile_loss).mean()

    loss_val = float(loss.item())
    if not np.isfinite(loss_val):
        print("[Train] Non-finite loss detected. Skipping update.")
        return None

    loss.backward()
    torch.nn.utils.clip_grad_norm_(online_model.parameters(), 10.0)
    optimizer.step()

    td_abs = td_errors.abs().mean(dim=1).detach().cpu().numpy()
    replay_buffer.update_priorities(idxs, td_abs)

    grad_step_counter += 1
    if grad_step_counter % TARGET_UPDATE_EVERY == 0:
        target_model.load_state_dict(online_model.state_dict())
        print(f"[Train] Target network hard-updated at grad_step={grad_step_counter} on {DEVICE}")

    if grad_step_counter % 500 == 0:
        print(f"[Train] step={global_step}, grad_step={grad_step_counter}, loss={loss_val:.4f}")

    return loss_val


# ================================================================
#  Helpers: moving avg, CSV, plotting
# ================================================================

def moving_avg(data, window=50):
    if len(data) < window:
        return float(np.mean(data)) if data else 0.0
    return float(np.mean(data[-window:]))


def save_step_csv(step_rows: List[Dict[str, Any]]):
    if not step_rows:
        return
    fieldnames = [
        "global_step",
        "batch_idx",
        "env_label",
        "episode",
        "step_in_episode",
        "reward",
        "cumulative_reward_episode",
        "queue_length",
        "queue_ma50",
        "loss",
    ]
    with open(RL_STEP_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in step_rows:
            writer.writerow(row)


def save_episode_csv(episode_rows: List[Dict[str, Any]]):
    if not episode_rows:
        return
    fieldnames = [
        "episode",
        "env_label",
        "batch_idx",
        "cumulative_reward",
        "avg_queue",
        "min_queue",
        "max_queue",
    ]
    with open(RL_EPISODE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in episode_rows:
            writer.writerow(row)


def plot_metrics(episode_rows: List[Dict[str, Any]]):
    if not episode_rows:
        return

    ep_to_rewards: Dict[int, List[float]] = {}
    ep_to_avg_queue: Dict[int, List[float]] = {}

    for row in episode_rows:
        ep = int(row["episode"])
        ep_to_rewards.setdefault(ep, []).append(float(row["cumulative_reward"]))
        ep_to_avg_queue.setdefault(ep, []).append(float(row["avg_queue"]))

    episodes_sorted = sorted(ep_to_rewards.keys())
    mean_rewards = [np.mean(ep_to_rewards[ep]) for ep in episodes_sorted]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_title(f"RL Training Metrics (run_id={RUN_ID})")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward", color="tab:blue")
    ax1.plot(episodes_sorted, mean_rewards, label="Mean Episode Reward", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    fig.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close(fig)
    print(f"[Plot] Saved training curves to {PLOT_PATH}")


# ================================================================
#  Vectorized SUMO Environment Management + Main Loop
# ================================================================

def run():
    global best_metric

    online_model, target_model, optimizer, replay_buffer = init_models_and_replay()

    global_step = 0
    replay_ratio_accum = REPLAY_RATIO_ACCUM_INIT

    step_rows: List[Dict[str, Any]] = []
    episode_rows: List[Dict[str, Any]] = []

    episode_counter = 0
    beta = PRIORITY_BETA_START

    try:
        while episode_counter < NUM_EPISODES:
            batch_size_envs = min(NUM_ENVS, NUM_EPISODES - episode_counter)
            print(f"[Batch] Starting {batch_size_envs} envs for episodes [{episode_counter}..{episode_counter + batch_size_envs - 1}]")

            conns = []
            for i in range(batch_size_envs):
                seed = GLOBAL_SEED + episode_counter + i
                label = f"env_{episode_counter}_{i}"
                traci.start(make_sumo_config(seed), label=label)
                conn = traci.getConnection(label)
                conns.append(conn)

            states = []
            prev_states = []
            cum_rewards = [0.0 for _ in range(batch_size_envs)]
            queues_per_ep: List[List[float]] = [[] for _ in range(batch_size_envs)]

            for env_idx in range(batch_size_envs):
                tc = conns[env_idx]
                s = get_state(tc)
                states.append(s)
                prev_states.append(None)

            for step_in_ep in range(STEPS_PER_EPISODE):
                for env_idx in range(batch_size_envs):
                    tc = conns[env_idx]
                    tc.simulationStep()

                    state = get_state(tc)
                    prev_state = prev_states[env_idx]
                    reward = get_reward(state, prev_state)
                    done = False  # fixed horizon

                    action = select_action(state, global_step, MODE, online_model)
                    apply_action_safe(action, tc)

                    replay_buffer.add(states[env_idx], action, reward, state, done)

                    cum_rewards[env_idx] += reward
                    queues_per_ep[env_idx].append(sum(state[:-1]))

                    prev_states[env_idx] = state
                    states[env_idx] = state

                    step_row = {
                        "global_step": global_step,
                        "batch_idx": episode_counter // NUM_ENVS,
                        "env_label": env_idx,
                        "episode": episode_counter + env_idx,
                        "step_in_episode": step_in_ep,
                        "reward": reward,
                        "cumulative_reward_episode": cum_rewards[env_idx],
                        "queue_length": sum(state[:-1]),
                        "queue_ma50": 0.0,  # filled later if you want moving avg
                        "loss": 0.0,
                    }
                    step_rows.append(step_row)

                    global_step += 1

                    replay_ratio_accum += REPLAY_RATIO / REPLAY_BASE_ENVS
                    updates = int(replay_ratio_accum)
                    replay_ratio_accum -= updates

                    for _ in range(updates):
                        loss_val = train_step(beta, online_model, target_model, optimizer, replay_buffer, global_step)
                        if loss_val is not None:
                            step_rows[-1]["loss"] = loss_val

                # anneal beta per environment step
                beta = min(
                    PRIORITY_BETA_END,
                    PRIORITY_BETA_START + (PRIORITY_BETA_END - PRIORITY_BETA_START) * (global_step / (NUM_EPISODES * STEPS_PER_EPISODE)),
                )

            # end of batch episodes: compute per-episode metrics
            for env_idx in range(batch_size_envs):
                ep_id = episode_counter + env_idx
                queues = queues_per_ep[env_idx]
                if len(queues) == 0:
                    avg_q = 0.0
                    min_q = 0.0
                    max_q = 0.0
                else:
                    avg_q = float(np.mean(queues))
                    min_q = float(np.min(queues))
                    max_q = float(np.max(queues))

                ep_row = {
                    "episode": ep_id,
                    "env_label": env_idx,
                    "batch_idx": episode_counter // NUM_ENVS,
                    "cumulative_reward": cum_rewards[env_idx],
                    "avg_queue": avg_q,
                    "min_queue": min_q,
                    "max_queue": max_q,
                }
                episode_rows.append(ep_row)

            # print and close SUMO connections at end of batch
            for i, conn in enumerate(conns):
                label = conn.getLabel()
                print(f"[Batch] Closing SUMO connection {i} (label={label})")
                conn.close()

            # checkpointing & logging at end of each batch
            replay_buffer.flush()
            replay_buffer.save(REPLAY_PATH)

            torch.save(
                {
                    "online": online_model.state_dict(),
                    "target": target_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                },
                LAST_MODEL_PATH,
            )

            # best model tracking by mean episode reward in this batch
            batch_rewards = [row["cumulative_reward"] for row in episode_rows if row["episode"] >= episode_counter]
            if batch_rewards:
                batch_mean_reward = float(np.mean(batch_rewards))
                if best_metric is None or batch_mean_reward > best_metric:
                    best_metric = batch_mean_reward
                    torch.save(
                        {
                            "online": online_model.state_dict(),
                            "target": target_model.state_dict(),
                            "optimizer": optimizer.state_dict(),
                        },
                        BEST_MODEL_PATH,
                    )
                    print(f"[Checkpoint] New best model with mean batch reward {best_metric:.3f}")

            # write CSVs and plot after each batch so they exist even if run stops early
            save_step_csv(step_rows)
            save_episode_csv(episode_rows)
            plot_metrics(episode_rows)

            episode_counter += batch_size_envs

    finally:
        # safety: close any remaining connections if something went wrong mid-batch
        try:
            all_labels = traci.getConnectionIDs()
        except Exception:
            all_labels = []
        for label in all_labels:
            try:
                print(f"[Finally] Closing leftover SUMO connection label={label}")
                traci.getConnection(label).close()
            except Exception:
                pass

        # ensure we at least persist what we have
        replay_buffer.flush()
        replay_buffer.save(REPLAY_PATH)
        save_step_csv(step_rows)
        save_episode_csv(episode_rows)
        plot_metrics(episode_rows)
        print("[Finally] Saved replay, CSVs, and plot with collected data.")


if __name__ == "__main__":
    run()
