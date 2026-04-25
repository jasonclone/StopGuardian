# rl/replay.py
import numpy as np
import pickle
import os
from dataclasses import dataclass
from typing import Optional

from config import (
    DEVICE,
    GAMMA,
    N_STEPS,
    BUFFER_SIZE,
    BATCH_SIZE,
    NOISY_SIGMA,
    MIN_REPLAY_SIZE,
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
    TAU,
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)

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
    def __init__(self, capacity = BUFFER_SIZE, alpha = PRIORITY_ALPHA, n_steps = N_STEPS, gamma = GAMMA):
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

        max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
        if max_prio == 0:
            max_prio = 1.0
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

            max_prio = self.priorities.max() if len(self.buffer) > 0 else 1.0
            if max_prio == 0:
                max_prio = 1.0
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity
            self.n_step_buffer.pop(0)

    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
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
        # avoid division by zero
        if weights.max() == 0:
            weights = np.ones_like(weights)
        else:
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
            # ensure index is within current buffer length
            if 0 <= i < len(self.priorities):
                self.priorities[i] = self._priority(float(td))

    def save(self, path: str):
        """
        Save buffer state to a file path (pickle).
        """
        state = ReplayState(
            buffer=self.buffer,
            pos=self.pos,
            priorities=self.priorities,
            n_step_buffer=self.n_step_buffer,
        )
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        """
        Load buffer state from a file path (pickle).
        """
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state: ReplayState = pickle.load(f)
        self.buffer = state.buffer
        self.pos = state.pos
        self.priorities = state.priorities
        self.n_step_buffer = state.n_step_buffer

    # New helper methods for checkpoint compatibility
    def save_to_bytes(self) -> bytes:
        """
        Return a bytes representation of the replay buffer state suitable for embedding
        into a checkpoint file.
        """
        state = ReplayState(
            buffer=self.buffer,
            pos=self.pos,
            priorities=self.priorities,
            n_step_buffer=self.n_step_buffer,
        )
        return pickle.dumps(state)

    def load_from_bytes(self, b: bytes):
        """
        Restore replay buffer state from bytes produced by save_to_bytes.
        """
        if not b:
            return
        state: ReplayState = pickle.loads(b)
        self.buffer = state.buffer
        self.pos = state.pos
        self.priorities = state.priorities
        self.n_step_buffer = state.n_step_buffer


# ================================================================
# Uniform Replay Buffer (non prioritized and single step returns used in standard DQN)
# ================================================================

class UniformReplayBuffer:
    def __init__(self, capacity = BUFFER_SIZE):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def add(self, s, a, r, s_next, done):
        transition = (s, a, r, s_next, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.buffer) == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in idxs]
        states, actions, rewards, next_states, dones = zip(*samples)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def flush(self):
        # single-step DQN: nothing to flush
        pass

    def save(self, path: str):
        state = {
            "buffer": self.buffer,
            "pos": self.pos,
        }
        with open(path, "wb") as f:
            pickle.dump(state, f)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.buffer = state["buffer"]
        self.pos = state["pos"]

    # New helper methods for checkpoint compatibility
    def save_to_bytes(self) -> bytes:
        """
        Return a bytes representation of the uniform buffer state suitable for embedding
        into a checkpoint file.
        """
        state = {
            "buffer": self.buffer,
            "pos": self.pos,
        }
        return pickle.dumps(state)

    def load_from_bytes(self, b: bytes):
        """
        Restore uniform buffer state from bytes produced by save_to_bytes.
        """
        if not b:
            return
        state = pickle.loads(b)
        self.buffer = state.get("buffer", [])
        self.pos = state.get("pos", 0)
