# rl/config.py
import torch_directml  # DirectML

DEVICE = torch_directml.device()

# ---------------- Default Hyperparameters ----------------

GAMMA = 0.99
N_STEPS = 5
BUFFER_SIZE = 20000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 1000
TARGET_UPDATE_FREQ = 1000
WARMUP_STEPS = 1000

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

CHECKPOINT_EVERY_EPISODES = 1

NUM_ATOMS = 51
V_MIN = -500.0
V_MAX = 0.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

# ---------------- Hyperparameter Override ----------------
def set_hparams(**kwargs):
    """
    Override any hyperparameter dynamically.
    Example:
        set_hparams(LEARNING_RATE=3e-4, GAMMA=0.97)
    """
    global_vars = globals()
    for key, value in kwargs.items():
        if key in global_vars:
            global_vars[key] = value
        else:
            raise KeyError(f"Unknown hyperparameter: {key}")
