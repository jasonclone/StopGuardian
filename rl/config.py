# rl/config.py
import torch_directml  # DirectML

DEVICE = torch_directml.device()

# ---------------- Default Hyperparameters ----------------

GAMMA = 0.96
N_STEPS = 5
BUFFER_SIZE = 200000
BATCH_SIZE = 64
MIN_REPLAY_SIZE = 1000
WARMUP_STEPS = MIN_REPLAY_SIZE

USE_EPSILON = False
EPSILON_START = 0.3
EPSILON_END = 0.01
EPSILON_DECAY_STEPS = 200000

ACTIONS = [0, 1]
NUM_ACTIONS = len(ACTIONS)

PRIORITY_ALPHA = 0.5
PRIORITY_BETA_START = 0.4
PRIORITY_BETA_END =  0.95

NOISY_SIGMA = 0.5

LEARNING_RATE =  2.25e-05

# Linear LR decay steps (used by Rainbow)
LR_DECAY_STEPS = 200000

CHECKPOINT_EVERY_EPISODES = 1

NUM_ATOMS = 51
V_MIN = -80.0
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