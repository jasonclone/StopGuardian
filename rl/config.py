# rl/config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# Hyperparameters
# ================================================================


# Reward scaling
REWARD_SCALE = 150.0
REWARD_CLIP = 1.0

# Normalization constants
MAX_VEH_QUEUE = 200.0
MAX_PED_QUEUE = 100.0

GAMMA = 0.99
N_STEPS = 5
BUFFER_SIZE = 100000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 5000
TARGET_UPDATE_FREQ = 1000
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


CHECKPOINT_EVERY_EPISODES = 1

# Distributional C51
NUM_ATOMS = 51
V_MIN = -100.0
V_MAX = 100.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

# Will be set after inspecting SUMO logic
NUM_PHASES = None