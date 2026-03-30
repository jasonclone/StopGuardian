# rl/config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================================================
# Hyperparameters
# ================================================================


# Reward 
REWARD_ABS_PENALTY_WEIGHT = 0.001
PED_REWARD_WEIGHT = 1.0
VEH_REWARD_WEIGHT = 1.0
REWARD_SCALE = 100.0

# Normalization constants
MAX_VEH_QUEUE = 40.0 # per lane, based on SUMO logic and observed data; will be used for state normalization and reward shaping
MAX_PED_QUEUE = 30.0 # all pedestrian lanes combined, based on SUMO logic and observed data; will be used for state normalization and reward shaping

GAMMA = 0.99
N_STEPS = 5
BUFFER_SIZE = 20000
BATCH_SIZE = 128
MIN_REPLAY_SIZE = 2000
TARGET_UPDATE_FREQ = 1000
WARMUP_STEPS = 2000 # steps globally where agent takes random actions before using model

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
V_MAX = 0.0
DELTA_Z = (V_MAX - V_MIN) / (NUM_ATOMS - 1)

# Will be set after inspecting SUMO logic
NUM_PHASES = None