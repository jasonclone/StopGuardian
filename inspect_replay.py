import pickle
import os
import sys
import types

# Create a fake 'replay' module so pickle stops crashing
replay = types.ModuleType("replay")
sys.modules["replay"] = replay

# Dummy classes that pickle expects
class ReplayBuffer:
    pass

class ReplayState:
    pass

# Attach them to the fake module
replay.ReplayBuffer = ReplayBuffer
replay.ReplayState = ReplayState

REPLAY_PATH = "results/rl/replay/replay.pkl"
index = 4999

with open(REPLAY_PATH, "rb") as f:
    data = pickle.load(f)

print("Total transitions collected:", len(data.buffer))
for i in range(len(data.buffer)):
    print("Example transition from index", i, ":", data.buffer[i])

size_mb = os.path.getsize(REPLAY_PATH) / (1024 * 1024)
print("Replay file size (MB):", round(size_mb, 3))