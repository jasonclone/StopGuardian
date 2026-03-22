import pickle
import os

# --- FIX: define a dummy ReplayState so pickle can resolve the name ---
class ReplayState:
    pass

REPLAY_PATH = "results/default/replay/replay.pkl"
index = 100

with open(REPLAY_PATH, "rb") as f:
    data = pickle.load(f)

print("Total transitions collected:", len(data.buffer))
print("Example transition from index", index, ":", data.buffer[index])

size_mb = os.path.getsize(REPLAY_PATH) / (1024 * 1024)
print("Replay file size (MB):", round(size_mb, 3))
