#test_env.py
from env.traffic_env import TrafficEnv
import traci

env = TrafficEnv()
state = env.reset()

# Add this line right after reset()
print("All lanes:", traci.lane.getIDList())

print("Initial state:", state)
print("State length:", len(state))
print("Num actions:", env.num_actions)

for i in range(5):
    s, r, d, _ = env.step(0)
    print(f"Step {i}: reward={r}, done={d}")

env.close()
