# Step 1: Add modules to provide access to specific libraries and functions
import os  # Module provides functions to handle file paths, directories, environment variables
import sys  # Module provides access to Python-specific system parameters and functions
import random
import numpy as np
import matplotlib.pyplot as plt  # Visualization

# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
import keras
from keras import layers
import csv
import time

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RL_STEP_CSV = os.path.join(RESULTS_DIR, "rl_metrics.csv")
RL_SUMMARY_CSV = os.path.join(RESULTS_DIR, "rl_summary.csv")


# Step 2: Establish path to SUMO (SUMO_HOME)
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci  # Static network information (such as reading and analyzing network files)

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',
    '-c', 'simulation/sumo/test.sumocfg',
    '--step-length', '0.10',
    '--delay', '1000',
    '--lateral-resolution', '0'
    ,'--seed', '12345'
]

# Step 5: Open connection between SUMO and Traci
traci.start(Sumo_config)
#traci.gui.setSchema("View #0", "real world")


# Step 6: Define Variables


current_phase = 0

#  Reinforcement Learning Hyperparameters 
TOTAL_STEPS = 10000    # The total number of simulation steps for continuous (online) training.

ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 0.1          # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)



# Step 7: Define Functions
 

def build_model(state_size, action_size):
    """
    Build a simple feedforward neural network that approximates Q-values.
    """
    model = keras.Sequential()                                 # Feedforward neural network
    model.add(layers.Input(shape=(state_size,)))               # Input layer
    model.add(layers.Dense(24, activation='relu'))             # First hidden layer
    model.add(layers.Dense(24, activation='relu'))             # Second hidden layer
    model.add(layers.Dense(action_size, activation='linear'))  # Output layer
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=0.001)
    )
    return model

def get_queue_length(detector_id): #8.Constraint 8
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_current_phase(tls_id): #8.Constraint 8
    return traci.trafficlight.getPhase(tls_id)

def get_state():
    # Aggregate queues per incoming whole road approach (N, E, S, W) by summing the 4 detectors (4 lanes) for each direction
    q_N = sum([
        get_queue_length("C_NC_1"),
        get_queue_length("C_NC_2"),
        get_queue_length("C_NC_3"),
        get_queue_length("C_NC_4")
    ])

    q_E = sum([
        get_queue_length("C_EC_1"),
        get_queue_length("C_EC_2"),
        get_queue_length("C_EC_3"),
        get_queue_length("C_EC_4")
    ])

    q_S = sum([
        get_queue_length("C_SC_1"),
        get_queue_length("C_SC_2"),
        get_queue_length("C_SC_3"),
        get_queue_length("C_SC_4")
    ])

    q_W = sum([
        get_queue_length("C_WC_1"),
        get_queue_length("C_WC_2"),
        get_queue_length("C_WC_3"),
        get_queue_length("C_WC_4")
    ])

    phase = get_current_phase("C")

    return (q_N, q_E, q_S, q_W, phase)

def to_array(state_tuple): 
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))



#* Create the DQN model
dummy_state = get_state() # make dummy state to get state size for building the model, actual values don't matter since we're only using it to determine input shape
state_size = len(dummy_state)
action_size = len(ACTIONS)
dqn_model = build_model(state_size, action_size)



def get_max_Q_value_of_state(s): #1. Objective Function
    state_array = to_array(s)
    Q_values = dqn_model.predict(state_array, verbose=0)[0]  # shape: (action_size,)
    return np.max(Q_values)

def get_reward(state): #2. Constraint 2
    """
    Simple reward function:
    Negative of total queue length to encourage shorter queues.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward


    
def apply_action(action, tls_id="C"):
    
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    current_phase_id = get_current_phase(tls_id)
    current_phase_state = program.phases[current_phase_id].state

    # Determine if minimum time has passed for the CURRENT phase. keep this if-elif-else order
    if 'y' in current_phase_state:
        phase_type = "yellow"
    elif 'G' in current_phase_state or 'g' in current_phase_state:
        phase_type = "green"
    else:
        phase_type = "red"

    
    # YELLOW and RED cannot be kept
    
    if phase_type in ["yellow", "red"]:
        return  # RL cannot act in yellow or red

    
    # GREEN phase — RL decides
    
    if phase_type == "green":
        if action == 0:
            return  # KEEP green

        if action == 1:
            num_phases = len(program.phases)
            next_phase = (current_phase_id + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)

            new_phase_state = program.phases[next_phase].state






def update_Q_table(old_state, action, reward, new_state): #6. Constraint 6
    """
    In DQN, we do a single-step gradient update instead of a table update.
    """
    # 1) Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model.predict(old_state_array, verbose=0)[0]
    # 2) Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)
        
    # 3) Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * (reward + GAMMA * best_future_q - Q_values_old[action])
    
    # 4) Train (fit) the DQN on this single sample
    dqn_model.fit(old_state_array, np.array([Q_values_old]), verbose=0)

def get_action_from_policy(state): #7. Constraint 7
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))



# Step 8: Fully Online Continuous Learning Loop


# Lists to record data for plotting and metrics and summary csvs
step_history = []
reward_history = []
queue_history = []
tls_phase_history = []


# For spawn logging to check if seeds match between RL and baseline, we need to record spawned vehicles and pedestrians at each step
spawned_vehicles_history = []
spawned_peds_history = []

cumulative_reward = 0.0

print("\n=== Starting Fully Online Continuous Learning (DQN) ===")
for step in range(TOTAL_STEPS):
    current_simulation_step = step  # sets current step for use in apply_action's timing logic
    
    state = get_state()
    action = get_action_from_policy(state)
    apply_action(action)
    
    #* for spawn logging to check if seeds match between RL and baseline, we need to step the simulation before checking the new state and reward
    prev_v = set(traci.vehicle.getIDList())
    prev_p = set(traci.person.getIDList())
    
    traci.simulationStep()  # Advance simulation by one step
    
    #* for spawn logging to check if seeds match between RL and baseline, we need to step the simulation before checking the new state and reward
    cur_v = set(traci.vehicle.getIDList())
    cur_p = set(traci.person.getIDList())

    spawned_vehicles = list(cur_v - prev_v)
    spawned_peds = list(cur_p - prev_p)
    spawned_vehicles_history.append(spawned_vehicles)
    spawned_peds_history.append(spawned_peds)

    
    # this new state and reward will be used for the Q-value update
    new_state = get_state()
    reward = get_reward(new_state)
    cumulative_reward += reward
    
    # update Q-table (DQN weights) based on the transition (state, action, reward, new_state)
    update_Q_table(state, action, reward, new_state)
    # After updating the Q-table, we can get the updated Q-values for the current state for logging purposes
    updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]


    # save every step for plotting and summary stats
    step_history.append(step)
    reward_history.append(cumulative_reward)
    queue_history.append(sum(new_state[:-1]))
    tls_phase_history.append(new_state[-1])

    
    if step % 100 == 0:
        print(f"Step {step}, Current_State: {state}, Action: {action}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}, Q-values(current_state): {updated_q_vals}")


# Step 9: Close connection between SUMO and Traci

traci.close()

# Build rl_metrics.csv rows
rl_rows = []
for i in range(len(step_history)):
    rl_rows.append({
        "step": step_history[i],
        "cumulative_reward": reward_history[i],
        "queue_length": queue_history[i],
        "tls_phase": tls_phase_history[i],
        "spawned_vehicles": ";".join(spawned_vehicles_history[i]),
        "spawned_peds": ";".join(spawned_peds_history[i])
    })

# Write rl_metrics.csv
with open(RL_STEP_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["step","cumulative_reward","queue_length","tls_phase","spawned_vehicles","spawned_peds"])
    writer.writeheader()
    writer.writerows(rl_rows)

# Write rl_summary.csv
with open(RL_SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "final_cumulative_reward",
        "min_queue_length",
        "max_queue_length",
        "avg_queue_length"
    ])
    writer.writeheader()
    writer.writerow({
        "final_cumulative_reward": cumulative_reward,
        "min_queue_length": min(queue_history),
        "max_queue_length": max(queue_history),
        "avg_queue_length": sum(queue_history) / len(queue_history)
    })



# ~~~ Print final model summary (replacing Q-table info) ~~~
print("\nOnline Training completed.")
print("DQN Model Summary:")
dqn_model.summary()


# Visualization of Results


# Plot Cumulative Reward over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, reward_history, marker='o', linestyle='-', label="Cumulative Reward")
plt.xlabel("Simulation Step")
plt.ylabel("Cumulative Reward")
plt.title("RL Training (DQN): Cumulative Reward over Steps")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "rl_cumulative_reward.png"))

# Plot Total Queue Length over Simulation Steps
plt.figure(figsize=(10, 6))
plt.plot(step_history, queue_history, marker='o', linestyle='-', label="Total Queue Length")
plt.xlabel("Simulation Step")
plt.ylabel("Total Queue Length")
plt.title("RL Training (DQN): Queue Length over Steps")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "rl_queue_length.png"))