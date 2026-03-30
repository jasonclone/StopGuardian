# rl/action.py

import random
import numpy as np
import torch
import traci


# ------------------------------------------------------------
# Safe action switching
# ------------------------------------------------------------
# def apply_action_safe(action, tls_id: str = "C"):
#     program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
#     phase = traci.trafficlight.getPhase(tls_id)
#     state = program.phases[phase].state

#     # Block switching during yellow or all-red. forces keep action on non green phase even if model selects switch.
#     if "y" in state or ("G" not in state and "g" not in state):
#         return

#     # go to next phase
#     if action == 1:
#         next_phase = (phase + 1) % len(program.phases)
#         traci.trafficlight.setPhase(tls_id, next_phase)
    
#     # otherwise action == 0, keep current phase

# ------------------------------------------------------------
# Realistic Safe Action Switching
# ------------------------------------------------------------

MIN_GREEN_STEPS = 20   # step length is 0.5 so this corresponds to 10 seconds minimum green time, which is reasonable for safety

def apply_action_safe(action, env, current_step_global=None):
    
    program = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
    phase = traci.trafficlight.getPhase(env.tls_id)
    state = program.phases[phase].state

    # 1. Block switching during yellow or all-red. forces keep action on non green phase even if model selects switch.
    if "y" in state or ("G" not in state and "g" not in state):
        return

    # 2. Enforce minimum green time. even if model selects switch, it will be blocked until minimum green time is satisfied.
    if current_step_global is not None and env.last_switch_step is not None:
        if current_step_global - env.last_switch_step < MIN_GREEN_STEPS:
            return

    # 3. If action == 1, switch to next phase
    if action == 1:
        next_phase = (phase + 1) % len(program.phases)
        traci.trafficlight.setPhase(env.tls_id, next_phase)
        env.last_switch_step = current_step_global


# ------------------------------------------------------------
# Action selection (normalization REQUIRED)
# ------------------------------------------------------------
def select_action(
    state,
    global_step,
    mode,
    online_model,
    device,
    actions,
    warmup_steps,
    use_epsilon,
    epsilon_start,
    epsilon_end,
    epsilon_decay_steps,
):
    # Convert state to tensor directly (NO normalization here)
    s_tensor = torch.from_numpy(np.array(state, dtype=np.float32).reshape(1, -1)).to(device)

    # Warmup
    if mode == "train" and global_step < warmup_steps:
        return random.choice(actions)

    # Epsilon-greedy
    if mode == "train" and use_epsilon:
        eps = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * global_step / epsilon_decay_steps,
        )
        if random.random() < eps:
            return random.choice(actions)

    # Greedy action
    online_model.eval()
    with torch.no_grad():
        q_vals = online_model.q_values(s_tensor)[0].cpu().numpy()

    return int(np.argmax(q_vals))