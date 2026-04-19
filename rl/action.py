# rl/action.py

import random
import numpy as np
import torch
import traci


# ------------------------------------------------------------
# Safe action switching
# ------------------------------------------------------------

def apply_action_safe(action, env, current_step_global=None):

    program = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
    phase = traci.trafficlight.getPhase(env.tls_id)

    # 3. Switch if action == 1
    if action == 1:
        next_phase = (phase + 1) % len(program.phases)
        traci.trafficlight.setPhase(env.tls_id, next_phase)
        env.last_switch_step = current_step_global



# ------------------------------------------------------------
# Action selection raw state input
# ------------------------------------------------------------


# Warmup state (module-level globals)
_warmup_current_phase = None # track current traffic signal phase index to allow realistic random durations per phase
_warmup_steps_left_in_phase = 0 # countdown for how many steps to stay in current signal phase before allowing switch 

def select_action(
    env,
    state_raw,
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
    normalize_state_torch,
):
    global _warmup_current_phase, _warmup_steps_left_in_phase

    # -------------------------
    # 1. Convert to tensor (GPU)
    # -------------------------
    s_tensor = torch.tensor(state_raw, dtype=torch.float32, device=device).unsqueeze(0)

    # -------------------------
    # 2. Normalize ON GPU
    # -------------------------
    s_tensor = normalize_state_torch(s_tensor)

    # -------------------------
    # 3. Warmup (realistic random phase durations)
    # -------------------------
    if mode == "train" and global_step < warmup_steps:
        program = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
        current_phase = traci.trafficlight.getPhase(env.tls_id)

        # SUMO phase duration is in seconds → convert to 0.5s steps
        max_steps = int(program.phases[current_phase].duration * 2)

        # If entering a new phase OR countdown expired → sample new duration
        if _warmup_current_phase != current_phase or _warmup_steps_left_in_phase <= 0:
            _warmup_current_phase = current_phase
            _warmup_steps_left_in_phase = random.randint(1, max_steps) # sample random duration for current phase (at least 1 steps, at most max_steps)

        # Count down
        _warmup_steps_left_in_phase -= 1

        # KEEP until countdown ends
        if _warmup_steps_left_in_phase > 0:
            return 0  # KEEP

        # Then NEXT
        return 1

    # -------------------------
    # 4. Epsilon-greedy
    # -------------------------
    if mode == "train" and use_epsilon:
        eps = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * global_step / epsilon_decay_steps,
        )
        if random.random() < eps:
            return random.choice(actions)

    # -------------------------
    # 5. Forward pass (GPU)
    # -------------------------
    if mode == "eval":
        online_model.eval()
    else:
        online_model.train()

    with torch.no_grad():
        q_vals = online_model.q_values(s_tensor)[0]

    return int(torch.argmax(q_vals).item())