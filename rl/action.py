# rl/action.py

import random
import numpy as np
import torch
import traci


# ------------------------------------------------------------
# Safe action switching
# ------------------------------------------------------------
def apply_action_safe(action, tls_id: str = "C"):
    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
    phase = traci.trafficlight.getPhase(tls_id)
    state = program.phases[phase].state

    # Block switching during yellow or all-red
    if "y" in state or ("G" not in state and "g" not in state):
        return

    # go to next phase
    if action == 1:
        next_phase = (phase + 1) % len(program.phases)
        traci.trafficlight.setPhase(tls_id, next_phase)


# ------------------------------------------------------------
# Action selection (normalization REQUIRED)
# ------------------------------------------------------------
def select_action(
    state,
    step,
    mode,
    online_model,
    device,
    actions,
    warmup_steps,
    use_epsilon,
    epsilon_start,
    epsilon_end,
    epsilon_decay_steps,
    normalize_state,   # <-- REQUIRED
):
    # Normalize using env.normalize_state
    s_norm = normalize_state(state).reshape(1, -1)
    s_tensor = torch.from_numpy(s_norm).float().to(device)

    # Warmup
    if mode == "train" and step < warmup_steps:
        return random.choice(actions)

    # Epsilon-greedy
    if mode == "train" and use_epsilon:
        eps = max(
            epsilon_end,
            epsilon_start - (epsilon_start - epsilon_end) * step / epsilon_decay_steps,
        )
        if random.random() < eps:
            return random.choice(actions)

    # Greedy action
    online_model.eval()
    with torch.no_grad():
        q_vals = online_model.q_values(s_tensor)[0].cpu().numpy()

    return int(np.argmax(q_vals))
