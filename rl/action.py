# rl/action.py
import os
import random
import torch
import traci
import logging


from config import (
    DEVICE,
    GAMMA,
    N_STEPS,
    BUFFER_SIZE,
    BATCH_SIZE,
    NOISY_SIGMA,
    MIN_REPLAY_SIZE,
    WARMUP_STEPS,
    USE_EPSILON,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    ACTIONS,
    NUM_ACTIONS,
    PRIORITY_ALPHA,
    PRIORITY_BETA_START,
    PRIORITY_BETA_END,
    TAU,
    LEARNING_RATE,
    CHECKPOINT_EVERY_EPISODES,
    NUM_ATOMS,
    V_MIN,
    V_MAX,
    DELTA_Z,
)


# -----------------------
# Logging configuration
# -----------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "stopguardian_enforcement.log")
logger = logging.getLogger("stopguardian_enforcement")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

# Warmup state
_warmup_current_phase = None
_warmup_steps_left_in_phase = 0

# Minimum green (in environment steps)
MIN_GREEN_STEPS = 10
MIN_GREEN_SECONDS = None  # optional: set seconds and compute steps from env.step_length


def _safe_get_phase_and_state(tls_id):
    """
    Robustly return (phase_index, phase_state_string).
    phase_state is the R/Y/G string (e.g., "GrGr" or "yryr").
    Returns (None, "") on failure.
    """
    try:
        phase = traci.trafficlight.getPhase(tls_id)
    except Exception as e:
        logger.exception("getPhase failed for tls_id=%s", tls_id)
        return None, ""
    try:
        phase_state = traci.trafficlight.getRedYellowGreenState(tls_id)
    except Exception as e:
        try:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            phase_state = program.phases[phase].state
        except Exception as e:
            logger.exception("Failed to obtain phase_state for tls_id=%s", tls_id)
            phase_state = ""
    return phase, phase_state


def apply_action_safe(action, env, current_step_global=None):
    """
    Apply action and log. IMPORTANT: set env.phase_start_step to the
    global_step value that the NEXT select_action() will receive.
    Given loop ordering, that is current_step_global + 1.
    """
    try:
        program = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
        phase = traci.trafficlight.getPhase(env.tls_id)
    except Exception as e:
        logger.exception("apply_action_safe: failed to query trafficlight for tls_id=%s", getattr(env, "tls_id", "UNKNOWN"))
        return

    if action == 1:
        next_phase = (phase + 1) % len(program.phases)
        try:
            traci.trafficlight.setPhase(env.tls_id, next_phase)
            logger.info("apply_action_safe: SWITCH applied tls=%s from phase=%s to next=%s at step=%s",
                        env.tls_id, phase, next_phase, current_step_global)
            try:
                env.last_switch_step = current_step_global
            except Exception as e:
                env.last_switch_step = None
            try:
                env.phase_start_step = (current_step_global + 1) if current_step_global is not None else None
                env._last_phase_seen = next_phase
            except Exception as e:
                pass
        except Exception as e:
            logger.exception("apply_action_safe: failed to setPhase for tls_id=%s to %s", env.tls_id, next_phase)


def select_action(
    env,
    state_raw,
    global_step,
    mode,
    online_model,
    normalize_state_torch,
    device = DEVICE,
    actions = ACTIONS,
    warmup_steps = WARMUP_STEPS,
    use_epsilon = False,
    epsilon_start = EPSILON_START,
    epsilon_end = EPSILON_END,
    epsilon_decay_steps = EPSILON_DECAY_STEPS,
):
    """
    Decide action with enforcement:
      - Run enforcement checks BEFORE any model forward pass so the agent's model
        is not invoked when switching is impossible.
      - Warmup behavior remains allowed to attempt switches, but warmup choices
        are still subject to the same enforcement (they can be overridden).
      - Only prepare tensors / call the model when enforcement allows non-warmup decisions.
    """
    global _warmup_current_phase, _warmup_steps_left_in_phase

    # Get phase and state robustly (cheap, used for pre-enforcement)
    current_phase, phase_state = _safe_get_phase_and_state(env.tls_id)

    # Compute min_green_steps from seconds if configured
    min_green_steps = MIN_GREEN_STEPS
    if MIN_GREEN_SECONDS is not None:
        try:
            step_length = float(getattr(env, "step_length", 0.5))
            min_green_steps = max(1, int(round(MIN_GREEN_SECONDS / step_length)))
        except Exception as e:
            logger.exception("select_action: failed to compute min_green_steps from seconds; falling back to MIN_GREEN_STEPS")

    # Initialize phase tracking only if both are missing
    if getattr(env, "phase_start_step", None) is None and getattr(env, "_last_phase_seen", None) is None:
        if current_phase is not None:
            env.phase_start_step = global_step
            env._last_phase_seen = current_phase
            logger.debug("select_action: initialized phase_start_step=%s _last_phase_seen=%s (global_step=%s)",
                         env.phase_start_step, env._last_phase_seen, global_step)

    # If phase changed according to SUMO, update _last_phase_seen but do NOT overwrite phase_start_step
    if current_phase is not None and env._last_phase_seen != current_phase:
        logger.debug("select_action: observed phase change from %s to %s at global_step=%s (phase_start=%s)",
                     env._last_phase_seen, current_phase, global_step, env.phase_start_step)
        env._last_phase_seen = current_phase

    # Pre-enforcement: block non-warmup agent computation early
    try:
        in_warmup = (mode == "train" and global_step < warmup_steps)

        # If NOT in warmup, enforce restrictions immediately and return KEEP (0)
        if not in_warmup:
            if phase_state and ("y" in phase_state or "Y" in phase_state):
                logger.debug("select_action: pre-enforcement blocked (YELLOW) at step=%s tls=%s", global_step, env.tls_id)
                return 0
            if phase_state and ("G" in phase_state or "g" in phase_state):
                if env.phase_start_step is not None:
                    steps_since_green = global_step - env.phase_start_step
                    if steps_since_green < min_green_steps:
                        logger.debug("select_action: pre-enforcement blocked (MIN_GREEN since=%s < %s) at step=%s tls=%s",
                                     steps_since_green, min_green_steps, global_step, env.tls_id)
                        return 0
                else:
                    logger.debug("select_action: pre-enforcement blocked (NO_PHASE_START_STEP) at step=%s tls=%s", global_step, env.tls_id)
                    return 0
            if phase_state == "":
                logger.debug("select_action: pre-enforcement blocked (UNKNOWN_PHASE_STATE) at step=%s tls=%s", global_step, env.tls_id)
                return 0
    except Exception as e:
        logger.exception("select_action: exception during pre-enforcement; defaulting to KEEP")
        return 0

    # Decide requested action
    requested_action = 0
    try:
        # Warmup: keep existing warmup behavior (allowed to attempt switches),
        # but we will still apply enforcement after warmup choice.
        if mode == "train" and global_step < warmup_steps:
            try:
                program = traci.trafficlight.getAllProgramLogics(env.tls_id)[0]
                cur_phase = traci.trafficlight.getPhase(env.tls_id)
                max_steps = int(program.phases[cur_phase].duration * 2)
            except Exception as e:
                max_steps = 1
            if _warmup_current_phase != current_phase or _warmup_steps_left_in_phase <= 0:
                _warmup_current_phase = current_phase
                _warmup_steps_left_in_phase = random.randint(1, max(1, max_steps))
            _warmup_steps_left_in_phase -= 1
            requested_action = 0 if _warmup_steps_left_in_phase > 0 else 1
        else:
            # Non-warmup: model / epsilon decision (we only reach here if pre-enforcement allowed it)
            try:
                s_tensor = torch.tensor(state_raw, dtype=torch.float32, device=device).unsqueeze(0)
                s_tensor = normalize_state_torch(s_tensor)
            except Exception as e:
                logger.exception("select_action: failed to prepare state tensor")
                return 0

            if mode == "train" and use_epsilon:
                eps = max(
                    epsilon_end,
                    epsilon_start - (epsilon_start - epsilon_end) * global_step / epsilon_decay_steps,
                )
                if random.random() < eps:
                    requested_action = random.choice(actions)
                else:
                    with torch.no_grad():
                        q_vals = online_model.q_values(s_tensor)[0]
                        requested_action = int(torch.argmax(q_vals).item())
            else:
                with torch.no_grad():
                    q_vals = online_model.q_values(s_tensor)[0]
                    requested_action = int(torch.argmax(q_vals).item())
    except Exception as e:
        logger.exception("select_action: error computing requested_action; defaulting to KEEP")
        requested_action = 0

    # Post-enforcement: ensure warmup attempts are still subject to restrictions
    blocked_reason = None
    final_action = requested_action
    try:
        # If phase is yellow -> block
        if phase_state and ("y" in phase_state or "Y" in phase_state):
            blocked_reason = "YELLOW_PHASE"
            final_action = 0
        elif phase_state and ("G" in phase_state or "g" in phase_state):
            if env.phase_start_step is not None:
                steps_since_green = global_step - env.phase_start_step
                if steps_since_green < min_green_steps:
                    blocked_reason = f"MIN_GREEN (since={steps_since_green} < {min_green_steps})"
                    final_action = 0
                else:
                    final_action = requested_action
            else:
                blocked_reason = "NO_PHASE_START_STEP"
                final_action = 0
        else:
            if phase_state == "":
                blocked_reason = "UNKNOWN_PHASE_STATE"
                final_action = 0
            else:
                final_action = requested_action
    except Exception as e:
        logger.exception("select_action: exception during post-enforcement; defaulting to KEEP")
        final_action = 0
        blocked_reason = "ENFORCEMENT_EXCEPTION"

    # Log the decision
    try:
        logger.info(
            "DECISION step=%s tls=%s cur_phase=%s phase_state=%s phase_start=%s requested=%s final=%s reason=%s",
            global_step, env.tls_id, current_phase, phase_state, env.phase_start_step, requested_action, final_action, blocked_reason
        )
    except Exception as e:
        pass

    return final_action

