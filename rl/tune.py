# rl/tune.py

import os
import sys
import time
import json

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import ray
from ray import tune, train
from ray.air import RunConfig
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search.optuna import OptunaSearch
from ray.tune.search import ConcurrencyLimiter
from ray.tune import with_resources

import torch
import torch_directml
import traci

from config import (
    set_hparams,
    ACTIONS,
    WARMUP_STEPS,
    USE_EPSILON,
    EPSILON_START,
    EPSILON_END,
    EPSILON_DECAY_STEPS,
    MIN_REPLAY_SIZE,
    NUM_ATOMS,
)

os.environ["TUNE_MODE"] = "1" # to signal tune workers to avoid importing TensorBoard in ray tune workers 


from action import apply_action_safe, select_action
from learner import train_step
from traffic_env import TrafficEnv, make_sumo_config
from train import RESULTS_DIR, RUN_DIR, RUN_ID, RUN_ID, CALIB_DIR,  init_models_and_replay   # now pure model init only


# get calibration data from b1_cal.py runs (if it exists) to apply tuned normalization and reward params to env before training
CALIB_DIR = os.path.join(RESULTS_DIR, "calib")
os.makedirs(CALIB_DIR, exist_ok=True)

CALIB_CSV = os.path.abspath(
    os.path.join(RESULTS_DIR, "calib", "b1_calibration.csv")
)


# ============================================================
# SUMO setup
# ============================================================
if "SUMO_HOME" not in os.environ:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

tools = os.path.join(os.environ["SUMO_HOME"], "tools")
sys.path.append(tools)


# ============================================================
# DirectML device
# ============================================================
DML_DEVICE = torch_directml.device()


# ============================================================
# Optuna Search + Concurrency Limiter
# ============================================================
optuna_search = ConcurrencyLimiter(
    OptunaSearch(metric="loss", mode="min"),
    max_concurrent=8,
)



# ============================================================
# Training Loop
# ============================================================
def train_wrapper(config):
    

    # each worker creates its own environment instance and applies calibration params if available, to avoid any cross-worker SUMO session issues and to ensure each worker benefits from calibration tuning if available
    env = TrafficEnv("C")



    # verify calibration source csv exists (run b1_cal.py), and if so apply auto-tuning to env params based on observed data
    if os.path.exists(CALIB_CSV):
        print("[INFO] Applying tuned normalization and reward params from calibration run...")
        env.calibrate_env_params(CALIB_CSV)
    else:
        print("[WARNING] Calibration CSV not found. Using default env normalization and reward params.")


    # avoid port collisions
    time.sleep(0.2)

    set_hparams(**config)

    MAX_EPISODES = 30
    STEPS_PER_EPISODE = 1000

    # ============================================================
    # 1. Determine state_size inside THIS worker's SUMO session
    # ============================================================
    from ray.train import get_context
    worker_id = str(get_context().get_trial_id())

    traci.start(make_sumo_config(), label=worker_id + "_init")
    traci.switch(worker_id + "_init")
    traci.simulationStep()

    env.initialize_from_sumo()
    dummy_state, _ = env.get_state()
    dummy_state_tensor = env.normalize_state_torch(dummy_state)
    state_size = dummy_state_tensor.shape[0]

    traci.close()

    # ============================================================
    # 2. Build models + replay buffer (NO SUMO HERE)
    # ============================================================
    online_model, target_model, replay_buffer, optimizer = init_models_and_replay(state_size)
    online_model.to(DML_DEVICE)
    target_model.to(DML_DEVICE)

    beta = config["PRIORITY_BETA_START"]
    beta_increment = (config["PRIORITY_BETA_END"] - config["PRIORITY_BETA_START"]) / (
        MAX_EPISODES * STEPS_PER_EPISODE
    )

    # ============================================================
    # 3. Episode Loop
    # ============================================================
    for ep in range(MAX_EPISODES):

        traci.start(make_sumo_config(), label=worker_id)
        traci.switch(worker_id)
        traci.simulationStep()

        env.initialize_from_sumo()
        env.reset_tracking()

        cumulative_reward = 0.0
        env_step_time_total = 0.0
        train_step_time_total = 0.0
        train_step_count = 0
        loss = 0

        try:
            for t in range(STEPS_PER_EPISODE):

                t_env_start = time.perf_counter()

                state_raw, _ = env.get_state()

                action = select_action(
                    env=env,
                    state_raw=state_raw,
                    global_step=t,
                    mode="train",
                    online_model=online_model,
                    device=DML_DEVICE,
                    actions=ACTIONS,
                    warmup_steps=WARMUP_STEPS,
                    use_epsilon=USE_EPSILON,
                    epsilon_start=EPSILON_START,
                    epsilon_end=EPSILON_END,
                    epsilon_decay_steps=EPSILON_DECAY_STEPS,
                    normalize_state_torch=env.normalize_state_torch,
                )

                apply_action_safe(action, env, current_step_global=t)

                traci.simulationStep()

                next_state_raw, _ = env.get_state()

                reward = env.get_reward(next_state_raw, state_raw, action)
                cumulative_reward += reward

                done = (t == STEPS_PER_EPISODE - 1)
                replay_buffer.add(state_raw, action, reward, next_state_raw, done)

                t_env_end = time.perf_counter()
                env_step_time_total += (t_env_end - t_env_start)

                # training
                t_train_start = time.perf_counter()

                loss = train_step(
                    beta=beta,
                    online_model=online_model,
                    target_model=target_model,
                    replay_buffer=replay_buffer,
                    optimizer=optimizer,
                    device=DML_DEVICE,
                    batch_size=config["BATCH_SIZE"],
                    min_replay_size=MIN_REPLAY_SIZE,
                    gamma=config["GAMMA"],
                    n_steps=config["N_STEPS"],
                    num_atoms=NUM_ATOMS,
                    v_min=config["V_MIN"],
                    v_max=config["V_MAX"],
                    delta_z=(config["V_MAX"] - config["V_MIN"]) / (NUM_ATOMS - 1),
                    normalize_state_torch=env.normalize_state_torch,
                )

                t_train_end = time.perf_counter()
                train_step_time_total += (t_train_end - t_train_start)
                train_step_count += 1

                beta = min(1.0, beta + beta_increment)

        finally:
            try:
                traci.close()
            except:
                pass

        avg_env_step_ms = (env_step_time_total / STEPS_PER_EPISODE) * 1000.0
        avg_train_step_ms = (train_step_time_total / max(train_step_count, 1)) * 1000.0

        train.report(
            {
                "loss": loss if loss is not None else float("inf"),
                "training_iteration": ep + 1,
                "avg_env_step_ms": avg_env_step_ms,
                "avg_train_step_ms": avg_train_step_ms,
            }
        )


# ============================================================
# Search Space
# ============================================================
search_space = {
    "LEARNING_RATE": tune.loguniform(1e-5, 2e-4),
    "GAMMA": tune.uniform(0.95, 0.99),
    "N_STEPS": tune.choice([3, 5, 7]),
    "BUFFER_SIZE": tune.lograndint(10000, 100000),
    "BATCH_SIZE": tune.choice([64, 128, 256]),
    "PRIORITY_ALPHA": tune.uniform(0.5, 0.7),
    "PRIORITY_BETA_START": tune.uniform(0.3, 0.5),
    "PRIORITY_BETA_END": tune.uniform(0.8, 0.95),
    "NOISY_SIGMA": tune.uniform(0.1, 0.3),
    "V_MIN": tune.uniform(-600.0, -400.0),
    "V_MAX": tune.uniform(400.0, 600.0),
}


# ============================================================
# Tuner
# ============================================================
def main():
    ray.init()

    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=30,
        grace_period=2,
        reduction_factor=2,
        time_attr="training_iteration",
    )

    tuner = tune.Tuner(
        with_resources(train_wrapper, resources={"cpu": 1}),
        tune_config=tune.TuneConfig(
            search_alg=optuna_search,
            scheduler=scheduler,
            num_samples=32,
        ),
        run_config=RunConfig(
            name="rainbow_directml_tuning",
            storage_path=os.path.abspath("results/tuning"),
        ),
        param_space=search_space,
    )

    results = tuner.fit()
    best = results.get_best_result(metric="loss", mode="min")

    print("\n=== BEST CONFIG FOUND ===")
    print(best.config)

    tuning_dir = os.path.abspath("results/tuning")
    os.makedirs(tuning_dir, exist_ok=True)

    with open(os.path.join(tuning_dir, "best_config.json"), "w") as f:
        json.dump(best.config, f, indent=2)

    with open(os.path.join(tuning_dir, "all_trials.jsonl"), "w") as f:
        for r in results:
            record = {"config": r.config, "metrics": r.metrics}
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()

