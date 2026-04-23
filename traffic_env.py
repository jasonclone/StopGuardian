# traffic_env.py

import os
import sys
import numpy as np
import csv
import matplotlib.pyplot as plt
import torch
import traci
import logging

# -----------------------
# Logging configuration
# -----------------------
LOG_PATH = os.path.join(os.path.dirname(__file__), "stopguardian_exceptions.log")
logger = logging.getLogger("stopguardian")
if not logger.handlers:
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    fh.setFormatter(fmt)
    logger.addHandler(fh)


class TrafficEnv:
    """
    Traffic environment wrapper around TraCI with stable phase bookkeeping.
    The training loop should:
      - call env.step() (which calls traci.simulationStep())
      - immediately call env.record_phase_start_if_changed(next_global_step)
        where next_global_step is the integer global step that will be passed
        to the next select_action() call (typically total_steps_global + 1).
    """

    def __init__(self, tls_id="C", step_length=0.5):
        self.tls_id = tls_id
        self.prev_veh_ids = set()
        self.prev_ped_ids = set()

        # bookkeeping
        self.last_switch_step = None

        # normalization constants
        self.MAX_VEH_QUEUE = 100
        self.MAX_PED_QUEUE = 20
        self.MAX_VEH_WAIT = 5500.0
        self.MAX_PED_WAIT = 1500.0
        self.MAX_VEH_THRU = 3
        self.MAX_PED_THRU = 2

        # runtime metrics
        self.veh_thru = 0
        self.ped_thru = 0
        self.veh_wait = 0.0
        self.ped_wait = 0.0

        self.prev_veh_wait = 0.0
        self.prev_ped_wait = 0.0
        self.prev_veh_thru = 0
        self.prev_ped_thru = 0

        # phase tracking for min-green enforcement
        # phase_start_step is the global_step value that the next select_action() will receive
        self.phase_start_step = None
        self._last_phase_seen = None

        # simulation step length (seconds) — used if you want time-based min-green
        self.step_length = float(step_length)

    # -------------------------
    # SUMO initialization
    # -------------------------
    def initialize_from_sumo(self):
        """
        Call after traci.start() and one traci.simulationStep() so lane metadata exists.
        """
        try:
            lanes = []
            for lane in traci.lane.getIDList():
                edge = traci.lane.getEdgeID(lane)
                # keep only incoming edges that end at intersection id (e.g., "...C")
                if edge.endswith(self.tls_id):
                    if not lane.startswith(":") and not lane.endswith("_0"):
                        lanes.append(lane)
            self.veh_lane_order = sorted(lanes)
            logger.info(f"[INIT] Vehicle Lane count: {len(self.veh_lane_order)}")
        except Exception as e:
            logger.exception("Failed to initialize lane order from SUMO in initialize_from_sumo()")
            raise RuntimeError("Failed to initialize lane order from SUMO. Ensure traci is connected and simulation has started.") from e

    def reset_tracking(self):
        try:
            self.prev_veh_ids = set()
            self.prev_ped_ids = set()
            # reset phase tracking so select_action can initialize phase_start_step on next call
            self.phase_start_step = None
            self._last_phase_seen = None
        except Exception:
            logger.exception("Exception in reset_tracking()")

    # -------------------------
    # Convenience wrappers
    # -------------------------
    def start(self, sumo_config):
        try:
            traci.start(sumo_config)
        except Exception:
            logger.exception("Exception starting SUMO with config: %s", sumo_config)
            raise

    def close(self):
        try:
            traci.close()
        except Exception:
            logger.exception("Exception closing TraCI connection")

    def step(self):
        try:
            traci.simulationStep()
        except Exception:
            logger.exception("Exception during traci.simulationStep()")

    def get_program(self):
        try:
            return traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
        except Exception:
            logger.exception("Exception in get_program() for tls_id=%s", self.tls_id)
            raise

    def get_phase(self):
        try:
            return traci.trafficlight.getPhase(self.tls_id)
        except Exception:
            logger.exception("Exception in get_phase() for tls_id=%s", self.tls_id)
            raise

    def set_phase(self, phase):
        try:
            traci.trafficlight.setPhase(self.tls_id, phase)
        except Exception:
            logger.exception("Exception in set_phase(%s) for tls_id=%s", phase, self.tls_id)

    def get_phase_state(self):
        """
        Return the R/Y/G string for the current phase. Uses program.phases fallback.
        """
        try:
            return traci.trafficlight.getRedYellowGreenState(self.tls_id)
        except Exception:
            try:
                program = self.get_program()
                current_phase = self.get_phase()
                return program.phases[current_phase].state
            except Exception:
                logger.exception("Failed to obtain phase state for tls_id=%s", self.tls_id)
                return ""

    def in_green_phase(self):
        try:
            return ("G" in self.get_phase_state())
        except Exception:
            logger.exception("Exception in in_green_phase()")
            return False

    def get_num_phases(self):
        try:
            program = self.get_program()
            return len(program.phases)
        except Exception:
            logger.exception("Exception in get_num_phases()")
            raise
        
    def compute_state_size(self):
        try:
            dummy_state, _ = self.get_state()
            dummy_tensor = self.normalize_state_torch(dummy_state)
            self.state_size = dummy_tensor.shape[0]
            return self.state_size
        except Exception:
            logger.exception("Exception in compute_state_size()")
            raise
    # -------------------------
    # Phase bookkeeping helper (call from main loop)
    # -------------------------
    def record_phase_start_if_changed(self, next_global_step):
        """
        Call this immediately after traci.simulationStep() with the integer
        next_global_step that will be passed to the next select_action() call.
        If SUMO reports a different phase than env._last_phase_seen, update
        env._last_phase_seen and set env.phase_start_step = next_global_step.
        """
        try:
            post_phase = traci.trafficlight.getPhase(self.tls_id)
            try:
                post_state = traci.trafficlight.getRedYellowGreenState(self.tls_id)
            except Exception:
                try:
                    program = traci.trafficlight.getAllProgramLogics(self.tls_id)[0]
                    post_state = program.phases[post_phase].state
                except Exception:
                    post_state = ""
            if getattr(self, "_last_phase_seen", None) is None or self._last_phase_seen != post_phase:
                prev = self._last_phase_seen
                self._last_phase_seen = post_phase
                self.phase_start_step = int(next_global_step) if next_global_step is not None else None
                logger.info(
                    "ENV: SUMO phase change detected tls=%s from=%s to=%s at sim; set phase_start_step=%s state=%s",
                    self.tls_id, prev, post_phase, self.phase_start_step, post_state
                )
        except Exception:
            logger.exception("ENV: failed to read phase after simulationStep()")

    # -------------------------
    # State extraction
    # -------------------------
    def get_state(self):
        try:
            # vehicle and pedestrian raw states
            veh = self.get_vehicle_state()
            ped_q, ped_w, ped_thru = self.get_ped_state()

            veh_thru = veh[-1] if veh else 0

            # unpack vehicle lists
            if len(veh) >= 1:
                n = (len(veh) - 1) // 2
                q_list = veh[:n]
                w_list = veh[n:2 * n]
            else:
                q_list, w_list = [], []

            # save previous values
            self.prev_veh_wait = self.veh_wait
            self.prev_ped_wait = self.ped_wait
            self.prev_veh_thru = self.veh_thru
            self.prev_ped_thru = self.ped_thru

            # update current
            self.veh_thru = veh_thru
            self.ped_thru = ped_thru
            self.veh_wait = float(sum(w_list)) if w_list else 0.0
            self.ped_wait = float(ped_w)

            phase = self.get_phase()

            # keep local last-phase seen in env; do NOT set phase_start_step here because env doesn't know global_step
            if self._last_phase_seen is None:
                self._last_phase_seen = phase
            elif self._last_phase_seen != phase:
                self._last_phase_seen = phase
                # phase_start_step will be set by record_phase_start_if_changed(next_global_step)
                # or by apply_action_safe (which should set phase_start_step = current_step + 1)

            state = q_list + [ped_q, phase]

            info = {
                "veh_q_list": q_list,
                "veh_w_list": w_list,
                "veh_thru": veh_thru,
                "ped_q": ped_q,
                "ped_w": ped_w,
                "ped_thru": ped_thru,
                "phase": phase,
            }

            return state, info
        except Exception:
            logger.exception("Exception in get_state()")
            raise

    def get_vehicle_state(self):
        try:
            current = set(traci.vehicle.getIDList())

            new = current - self.prev_veh_ids
            left = self.prev_veh_ids - current
            throughput = len(left)

            lane_q = {}
            lane_w = {}

            for vid in current:
                if vid in new:
                    continue
                try:
                    lane = traci.vehicle.getLaneID(vid)
                    wt = traci.vehicle.getWaitingTime(vid)
                    speed = traci.vehicle.getSpeed(vid)

                    if lane.endswith("_0"):
                        continue
                    if wt <= 1:
                        continue

                    if lane not in lane_q:
                        lane_q[lane] = 0
                        lane_w[lane] = 0.0

                    if speed < 0.1:
                        lane_q[lane] += 1
                        lane_w[lane] += wt
                except Exception:
                    logger.exception("Exception while processing vehicle id=%s", vid)
                    continue

            self.prev_veh_ids = current

            if not hasattr(self, "veh_lane_order"):
                raise RuntimeError("Vehicle lane order not initialized. Call initialize_from_sumo() first.")

            for l in self.veh_lane_order:
                if l not in lane_q:
                    lane_q[l] = 0
                    lane_w[l] = 0.0

            q_list = [lane_q[l] for l in self.veh_lane_order]
            w_list = [lane_w[l] for l in self.veh_lane_order]

            return q_list + w_list + [throughput]
        except Exception:
            logger.exception("Exception in get_vehicle_state()")
            raise

    def get_ped_state(self):
        try:
            current = set(traci.person.getIDList())

            new = current - self.prev_ped_ids
            left = self.prev_ped_ids - current
            throughput = len(left)

            ped_q = 0
            ped_w = 0.0

            for pid in current:
                if pid in new:
                    continue
                try:
                    wt = traci.person.getWaitingTime(pid)
                    speed = traci.person.getSpeed(pid)
                    if wt > 1:
                        ped_w += wt
                        if speed < 0.1:
                            ped_q += 1
                except Exception:
                    logger.exception("Exception while processing person id=%s", pid)
                    continue

            self.prev_ped_ids = current

            return ped_q, ped_w, throughput
        except Exception:
            logger.exception("Exception in get_ped_state()")
            raise

    def normalize_state_torch(self, states):
        try:
            if not torch.is_tensor(states):
                states = torch.as_tensor(states, dtype=torch.float32)

            single_input = False
            if states.dim() == 1:
                states = states.unsqueeze(0)
                single_input = True

            veh_vals = states[:, :-2]
            ped_q = states[:, -2]
            phase = states[:, -1].long()

            veh_norm = veh_vals / self.MAX_VEH_QUEUE
            ped_norm = (ped_q / self.MAX_PED_QUEUE).unsqueeze(1)

            num_phases = self.get_num_phases()

            phase_onehot = torch.zeros(
                (states.size(0), num_phases),
                device=states.device,
                dtype=states.dtype
            )
            phase_onehot.scatter_(1, phase.unsqueeze(1), 1.0)

            out = torch.cat([veh_norm, ped_norm, phase_onehot], dim=1)

            if single_input:
                return out.squeeze(0)
            return out
        except Exception:
            logger.exception("Exception in normalize_state_torch()")
            raise

    def get_reward(self, state, prev_state=None, action=None):
        try:
            *veh_vals, ped_q, phase = state

            veh_total = float(sum(veh_vals))
            veh_norm = veh_total / (self.MAX_VEH_QUEUE)
            ped_norm = float(ped_q) / (self.MAX_PED_QUEUE)
            current_congestion = veh_norm + ped_norm

            if prev_state is not None:
                *prev_veh_vals, prev_ped_q, prev_phase = prev_state
                prev_veh_total = float(sum(prev_veh_vals))
                prev_veh_norm = prev_veh_total / (self.MAX_VEH_QUEUE)
                prev_ped_norm = float(prev_ped_q) / (self.MAX_PED_QUEUE)
                prev_congestion = prev_veh_norm + prev_ped_norm
            else:
                prev_congestion = current_congestion

            delta = prev_congestion - current_congestion

            reward = -current_congestion


            return float(reward)
        except Exception:
            logger.exception("Exception in get_reward()")
            raise


# Helper functions (unchanged except for logging)
def make_sumo_config(
    cfg_path=None,
    step_length="0.50",
    lateral_res="0",
    gui=False,
):
    try:
        cur = os.path.abspath(os.path.dirname(__file__))
        while True:
            candidate = os.path.join(cur, "simulation", "sumo", "test.sumocfg")
            if os.path.exists(candidate):
                PROJECT_ROOT = cur
                break
            parent = os.path.abspath(os.path.join(cur, ".."))
            if parent == cur:
                raise RuntimeError("Could not locate simulation/sumo/test.sumocfg")
            cur = parent

        if cfg_path is None:
            cfg_path = os.path.join(PROJECT_ROOT, "simulation", "sumo", "test.sumocfg")

        cfg_path = os.path.normpath(os.path.abspath(cfg_path))

        binary = "sumo-gui" if gui else "sumo"

        cmd = [
            binary,
            "-c", cfg_path,
            "--step-length", str(step_length),
            "--lateral-resolution", str(lateral_res),
            "--random"
        ]

        return cmd
    except Exception:
        logger.exception("Exception in make_sumo_config()")
        raise


def one_hot_phase(phase_idx: int, num_phases: int) -> np.ndarray:
    vec = np.zeros(num_phases, dtype=np.float32)
    if 0 <= phase_idx < num_phases:
        vec[phase_idx] = 1.0
    return vec


def save_step_csv(step_rows, path):
    try:
        if not step_rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=step_rows[0].keys())
            writer.writeheader()
            writer.writerows(step_rows)
    except Exception:
        logger.exception("Exception in save_step_csv(%s)", path)


def save_episode_csv(episode_rows, path):
    try:
        if not episode_rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=episode_rows[0].keys())
            writer.writeheader()
            writer.writerows(episode_rows)
    except Exception:
        logger.exception("Exception in save_episode_csv(%s)", path)


def plot_metrics(episode_rows, path, MODE=None, RUN_ID=None):
    try:
        if not episode_rows:
            return
        eps = [r["episode"] for r in episode_rows]
        cum_rewards = [r["cumulative_reward"] for r in episode_rows]
        avg_veh_q = [r["avg_vehicle_queue"] for r in episode_rows]
        avg_ped_q = [r["avg_ped_queue"] for r in episode_rows]

        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot(eps, cum_rewards, marker="o")
        plt.xlabel("Episode")
        plt.ylabel("Cumulative Reward")
        mode_str = MODE.upper() if MODE else "Mode: N/A"
        run_str = RUN_ID if RUN_ID else "RUN_ID: N/A"
        plt.title(f"Episode Metrics ({mode_str} - {run_str})")
        plt.grid(True)

        plt.subplot(2, 1, 2)
        plt.plot(eps, avg_veh_q, marker="o", label="Avg Vehicle Queue")
        plt.plot(eps, avg_ped_q, marker="o", label="Avg Pedestrian Queue")
        plt.xlabel("Episode")
        plt.ylabel("Avg Queues")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        logger.exception("Exception in plot_metrics(%s)", path)
