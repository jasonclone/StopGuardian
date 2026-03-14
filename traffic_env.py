# traffic_env.py
"""
Reinforcement Learning Environment for Traffic Signal Control at Junction C.

This environment matches the fixed-time baseline exactly:
- Same reward: -(vehicle_queue + ped_queue)
- Same queue definitions
- Same waiting-time accumulation
- Same episode length (10,000 steps)
- Same SUMO step-length (0.1 seconds)

The RL agent has ONLY TWO ACTIONS:
    0 = KEEP (let SUMO continue the current phase)
    1 = NEXT (advance to the next phase, but ONLY after minimum duration)

Pedestrian safety is handled entirely by SUMO’s built-in phase structure.
We do NOT override pedestrian phases or create custom ped actions.
"""

import time
from typing import Dict, Any, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import traci



# Simulation Configuration


SUMO_CFG = "simulation/sumo/test.sumocfg"

# IMPORTANT:
# SUMO must run with --step-length equal to STEP_LENGTH.
# Your .sumocfg already sets this correctly.
STEP_LENGTH = 0.1

# One episode = 10,000 steps (1000 seconds)
SIM_STEPS = 10000

TL_ID = "C"

# Minimum durations for each of the 14 phases (from your XML)
MIN_PHASE_DURATION = [
    15, 10, 5, 2,
    10, 5, 2,
    15, 10, 5, 2,
    10, 5, 2
]



# Utility Functions


def wait_for_tl(tl_id: str, timeout: float = 15.0) -> bool:
    """
    Wait until SUMO reports that the traffic light exists.
    This avoids race conditions where TraCI connects before SUMO loads the network.
    """
    start = time.time()
    while time.time() - start < timeout:
        try:
            if tl_id in traci.trafficlight.getIDList():
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def get_vehicle_queue(lane: str, new_vehicles) -> int:
    """
    Count vehicles that are:
        - stopped (speed < 0.1)
        - have waitingTime > 0
    This matches the baseline queue definition exactly.
    """
    queue = 0
    for vid in traci.lane.getLastStepVehicleIDs(lane):
        if vid in new_vehicles:
            continue
        try:
            if traci.vehicle.getSpeed(vid) < 0.1 and traci.vehicle.getWaitingTime(vid) > 0:
                queue += 1
        except Exception:
            # SUMO sometimes throws transient errors; safe to ignore.
            continue
    return queue


def get_ped_queue(ped_ids, new_peds) -> int:
    """
    Count pedestrians that are:
        - stopped (speed < 0.1)
        - have waitingTime > 0
    Matches baseline definition.
    """
    queue = 0
    for pid in ped_ids:
        if pid in new_peds:
            continue
        try:
            if traci.person.getSpeed(pid) < 0.1 and traci.person.getWaitingTime(pid) > 0:
                queue += 1
        except Exception:
            continue
    return queue



# RL Environment


class TrafficEnv(gym.Env):
    """
    Gymnasium-compatible RL environment for controlling TLS C.

    ACTIONS:
        0 = KEEP (do nothing, let SUMO continue)
        1 = NEXT (advance to next phase, but only after min duration)

    The agent CANNOT:
        - skip safety phases
        - activate pedestrian phases manually
        - override yellow/all-red
        - modify pedestrian timing

    SUMO handles all safety transitions automatically.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode: str = None):
        super().__init__()

        self.render_mode = render_mode

        # Ensure consistency with SUMO's --step-length
        self.step_length = STEP_LENGTH

        # Episode length
        self.max_steps = SIM_STEPS

        # Two actions: KEEP or NEXT
        self.action_space = spaces.Discrete(2)

        # Observation vector bounds
        high = np.array([
            200.0,   # vehicle_queue
            200.0,   # ped_queue
            1e4,     # total_vehicle_wait
            1e4,     # total_ped_wait
            500.0,   # vehicle_count
            500.0,   # ped_count
            13.0     # current_phase_index
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=np.zeros_like(high),
            high=high,
            dtype=np.float32
        )

        # Internal state
        self.vehicle_lanes = []
        self.veh_accum_wait: Dict[str, float] = {}
        self.ped_accum_wait: Dict[str, float] = {}

        self.current_phase = 0
        self.phase_elapsed = 0.0

        self.step_count = 0
        self.cumulative_reward = 0.0


    
    # SUMO Control
    

    def _start_sumo(self):
        """
        Launch SUMO and initialize lane lists and phase state.
        """
        args = [
            "sumo-gui" if self.render_mode == "human" else "sumo",
            "-c", SUMO_CFG,
            "--start",
            "--no-step-log",
            "--step-length", str(self.step_length)
        ]
        traci.start(args)

        if not wait_for_tl(TL_ID):
            raise RuntimeError(f"Traffic light '{TL_ID}' not found.")

        # Identify vehicle lanes (exclude pedestrian-only lanes)
        incoming_lanes = list(set(traci.trafficlight.getControlledLanes(TL_ID)))
        self.vehicle_lanes = [
            lane for lane in incoming_lanes
            if "pedestrian" not in traci.lane.getAllowed(lane)
        ]

        # Initialize phase tracking
        self.current_phase = traci.trafficlight.getPhase(TL_ID)
        self.phase_elapsed = 0.0


    def _close_sumo(self):
        """Safely close TraCI."""
        try:
            traci.close()
        except Exception:
            pass


    
    # Gym API
    

    def reset(self, *, seed=None, options=None):
        """
        Reset SUMO and return the first observation.
        """
        super().reset(seed=seed)

        self._close_sumo()
        self._start_sumo()

        self.veh_accum_wait.clear()
        self.ped_accum_wait.clear()

        self.step_count = 0
        self.cumulative_reward = 0.0

        obs = self._compute_observation()
        return obs, {}


    def step(self, action: int):
        """
        One RL step = one SUMO simulation step (0.1 seconds).
        """
        self.step_count += 1

        # Track agents before step
        prev_vehicles = set(traci.vehicle.getIDList())
        prev_peds = set(traci.person.getIDList())

        # Advance SUMO by STEP_LENGTH seconds
        traci.simulationStep()

        # Track agents after step
        current_vehicles = set(traci.vehicle.getIDList())
        current_peds = set(traci.person.getIDList())

        new_vehicles = current_vehicles - prev_vehicles
        new_peds = current_peds - prev_peds

        exited_vehicles = prev_vehicles - current_vehicles
        exited_peds = prev_peds - current_peds

        # Initialize new agents
        for vid in new_vehicles:
            self.veh_accum_wait[vid] = 0.0
        for pid in new_peds:
            self.ped_accum_wait[pid] = 0.0

        # Update accumulated waiting time
        for vid in current_vehicles:
            try:
                if traci.vehicle.getSpeed(vid) < 0.1 and traci.vehicle.getWaitingTime(vid) > 0:
                    self.veh_accum_wait[vid] += self.step_length
            except Exception:
                pass

        for pid in current_peds:
            try:
                if traci.person.getWaitingTime(pid) > 0:
                    self.ped_accum_wait[pid] += self.step_length
            except Exception:
                pass

        # Remove exited agents
        for vid in exited_vehicles:
            self.veh_accum_wait.pop(vid, None)
        for pid in exited_peds:
            self.ped_accum_wait.pop(pid, None)

        # Compute queues
        vehicle_queue = sum(get_vehicle_queue(l, new_vehicles) for l in self.vehicle_lanes)
        ped_queue = get_ped_queue(current_peds, new_peds)

        # Compute total waits
        total_vehicle_wait = sum(self.veh_accum_wait.values())
        total_ped_wait = sum(self.ped_accum_wait.values())

        # Reward matches baseline exactly
        reward = -(vehicle_queue + ped_queue)
        self.cumulative_reward += reward

        # Phase tracking
        phase_now = traci.trafficlight.getPhase(TL_ID)
        if phase_now != self.current_phase:
            # SUMO advanced automatically
            self.current_phase = phase_now
            self.phase_elapsed = 0.0
        else:
            self.phase_elapsed += self.step_length

        # Apply RL action only after minimum duration
        min_dur = MIN_PHASE_DURATION[self.current_phase]
        if self.phase_elapsed >= min_dur:
            if action == 1:  # NEXT
                next_phase = (self.current_phase + 1) % len(MIN_PHASE_DURATION)
                traci.trafficlight.setPhase(TL_ID, next_phase)
                self.current_phase = next_phase
                self.phase_elapsed = 0.0
            # KEEP = do nothing

        # Build observation
        obs = np.array([
            float(vehicle_queue),
            float(ped_queue),
            float(total_vehicle_wait),
            float(total_ped_wait),
            float(len(current_vehicles)),
            float(len(current_peds)),
            float(self.current_phase)
        ], dtype=np.float32)

        # Episode termination
        terminated = False
        truncated = self.step_count >= self.max_steps

        info = {"cumulative_reward": self.cumulative_reward}

        return obs, float(reward), terminated, truncated, info


    def _compute_observation(self):
        """
        Warm-up step to avoid returning all zeros at reset.
        This matches the baseline behavior more closely.
        """
        prev_vehicles = set(traci.vehicle.getIDList())
        prev_peds = set(traci.person.getIDList())

        traci.simulationStep()

        current_vehicles = set(traci.vehicle.getIDList())
        current_peds = set(traci.person.getIDList())

        new_vehicles = current_vehicles - prev_vehicles
        new_peds = current_peds - prev_peds

        for vid in new_vehicles:
            self.veh_accum_wait[vid] = 0.0
        for pid in new_peds:
            self.ped_accum_wait[pid] = 0.0

        vehicle_queue = sum(get_vehicle_queue(l, new_vehicles) for l in self.vehicle_lanes)
        ped_queue = get_ped_queue(current_peds, new_peds)

        total_vehicle_wait = sum(self.veh_accum_wait.values())
        total_ped_wait = sum(self.ped_accum_wait.values())

        obs = np.array([
            float(vehicle_queue),
            float(ped_queue),
            float(total_vehicle_wait),
            float(total_ped_wait),
            float(len(current_vehicles)),
            float(len(current_peds)),
            float(self.current_phase)
        ], dtype=np.float32)

        return obs


    def render(self):
        """SUMO-GUI already provides visualization."""
        pass

    def close(self):
        self._close_sumo()
