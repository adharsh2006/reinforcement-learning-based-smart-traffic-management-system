# src/environment.py (Faster Reloading Version)
import os
import sys
import numpy as np
from src.config import *

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
import traci
import traci.exceptions

class SumoEnvironment:
    def __init__(self, use_gui=True):
        self.use_gui = use_gui
        if self.use_gui:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo-gui')
        else:
            self.sumo_binary = os.path.join(os.environ['SUMO_HOME'], 'bin', 'sumo')
        
        self.sumo_cmd = [
            self.sumo_binary, "-c", SUMO_CONFIG_FILE,
            "--log-file", "sumo.log",
            "--error-log", "sumo-errors.log"
        ]
        self.state_size = STATE_SIZE
        
        # Start SUMO ONCE when the environment is created
        traci.start(self.sumo_cmd)

    def close(self):
        # This will be called only once at the very end of training
        traci.close()

    def reset(self):
        # Reload the simulation with the same configuration to reset it
        traci.load(["-c", SUMO_CONFIG_FILE, "--begin", "0"])
        self.step_count = 0
        return self._get_state()

    def step(self, action):
        self._set_traffic_light_phase(action)
        for _ in range(PHASE_DURATION):
            traci.simulationStep()
        self._set_yellow_phase(action)
        for _ in range(YELLOW_DURATION):
            traci.simulationStep()

        self.step_count += (PHASE_DURATION + YELLOW_DURATION)
        
        next_state = self._get_state()
        reward = self._calculate_reward()
        done = self.step_count >= SIM_DURATION

        return next_state, reward, done

    # --- Helper methods _get_state, _calculate_reward, etc. remain the same ---
    def _get_state(self):
        state = []
        for lane in INCOMING_LANES:
            queue_length = traci.lane.getLastStepHaltingNumber(lane)
            waiting_time = traci.lane.getWaitingTime(lane)
            state.extend([queue_length, waiting_time])
        return np.reshape(state, [1, self.state_size])

    def _calculate_reward(self):
        total_waiting_time = sum(traci.lane.getWaitingTime(lane) for lane in INCOMING_LANES)
        return -total_waiting_time

    def _set_traffic_light_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 0)
        elif action == 1:
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 2)
            
    def _set_yellow_phase(self, action):
        if action == 0:
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 1)
        elif action == 1:
            traci.trafficlight.setPhase(TRAFFIC_LIGHT_ID, 3)