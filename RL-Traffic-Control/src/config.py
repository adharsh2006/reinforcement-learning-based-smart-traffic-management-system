# src/config.py

# --- AGENT ---
STATE_SIZE = 8       # Number of state features (4 lanes * 2 values: queue + waiting_time)
ACTION_SIZE = 2      # Number of possible actions (phases for the traffic light)
LEARNING_RATE = 0.001
GAMMA = 0.95         # Discount factor
EPSILON = 1.0        # Exploration rate
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 2000
BATCH_SIZE = 64

# --- TRAINING ---
NUM_EPISODES = 2

# --- SUMO ENVIRONMENT ---
SUMO_CONFIG_FILE = "sumo_files/config.sumocfg"
SIM_DURATION = 600  # seconds
PHASE_DURATION = 10  # seconds
YELLOW_DURATION = 3  # seconds
TRAFFIC_LIGHT_ID = "J1"
INCOMING_LANES = ["L_to_J1_0", "L_to_J1_1", "T1_to_J1_0", "B1_to_J1_0"]