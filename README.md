# Reinforcement Learning for Traffic Signal Optimization 🚦

## Project Overview

This project implements a Deep Reinforcement Learning agent to control traffic signal timing in a simulated urban environment. The agent, built with TensorFlow and Keras, learns an optimal policy to minimize vehicle waiting times by analyzing traffic density in real-time. The simulation is powered by **SUMO (Simulation of Urban MObility)**.

The agent uses a **Double Deep Q-Network (Double DQN)** algorithm, an advanced technique that provides more stable and robust learning compared to a standard DQN.



---

## Features

-   **Intelligent Agent:** A Double DQN agent that learns and adapts to traffic conditions.
-   **Modular Codebase:** The project is structured with separate modules for the agent, environment, and configuration, making it easy to understand and extend.
-   **Realistic Simulation:** A multi-intersection SUMO environment with varied traffic flows, including cars, trucks, and buses.
-   **Performance Tracking:** Automatically saves training rewards to a CSV file and generates a performance plot upon completion.
-   **Persistent Models:** Saves trained model weights periodically with unique timestamps, so no progress is ever lost.

---

## Project Structure

The repository is organized into a clean and professional structure:

```
RL-Traffic-Control-Pro/
├── logs/              # Stores output files like rewards.csv and plots
├── models/            # Stores saved model weights (.weights.h5 files)
├── src/               # Contains all the Python source code
│   ├── agent.py       # The Double DQN agent class
│   ├── config.py      # All hyperparameters and settings
│   ├── environment.py # The SUMO environment wrapper
│   └── main.py        # The main script to run training
└── sumo_files/        # Contains all SUMO simulation files (.xml, .sumocfg)
```

---

## Prerequisites

Before you begin, ensure you have the following installed:

1.  **SUMO:** Download from the [official SUMO website](https://www.eclipse.org/sumo/docs/Downloads.php).
    -   **Crucial:** During installation, ensure you add SUMO to your system's `PATH`.
2.  **Python 3.8+:** Download from the [Python website](https://www.python.org/downloads/).
3.  A Python virtual environment is highly recommended to manage dependencies.

---

## Setup and Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/RL-Traffic-Control-Pro.git](https://github.com/your-username/RL-Traffic-Control-Pro.git)
    cd RL-Traffic-Control-Pro
    ```

2.  **Create and Activate a Virtual Environment:**
    ```bash
    # Create the environment
    python -m venv .venv

    # Activate it (Windows)
    .\.venv\Scripts\activate

    # Activate it (macOS/Linux)
    source .venv/bin/activate
    ```

3.  **Install Required Python Libraries:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```

4.  **Generate the SUMO Network File:**
    This step is required to build the `network.net.xml` file that SUMO uses.
    ```bash
    netconvert --node-files=sumo_files/network.nod.xml --edge-files=sumo_files/network.edg.xml --output-file=sumo_files/network.net.xml
    ```

---

## How to Run the Project

### Training the Agent

To start the training process, run the main script from the project's root directory:

```bash
python -m src.main
```

By default, this runs in **headless mode** (`use_gui=False`) for maximum speed. You will see the progress printed in your terminal after each episode.

### Visualizing the Training

If you want to watch the simulation live, you need to enable the GUI.

1.  Open `src/main.py`.
2.  Change the line `env = SumoEnvironment(use_gui=False)` to `env = SumoEnvironment(use_gui=True)`.
3.  Run the script again. The SUMO GUI window will appear.

---

## Configuration

All key hyperparameters for the agent and the simulation can be easily modified in the **`src/config.py`** file, including:

-   `NUM_EPISODES`: The total number of episodes to train for.
-   `SIM_DURATION`: The length of each episode in simulation seconds.
-   `LEARNING_RATE`, `GAMMA`, `EPSILON`: Core parameters for the RL agent.

---

## Output and Results

After a training run is complete, the script will generate the following files with a unique timestamp:

-   **Models:** Saved model weights will be in the `models/` folder (e.g., `run_TIMESTAMP_episode_25.weights.h5`).
-   **CSV Data:** A CSV file with the reward for each episode will be saved in the `logs/` folder (e.g., `rewards_TIMESTAMP.csv`).
-   **Plot Image:** A PNG image of the training graph will be saved in the `logs/` folder (e.g., `plot_TIMESTAMP.png`).

You can use the CSV file in any spreadsheet software to create your own visualizations of the agent's learning progress.
## Author

-   **Owner:** ADHARSH V S
-   **Email:** vsadharsh0@gmail.com
