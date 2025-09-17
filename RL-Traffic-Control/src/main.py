# src/main.py (Upgraded Version with Timestamping and CSV Export)

import sys
import csv
import datetime
import numpy as np
import matplotlib.pyplot as plt
from src.agent import DQNAgent
from src.environment import SumoEnvironment
from src.config import *

if __name__ == "__main__":
    env = SumoEnvironment(use_gui=False) # Recommend headless for long training
    agent = DQNAgent()
    
    episode_rewards = []
    
    # --- Generate a unique timestamp for this training run ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"Starting new training run with ID: {timestamp}")
    # -----------------------------------------------------------

    # Main training loop
    for e in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            total_reward += reward
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay()
        
        agent.update_target_model()
        episode_rewards.append(total_reward)
        
        print(f"Episode: {e+1}/{NUM_EPISODES}, Total Reward: {total_reward:.2f}, Epsilon: {agent.epsilon:.2f}")

        # --- Save model with a unique timestamp ---
        if e % 5 == 0 and e > 0:
            model_filename = f"models/run_{timestamp}_episode_{e}.weights.h5"
            agent.save(model_filename)
            print(f"Model saved: {model_filename}")
        # ---------------------------------------------

    env.close()

    # --- Save results to both a CSV file and a plot image ---
    print("\nTraining finished. Saving results...")
    
    # 1. Save to CSV
    csv_file_path = f'logs/rewards_{timestamp}.csv'
    with open(csv_file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Episode', 'TotalReward'])
        for i, reward in enumerate(episode_rewards):
            writer.writerow([i + 1, reward])
    print(f"Reward data saved to {csv_file_path}")

    # 2. Save Plot Image
    plot_file_path = f'logs/plot_{timestamp}.png'
    plt.figure(figsize=(10, 6))
    plt.plot(episode_rewards)
    plt.title(f'Training Rewards per Episode (Run ID: {timestamp})')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)
    plt.savefig(plot_file_path)
    print(f"Final plot saved to {plot_file_path}")
    # ----------------------------------------------------------------

    print("\nProcess complete. Exiting.")
    sys.exit(0)