# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 20:56:28 2025

@author: Shubhendu
"""
# Importing necessary libraries
import os
import numpy as np
import pandas as pd
import torch as th
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import PPO
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('/content/Clipped_NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow_af = data['Inflow** (cfs)'].values * 1.98211  # Convert to AF
release_af = data['Total Release (cfs)'].values * 1.98211  # Convert to AF

# Split training data
train_storage = storage[:18000]
train_evaporation = evaporation[:18000]
train_inflow = inflow_af[:18000]
train_release = release_af[:18000]

# Calculate Mean and Std for Training Data
means = np.array([np.mean(train_storage), np.mean(train_evaporation), np.mean(train_inflow), np.mean(train_release)])
stds = np.array([np.std(train_storage), np.std(train_evaporation), np.std(train_inflow), np.std(train_release)])

# Normalize Function
def normalize(data, mean, std):
    return (data - mean) / std

# Normalize All Data
storage = normalize(storage, means[0], stds[0])
evaporation = normalize(evaporation, means[1], stds[1])
inflow = normalize(inflow_af, means[2], stds[2])
release = normalize(release_af, means[3], stds[3])

# Combined Dataset
dataset = np.column_stack([storage, evaporation, inflow, release])
train_data = dataset[:18000]
test_data = dataset[18000:]

# Define storage and release limits
min_storage_af = normalize(500000, means[0], stds[0])
max_storage_af = normalize(1731750, means[0], stds[0])
min_release_af = normalize(0*1.98211, means[3], stds[3])
max_release_af = normalize(5000*1.98211, means[3], stds[3])

# Define the custom Reservoir environment
class ReservoirEnv(Env):
    def __init__(self, dataset, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length):
        super(ReservoirEnv, self).__init__()
        self.data = dataset
        self.episode_length = episode_length
        self.min_storage_af = min_storage_af
        self.max_storage_af = max_storage_af
        self.min_release_af = min_release_af
        self.max_release_af = max_release_af
        self.reward_history, self.episode_reward = [], []
        self.storage_history, self.mean_storage_history = [], []
        self.release_history, self.mean_release_history = [], []
        self.current_step, self.episode_step_count = 0, 0

        # Define action space and observation space
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=np.array([-np.inf,-np.inf,-np.inf]),
                                     high=np.array([np.inf,np.inf,np.inf]),
                                     dtype=np.float32)

    def reset(self, seed=None):
        self.current_step = np.random.randint(0, len(self.data) - self.episode_length)  # Random episode start
        self.episode_step_count = 0
        self.reward_history.clear()
        self.storage_history.clear()
        self.release_history.clear()
        self.current_storage = self.data[self.current_step, 0]
        self.prev_release = None  # Reset previous release
        return self._get_observation(), {}

    def _get_observation(self):
        return np.array([self.current_storage, self.data[self.current_step, 1], self.data[self.current_step, 2]])

    def step(self, actions):
        # Extract actions
        normalized_regular_release = (actions[0] + 1) / 2
        normalized_fisheries_release = (actions[1] + 1) / 2

        # Convert normalized actions to actual release values
        regular_release_af = self.min_release_af + normalized_regular_release * (self.max_release_af - self.min_release_af)
        fisheries_release_af = self.min_release_af + normalized_fisheries_release * (self.max_release_af - self.min_release_af)

        # Total release
        total_release_af = regular_release_af + fisheries_release_af

        inflow_af = self.data[self.current_step, 2]
        evaporation_af = self.data[self.current_step, 1]
        self.current_storage = (self.current_storage*stds[0]+means[0])+((inflow_af*stds[2]+means[2]) - (evaporation_af*stds[1]+means[1]) - (total_release_af*stds[3]+means[3]))
        self.current_storage = (self.current_storage-means[0])/stds[0]

        # Compute Reward
        if self.min_storage_af <= self.current_storage <= self.max_storage_af:
          target_storage = (self.max_storage_af + self.min_storage_af) / 2
          reward = 1 - abs(self.current_storage - target_storage) / (self.max_storage_af - self.min_storage_af)
        else:
          reward = -1

        # Penalize for total release exceeding 5000 cfs
        if total_release_af >= max_release_af:
            reward -= 0.8
        else:
            reward += 0.5

        # Penalize for fisheries releasing below 350 cfs
        fisheries_threshold_af = 350 * 1.98211
        fisheries_threshold_normalized = (fisheries_threshold_af - means[3]) / stds[3]
        # Apply reward/penalty only to fisheries release
        if fisheries_release_af >= fisheries_threshold_normalized:
            reward += 0.5
        else:
            reward -= 0.2

        self.prev_release = total_release_af
        self.reward_history.append(reward)
        self.storage_history.append(self.current_storage)
        self.current_step += 1
        self.episode_step_count += 1
        done = self.current_step >= len(self.data) or (self.episode_step_count >= self.episode_length)

        if done:
            self.episode_reward.append(sum(self.reward_history))
            self.mean_storage_history.append(np.mean(self.storage_history))
            self.release_history.append(total_release_af)
            self.mean_release_history.append(np.mean(self.release_history))

        observation = self._get_observation() if not done else np.zeros(3)
        return observation, reward, done, False, {}

# Set up environment
episode_length = 3600
env = ReservoirEnv(train_data, min_storage_af, max_storage_af, min_release_af, max_release_af, episode_length)

# Initialize models for both agents
regular_agent = PPO("MlpPolicy", env, verbose=0)
fisheries_agent = PPO("MlpPolicy", env, verbose=0)

# Training parameters
n_episodes = 500
total_timesteps = n_episodes * episode_length

# Custom training loop
for episode in range(n_episodes):
    obs, _ = env.reset()
    done = False
    while not done:
        # Each agent selects an action based on the shared observation
        action_regular, _ = regular_agent.predict(obs, deterministic=False)
        action_fisheries, _ = fisheries_agent.predict(obs, deterministic=False)

        # Combine actions
        actions = np.array([action_regular[0], action_fisheries[0]])

        # Step the environment
        next_obs, reward, done, _, _ = env.step(actions)

        # Store transitions and train both agents
        regular_agent.replay_buffer.add(obs, action_regular, reward, next_obs, done, {})
        fisheries_agent.replay_buffer.add(obs, action_fisheries, reward, next_obs, done, {})

        obs = next_obs

    # Update both agents
    regular_agent.train()
    fisheries_agent.train()

# Save models
regular_agent.save("ppo_regular_agent")
fisheries_agent.save("ppo_fisheries_agent")

# Save cumulative reward values to CSV
pd.DataFrame(env.episode_reward).to_csv("cumulative_reward_per_episode.csv", index=False)

# Plot cumulative rewards per episode
plt.figure(figsize=(12, 6))
plt.plot(env.episode_reward, label="Cumulative Reward per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("cumulative_rewards_per_episode.png")
plt.show()

# Plot mean storage per episode
plt.figure(figsize=(12, 6))
plt.plot(env.mean_storage_history, label="Mean Storage per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Mean Storage")
plt.title("Mean Storage per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("mean_storage_per_episode.png")
plt.show()

# Plot mean release per episode
plt.figure(figsize=(12, 6))
plt.plot(env.mean_release_history, label="Mean Release per Episode", color="blue")
plt.xlabel("Episode")
plt.ylabel("Mean Release")
plt.title("Mean Release per Episode vs Episode Number")
plt.grid()
plt.legend()
plt.savefig("mean_release_per_episode.png")
plt.show()

print("Training complete. Model, Storage, Release, and Reward plots saved.")