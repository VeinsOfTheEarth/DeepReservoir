# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 14:06:56 2024

@author: Shubhendu
"""
import os
import numpy as np
import pandas as pd
import torch as th
import gymnasium
from gymnasium import Env
from gymnasium.spaces import Box
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Define reservoir release and storage limits
navajo_min_release = 250
navajo_max_release = 5000
min_storage_af = 500000
max_storage_af = 1900000

# Define the Reservoir environment
class ReservoirEnv(Env):
    def __init__(self, data, scaler):
        super(ReservoirEnv, self).__init__()

        self.data = data
        self.scaler = scaler
        self.current_step = 0
        self.storage_history = []
        self.predicted_release_history = []
        self.reward_history = []

        # Extract and store the original unscaled storage and release values
        unscaled_data = self.scaler.inverse_transform(self.data)
        self.actual_storage_history = unscaled_data[:, 0].tolist()  # Unscaled storage column
        self.actual_release_history = unscaled_data[:, 3].tolist()  # Unscaled release column

        # Define action and observation space
        self.action_space = Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = Box(low=0, high=1, shape=(3,), dtype=np.float32)  # Scaled storage, evaporation, inflow

    def reset(self, seed=None):
        self.current_step = 0
        self.storage_history.clear()
        self.predicted_release_history.clear()
        self.reward_history.clear()

        # Unscale initial storage value to use in calculations
        self.current_storage = self.scaler.inverse_transform(self.data[self.current_step:self.current_step+1, :])[0, 0]

        # Record initial storage
        self.storage_history.append(self.current_storage)
        info = {}
        return self._get_observation(), info

    def _get_observation(self):
        # Return the current day's scaled storage, evaporation, inflow
        obs = self.data[self.current_step, :3]  # Already scaled
        return obs

    def step(self, action):
        # Scale predicted release to actual range
        release_scaled = action[0]
        release_af = navajo_min_release + release_scaled * (navajo_max_release - navajo_min_release)
        self.predicted_release_history.append(release_af)

        # Unscale inflow and evaporation for accurate calculations
        inflow_af = self.scaler.inverse_transform(self.data[self.current_step:self.current_step+1, :])[0, 2]  # Original inflow in af
        evaporation_af = self.scaler.inverse_transform(self.data[self.current_step:self.current_step+1, :])[0, 1]  # Original evaporation in af

        # Update storage with current release, inflow, and evaporation, ensuring non-negative storage
        # self.current_storage = max(0, self.current_storage - release_af + inflow_af - evaporation_af)
        self.current_storage = self.current_storage - release_af + inflow_af - evaporation_af

        # Enforce maximum and minimum storage limits
        self.current_storage = min(max(self.current_storage, min_storage_af), max_storage_af)
        self.storage_history.append(self.current_storage)

        # Print predicted storage for debugging
        print(f"Step {self.current_step}: Predicted Storage = {self.current_storage}")

        # Reward based on release within acceptable range and storage limits
        release_reward = -0.1 * abs(release_af - np.clip(release_af, navajo_min_release, navajo_max_release))

        # Additional reward for keeping storage within range
        if min_storage_af <= self.current_storage <= max_storage_af:
            storage_reward = 1 - (abs(self.current_storage - (min_storage_af + max_storage_af) / 2) / (max_storage_af - min_storage_af))
        else:
            storage_reward = -2.0  # Heavy penalty for out-of-bound storage

        # Total reward
        reward = release_reward + storage_reward
        self.reward_history.append(reward)

        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        observation = self._get_observation() if not done else np.zeros(3)
        truncated = False
        info = {}

        return observation, reward, done, truncated, info

    def render(self, mode='human', close=False):
        pass

# Load data
data = pd.read_csv('/content/NAVAJORESERVOIR08-18-2024T16.48.23.csv')
storage = data['Storage (af)'].values
evaporation = data['Evaporation (af)'].values
inflow = data['Inflow** (cfs)'].values
release = data['Total Release (cfs)'].values

# Combine data and scale
dataset = np.column_stack((storage, evaporation, inflow, release))
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(dataset)

# Initialize the environment
env = ReservoirEnv(scaled_data, scaler)
env = DummyVecEnv([lambda: env])

# Set up TD3 with noise for exploration
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.15 * np.ones(n_actions))
model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=1)

# Train the model and collect rewards
total_timesteps = 68000
model.learn(total_timesteps=total_timesteps)

# Save the model
model.save("td3_reservoir_agent")
model = TD3.load("td3_reservoir_agent")

# # Testing the model and collecting storage and release predictions
# obs, _ = env.reset()
# for i in range(len(scaled_data) - 1):
#     action, _ = model.predict(obs)
#     obs, rewards, done, _, _ = env.step(action)
#     if done:
#         break

# Plot storage, release, and rewards after training
plt.figure(figsize=(14, 10))

# Plot storage comparison
plt.subplot(3, 1, 1)
plt.plot(env.envs[0].storage_history, label="Predicted Storage", color='blue')
plt.plot(env.envs[0].actual_storage_history, label="Actual Storage", color='red', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Storage (af)')
plt.title('Predicted vs Actual Storage over Training Period')
plt.legend(loc='upper right')

# Plot release comparison
plt.subplot(3, 1, 2)
plt.plot(env.envs[0].predicted_release_history, label="Predicted Release", color='purple')
plt.plot(env.envs[0].actual_release_history, label="Actual Release", color='orange', linestyle='--')
plt.xlabel('Days')
plt.ylabel('Release (af)')
plt.title('Predicted vs Actual Release over Training Period')
plt.legend(loc='upper right')

# Plot rewards
plt.subplot(3, 1, 3)
plt.plot(env.envs[0].reward_history, label="Daily Rewards", color='green')
plt.xlabel('Days')
plt.ylabel('Reward')
plt.title('Rewards over Training Period')
plt.legend(loc='upper right')

plt.tight_layout()
plt.show()